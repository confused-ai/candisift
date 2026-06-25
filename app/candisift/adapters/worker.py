"""Durable background worker — drives the application from the queue inward.

A driving adapter: it pulls persisted tasks and calls the use-case. Runs in a
daemon thread started by the web app (so `uvicorn app:app` gives you API + UI +
worker in one process), or standalone via `python -m ats.adapters.worker`.

Crash safety: on start it reclaims tasks whose lease expired (a previous process
died mid-run). Each task is leased; on success -> done, on exception -> retried up
to max_attempts, then dead-lettered (status=failed).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback

from app.candisift.domain import ports
from app.candisift.domain.models import TaskType
from app.candisift.application.screening_service import ScreeningService, PermanentTaskError

log = logging.getLogger("candisift.worker")


class Worker:
    def __init__(
        self,
        queue: ports.TaskQueue,
        service: ScreeningService,
        *,
        lease_seconds: int = 300,
        idle_sleep: float = 1.0,
        reclaim_every: float = 30.0,
        concurrency: int = 1,
    ) -> None:
        self._queue = queue
        self._service = service
        self._lease = lease_seconds
        self._idle = idle_sleep
        self._reclaim_every = reclaim_every
        self._concurrency = max(1, int(concurrency))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._handlers = {
            TaskType.ingest_resume: service.handle_ingest_task,
            TaskType.screen: service.handle_screen_task,
        }

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        n = self._queue.reclaim_expired()
        if n:
            log.info("reclaimed %d orphaned task(s) on startup", n)
        # N daemon threads; claim_next is atomic so they never double-claim.
        for i in range(self._concurrency):
            t = threading.Thread(target=self._loop, name=f"ats-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("worker started (%d thread(s))", self._concurrency)

    def stop(self, timeout: float | None = None) -> None:
        """Stop accepting work and drain in-flight tasks. Default drain = one lease
        window: a 5s join would kill a daemon thread mid-LLM, leaving the task running
        until its lease expired (then reclaim+rerun — wasted spend). Idle threads exit
        within `idle_sleep` regardless, so this only waits when a task is genuinely
        running. Correctness past the drain is still covered by lease reclaim +
        ownership-checked completion + idempotent handlers."""
        if timeout is None:
            timeout = float(self._lease)
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        log.info("worker stopped")

    # ---- loop -------------------------------------------------------------

    def _loop(self) -> None:
        last_reclaim = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_reclaim >= self._reclaim_every:
                self._queue.reclaim_expired()
                last_reclaim = now

            task = self._queue.claim_next(self._lease)
            if task is None:
                self._stop.wait(self._idle)
                continue
            self._run_one(task)

    def _run_one(self, task) -> None:
        handler = self._handlers.get(task.type)
        if handler is None:
            self._queue.fail(task.id, f"no handler for {task.type}", retry=False)
            return

        # Heartbeat the lease while the handler runs (a screen can exceed one lease
        # window under LLM throttling). A live worker keeps pushing lease_until forward
        # so reclaim_expired only fires on a genuinely dead worker — this is what stops
        # a slow task from being reclaimed and double-run. The beat thread shares the
        # current lease via `box`; if it loses ownership the result is discarded.
        box = {"lease": task.lease_until, "lost": False}
        stop_beat = threading.Event()
        interval = max(1.0, self._lease / 3.0)

        def _beat() -> None:
            while not stop_beat.wait(interval):
                cur = box["lease"]
                if cur is None:
                    return
                new = self._queue.heartbeat(task.id, cur, self._lease)
                if new is None:
                    box["lost"] = True
                    return
                box["lease"] = new

        beat = threading.Thread(target=_beat, name=f"hb-{task.id[:8]}", daemon=True)
        beat.start()
        try:
            handler(task.payload)
        except PermanentTaskError as exc:
            stop_beat.set(); beat.join(timeout=2)
            log.warning("task %s (%s) permanently failed: %s", task.id, task.type.value, exc)
            self._queue.fail(task.id, str(exc), retry=False, lease_until=box["lease"])
            return
        except Exception as exc:  # noqa: BLE001 — worker must not die on a bad task
            stop_beat.set(); beat.join(timeout=2)
            err = f"{exc}\n{traceback.format_exc()}"
            log.exception("task %s (%s) failed", task.id, task.type.value)
            self._queue.fail(task.id, err, retry=True, lease_until=box["lease"])
            return

        stop_beat.set(); beat.join(timeout=2)
        if box["lost"]:
            log.warning("task %s (%s) finished but its lease was reclaimed; "
                        "discarding (a reclaimer owns it now)", task.id, task.type.value)
            return
        if self._queue.complete(task.id, lease_until=box["lease"]):
            log.info("task %s (%s) done", task.id, task.type.value)
        else:
            log.warning("task %s (%s) finished but lease was lost; not marked done",
                        task.id, task.type.value)


def run_standalone() -> None:
    """Run the worker on its own (separate process from the web app).

    Traps SIGTERM and SIGINT so a container / systemd / k8s stop drains
    gracefully: the stop event is set, in-flight task threads are given
    `stop(timeout)` to finish, and anything still running past that is reclaimed
    via its expired lease on the next startup (at-least-once, idempotent)."""
    import signal
    from app.candisift.adapters.http.container import build_container
    logging.basicConfig(level=logging.INFO)
    c = build_container()
    w = Worker(c.queue, c.service,
               lease_seconds=c.settings.worker_lease_seconds,
               concurrency=c.settings.worker_concurrency)
    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    w.start()
    shutdown.wait()          # block until a stop signal arrives
    log.info("shutdown signal received; draining worker")
    w.stop()


if __name__ == "__main__":
    run_standalone()
