"""Durable task queue backed by the SQLite `task` table — survives restarts.

Why DB-backed instead of in-process BackgroundTasks: tasks must outlive the
process. If the worker (or the whole box) dies mid-screen, the task is still in
the table with status='running' and an expired lease, and gets reclaimed on the
next startup. At-least-once delivery; handlers are written idempotent
(dedup on ingest, deterministic result id on screen).

ponytail: single in-process worker by default — `claim_next` increments attempts
under SQLite's write lock, which serializes claims. For multiple worker processes
or higher throughput, swap this adapter for Redis/RQ or Celery; the TaskQueue port
stays the same. The upgrade is one adapter, zero application changes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.candisift.domain.models import Task, TaskStatus, TaskType
from .db import TaskRow, is_unique_violation


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_task(row: TaskRow) -> Task:
    return Task(
        id=row.id, type=TaskType(row.type), payload=row.payload_json,
        status=TaskStatus(row.status), attempts=row.attempts, max_attempts=row.max_attempts,
        last_error=row.last_error, lease_until=row.lease_until, available_at=row.available_at,
        created_at=row.created_at, updated_at=row.updated_at,
    )


class SqliteTaskQueue:
    def __init__(self, engine: Engine, max_attempts: int = 3,
                 retry_base_seconds: float = 0.0, retry_max_seconds: float = 300.0) -> None:
        # base=0 => no backoff (immediate retry); the composition root injects the
        # production value (Settings.worker_retry_base_seconds) so policy lives there.
        self._engine = engine
        self._max_attempts = max_attempts
        self._retry_base = max(0.0, float(retry_base_seconds))
        self._retry_max = max(0.0, float(retry_max_seconds))

    def _retry_delay(self, attempts: int) -> float:
        """Exponential backoff with a cap: base * 2^(attempts-1), <= retry_max.
        attempts is the count AFTER the failed try (incremented at claim), so the
        first failure waits `base`, the second `base*2`, etc."""
        if self._retry_base <= 0:
            return 0.0
        return min(self._retry_max, self._retry_base * (2 ** max(0, attempts - 1)))

    def enqueue(self, task_type: TaskType, payload: dict, staged: bool = False,
                task_id: str | None = None) -> str:
        """Enqueue a task. Pass a deterministic `task_id` to make the enqueue
        IDEMPOTENT: a re-run (e.g. ingest succeeded but its `complete()` was lost to a
        crash, so the task runs again) that re-enqueues the same downstream screen task
        is silently ignored instead of creating a duplicate screen — which would
        double the LLM spend. Race-safe via the primary-key UNIQUE constraint."""
        tid = task_id or uuid.uuid4().hex
        now = _now()
        status = TaskStatus.staged if staged else TaskStatus.pending
        with Session(self._engine) as s:
            try:
                s.add(TaskRow(
                    id=tid, type=task_type.value, payload_json=payload,
                    status=status.value, max_attempts=self._max_attempts,
                    available_at=None, created_at=now, updated_at=now,  # NULL = claim now (matches reclaim/requeue)
                ))
                s.commit()
            except (IntegrityError, ValueError) as e:
                s.rollback()        # same id already enqueued -> idempotent no-op
                if not is_unique_violation(e):
                    raise           # a real error, not the expected dup-key collision
        return tid

    def release(self, task_ids: list[str]) -> int:
        """Flip staged tasks to pending (recruiter confirmed the cost)."""
        now = _now()
        released = 0
        with Session(self._engine) as s:
            for tid in task_ids:
                row = s.get(TaskRow, tid)
                if row and row.status == TaskStatus.staged.value:
                    row.status = TaskStatus.pending.value
                    row.updated_at = now
                    s.add(row)
                    released += 1
            s.commit()
        return released

    def claim_next(self, lease_seconds: int) -> Task | None:
        """Claim the oldest pending task. Marks running + leases + counts the attempt.

        The claim is atomic: pick the oldest pending id, then flip it with a
        conditional UPDATE (`WHERE status='pending'`). If a concurrent worker
        grabbed it between the pick and the update, rowcount is 0 and we report
        "nothing claimed" (the caller polls again). This makes N worker threads
        safe — no SELECT-then-mutate race, no double-claim."""
        now = _now()
        lease = now + timedelta(seconds=lease_seconds)
        with Session(self._engine) as s:
            tid = s.exec(
                select(TaskRow.id)
                .where(TaskRow.status == TaskStatus.pending.value)
                # skip tasks still in their retry backoff window (NULL = available now)
                .where(or_(TaskRow.available_at == None, TaskRow.available_at <= now))  # noqa: E711
                .order_by(TaskRow.created_at)
                .limit(1)
            ).first()
            if tid is None:
                return None
            res = s.execute(
                update(TaskRow)
                .where(TaskRow.id == tid, TaskRow.status == TaskStatus.pending.value)
                .values(
                    status=TaskStatus.running.value,
                    attempts=TaskRow.attempts + 1,   # counted at claim -> poison tasks can't loop forever
                    lease_until=lease,
                    updated_at=now,
                )
            )
            s.commit()
            if res.rowcount == 0:        # lost the race to another worker
                return None
            return _to_task(s.get(TaskRow, tid))

    def complete(self, task_id: str, lease_until: datetime | None = None) -> bool:
        """Mark done — but ONLY if we still own the lease. After a lease expires and
        another worker reclaims the task, the original (slow) worker must not write its
        result over the reclaimer's: the conditional UPDATE matches nothing and we
        return False so the caller knows its work was superseded. Pass the lease_until
        from the claimed Task to enable the check; omit it for legacy unconditional."""
        now = _now()
        with Session(self._engine) as s:
            q = update(TaskRow).where(TaskRow.id == task_id,
                                      TaskRow.status == TaskStatus.running.value)
            if lease_until is not None:
                q = q.where(TaskRow.lease_until == lease_until)
            res = s.execute(q.values(status=TaskStatus.done.value, updated_at=now))
            s.commit()
            return res.rowcount > 0

    def heartbeat(self, task_id: str, lease_until: datetime, lease_seconds: int) -> datetime | None:
        """Extend a running task's lease while we still own it. Returns the NEW
        lease_until (thread it back for the next beat / the final complete), or None if
        ownership was lost (the task was reclaimed) — the worker should then stop, as a
        reclaimer is now running it. This is what makes 'lease expiry == worker death'
        true: a live worker keeps pushing the lease forward, so reclaim only fires on a
        genuinely dead one — not on a task that's merely slower than one lease window."""
        now = _now()
        new_lease = now + timedelta(seconds=lease_seconds)
        with Session(self._engine) as s:
            res = s.execute(
                update(TaskRow)
                .where(TaskRow.id == task_id,
                       TaskRow.status == TaskStatus.running.value,
                       TaskRow.lease_until == lease_until)
                .values(lease_until=new_lease, updated_at=now)
            )
            s.commit()
            return new_lease if res.rowcount > 0 else None

    def fail(self, task_id: str, error: str, retry: bool,
             lease_until: datetime | None = None) -> None:
        with Session(self._engine) as s:
            row = s.get(TaskRow, task_id)
            if row is None or row.status != TaskStatus.running.value:
                return
            if lease_until is not None and row.lease_until != lease_until:
                return                  # lease lost to a reclaimer — not ours to fail
            row.last_error = error[:2000]
            row.updated_at = _now()
            if retry and row.attempts < row.max_attempts:
                row.status = TaskStatus.pending.value   # back to the queue
                row.lease_until = None
                # backoff: hold it out of claim until the delay elapses so a transient
                # failure can't hot-loop through the whole attempt budget in milliseconds.
                row.available_at = _now() + timedelta(seconds=self._retry_delay(row.attempts))
            else:
                row.status = TaskStatus.failed.value     # dead-letter
            s.add(row)
            s.commit()

    def reclaim_expired(self) -> int:
        """Crash recovery: running tasks whose lease expired go back to pending.
        Attempts were already counted at claim, so poison tasks still dead-letter."""
        now = _now()
        reclaimed = 0
        with Session(self._engine) as s:
            rows = s.exec(
                select(TaskRow)
                .where(TaskRow.status == TaskStatus.running.value)
                .where(TaskRow.lease_until < now)
            ).all()
            for row in rows:
                if row.attempts >= row.max_attempts:
                    row.status = TaskStatus.failed.value
                    row.last_error = "lease expired, attempts exhausted"
                else:
                    row.status = TaskStatus.pending.value
                    row.lease_until = None
                    row.available_at = None   # crash, not a logical failure -> retry now
                row.updated_at = now
                s.add(row)
                reclaimed += 1
            s.commit()
        return reclaimed

    def stats(self) -> dict:
        with Session(self._engine) as s:
            rows = s.exec(
                select(TaskRow.status, func.count()).group_by(TaskRow.status)
            ).all()
        return {status: count for status, count in rows}

    def list_by_status(self, status: TaskStatus, limit: int = 100) -> list[Task]:
        with Session(self._engine) as s:
            rows = s.exec(
                select(TaskRow).where(TaskRow.status == status.value)
                .order_by(TaskRow.updated_at.desc()).limit(limit)
            ).all()
            return [_to_task(r) for r in rows]

    def job_progress(self, job_id: str) -> dict:
        """Task-status counts + failure errors for one job (its ingest + screen
        tasks both carry job_id in the payload). Powers the live progress card so
        a recruiter sees work happening — and sees failures instead of a silent
        Results(0). ponytail: full table scan; add a job_id column + index if the
        task table ever grows large."""
        out = {"staged": 0, "pending": 0, "running": 0, "done": 0, "failed": 0, "errors": []}
        with Session(self._engine) as s:
            rows = s.exec(select(TaskRow.status, TaskRow.payload_json, TaskRow.last_error)).all()
        for status, payload, err in rows:
            if not (isinstance(payload, dict) and payload.get("job_id") == job_id):
                continue
            out[status] = out.get(status, 0) + 1
            if status == TaskStatus.failed.value and err:
                out["errors"].append(err.splitlines()[0][:300])
        return out

    def requeue(self, task_id: str) -> bool:
        """Manually retry a dead-lettered task (resets attempts). Returns True if requeued."""
        with Session(self._engine) as s:
            row = s.get(TaskRow, task_id)
            if row is None or row.status != TaskStatus.failed.value:
                return False
            row.status = TaskStatus.pending.value
            row.attempts = 0
            row.last_error = ""
            row.lease_until = None
            row.available_at = None   # manual retry is immediate
            row.updated_at = _now()
            s.add(row)
            s.commit()
        return True
