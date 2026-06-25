// Same-origin external script (CSP: script-src 'self'). Progressive enhancement —
// the UI works fully without it. Adds two affordances to the recruiter UI:
//   1. a submit spinner during synchronous waits (job creation = 2 LLM calls ~15s,
//      upload = cost estimate),
//   2. a selected-file preview with per-file delete on the resume dropzone, so the
//      recruiter sees what's queued and can drop a file before paying for a screen.
(function () {
  function fmtSize(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(0) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
  }

  // Re-render the file list from input.files (the source of truth that gets POSTed).
  function render(input, list) {
    list.innerHTML = '';
    Array.from(input.files).forEach(function (file, idx) {
      var row = document.createElement('div');
      row.className = 'file-row';

      var name = document.createElement('span');
      name.className = 'file-name';
      name.textContent = file.name;

      var size = document.createElement('span');
      size.className = 'file-size';
      size.textContent = fmtSize(file.size);

      var del = document.createElement('button');
      del.type = 'button';                 // not a submit button
      del.className = 'file-del';
      del.textContent = '✕';
      del.setAttribute('aria-label', 'Remove ' + file.name);
      del.addEventListener('click', function () {
        // input.files is read-only; rebuild it via DataTransfer minus this file.
        var dt = new DataTransfer();
        Array.from(input.files).forEach(function (f, j) { if (j !== idx) dt.items.add(f); });
        input.files = dt.files;
        render(input, list);
      });

      row.appendChild(name);
      row.appendChild(size);
      row.appendChild(del);
      list.appendChild(row);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('input[type="file"]').forEach(function (input) {
      var form = input.closest('form');
      var list = form && form.querySelector('[data-file-list]');
      if (!list) return;
      input.addEventListener('change', function () { render(input, list); });
    });
  });

  // Submit spinner + timeout recovery: mark the submit button loading (CSS shows the
  // spinner + pointer-events:none to block a double submit). If the request hangs,
  // recover after 30s so the button never locks forever. Skips when a form-level
  // confirm handler cancelled the submit (defaultPrevented).
  document.addEventListener('submit', function (e) {
    if (e.defaultPrevented) return;
    var btn = e.target.querySelector('button[type="submit"], button:not([type])');
    if (!btn || btn.classList.contains('is-loading')) return;
    if (btn.dataset.loading) btn.textContent = btn.dataset.loading;
    btn.classList.add('is-loading');
    btn.setAttribute('aria-busy', 'true');
    setTimeout(function () {
      if (btn.classList.contains('is-loading')) {
        btn.classList.remove('is-loading');
        btn.removeAttribute('aria-busy');
        btn.insertAdjacentHTML('afterend',
          '<span class="flash flash-bad" role="status">Request timed out — try again.</span>');
      }
    }, 30000);
  });

  // Print to PDF: <button data-print> triggers browser print dialog.
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-print]')) window.print();
  });

  // Clipboard copy: <button data-copy-from="#some-id"> copies value/text of target.
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-copy-from]');
    if (!btn) return;
    var el = document.querySelector(btn.getAttribute('data-copy-from'));
    if (!el) return;
    var text = el.value !== undefined ? el.value : el.textContent;
    navigator.clipboard.writeText(text).then(function () {
      var orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.style.background = 'var(--ok)';
      btn.style.color = '#04220f';
      setTimeout(function () {
        btn.textContent = orig;
        btn.style.background = '';
        btn.style.color = '';
      }, 2000);
    });
  });
})();

// --- WS1 design-system progressive enhancements (CSP: external same-origin only) ---
document.addEventListener('DOMContentLoaded', function () {
  // drag-active feedback on dropzones (audit: zone gave zero feedback on dragover)
  document.querySelectorAll('.dropzone').forEach(function (zone) {
    ['dragenter', 'dragover'].forEach(function (e) {
      zone.addEventListener(e, function () { zone.classList.add('is-dragover'); });
    });
    ['dragleave', 'drop'].forEach(function (e) {
      zone.addEventListener(e, function () { zone.classList.remove('is-dragover'); });
    });
  });

  // confirm dialog on destructive POSTs (audit: no confirm on recruiter accept/reject)
  document.querySelectorAll('form[data-confirm]').forEach(function (form) {
    form.addEventListener('submit', function (ev) {
      if (!window.confirm(form.getAttribute('data-confirm'))) { ev.preventDefault(); }
    });
  });

  // flash dismiss + auto-dismiss
  document.querySelectorAll('[data-flash]').forEach(function (flash) {
    var close = flash.querySelector('[data-flash-close]');
    if (close) { close.addEventListener('click', function () { flash.remove(); }); }
    if (flash.classList.contains('flash-ok')) {
      setTimeout(function () { flash.remove(); }, 6000);
    }
  });
});
