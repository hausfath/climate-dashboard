/* Topbar fitting.
 *
 * Keeps as much of the topbar as fits on one row, shedding detail one step
 * at a time instead of at fixed breakpoints (the space needed depends on the
 * fonts and labels, not just the screen width). Steps, cumulative classes on
 * <body> (styles at the end of theme.css):
 *   tb-1 hide the data-source subtitle   tb-2 tighter spacing
 *   tb-3 hide the "Updated" text          tb-4 short tab names
 *   tb-5 wrap to two rows
 * Below 900px the CSS media queries already wrap and shorten, so this only
 * refines what is left.
 */
(function () {
  var MAX = 5, level = -1, busy = false;
  function overflows(bar) {
    var right = bar.querySelector(".topbar-right");
    if (bar.scrollWidth > bar.clientWidth + 1) return true;
    if (right && right.scrollWidth > right.clientWidth + 1) return true;
    var last = bar.querySelector(".topbar-right > :last-child");
    return !!last && last.getBoundingClientRect().right > document.documentElement.clientWidth - 4;
  }
  function apply(n) {
    for (var i = 1; i <= MAX; i++) document.body.classList.toggle("tb-" + i, i <= n);
  }
  function fit() {
    var bar = document.querySelector(".topbar");
    if (!bar) { setTimeout(fit, 300); return; }  // Dash renders async
    if (!bar.dataset.fitWatch) {
      // the brand subtitle and "Updated …" text change after load (tab switches)
      bar.dataset.fitWatch = "1";
      new MutationObserver(schedule).observe(bar, { childList: true, subtree: true, characterData: true });
    }
    var n = 0;
    apply(0);
    while (n < MAX && overflows(bar)) apply(++n);
    if (n !== level) {
      level = n;
      // topbar height may have changed: let the iframe tabs resize themselves
      busy = true; window.dispatchEvent(new Event("resize")); busy = false;
    }
  }
  var raf = 0;
  function schedule() { if (busy) return; cancelAnimationFrame(raf); raf = requestAnimationFrame(fit); }
  window.addEventListener("resize", schedule);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedule);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", schedule);
  else schedule();
})();
