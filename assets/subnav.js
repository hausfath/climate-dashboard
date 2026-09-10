/* In-page section navigation (ENSO tab sub-nav pills + linked KPI cards).
 *
 * The tab-switch callback owns the URL hash (#global / #enso / …) and
 * rewrites it on every trigger, so a plain <a href="#sec-…"> jump gets
 * undone a moment later. Instead: intercept clicks on .subnav a and
 * .kpi-link, scroll the target section into view, and never touch the hash.
 *
 * Shared links that arrive with a section hash (#sec-coupled) still work:
 * the server maps them to the ENSO tab, and this script captures the hash
 * at load and scrolls once the section has rendered.
 */
(function () {
  var TOPBAR = 66;  // sticky topbar height + breathing room

  function scrollToId(id) {
    var el = document.getElementById(id);
    if (!el) return false;
    // Vertical-only: scrollIntoView can also nudge horizontally when a wide
    // Plotly graph makes the document overflow, which shifts the topbar.
    var top = el.getBoundingClientRect().top + window.pageYOffset - TOPBAR;
    window.scrollTo({ top: top, left: 0, behavior: 'smooth' });
    return true;
  }

  document.addEventListener('click', function (ev) {
    var a = ev.target.closest('.subnav a, a.kpi-link');
    if (!a) return;
    var href = a.getAttribute('href') || '';
    if (href.charAt(0) !== '#') return;
    ev.preventDefault();
    scrollToId(href.slice(1));
  });

  // Deep link on first load: wait for Dash to render the section, then scroll.
  var initial = (window.location.hash || '').slice(1);
  if (initial.indexOf('sec-') === 0) {
    var tries = 0;
    var timer = setInterval(function () {
      tries += 1;
      var el = document.getElementById(initial);
      var visible = el && el.offsetParent !== null;
      if ((visible && scrollToId(initial)) || tries > 40) clearInterval(timer);
    }, 250);
  }
})();
