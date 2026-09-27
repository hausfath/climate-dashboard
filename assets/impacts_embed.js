/* El Niño Impacts tab.
 *
 * The tab embeds the interactive impacts map (elnino_map/, served same-origin
 * at /elnino-map/ by a Flask route in src/dashboard.py). As with the Warming
 * Map tab (map_embed.js):
 *   1. Lazy-load: the iframe gets its src the first time the tab is shown.
 *   2. Sync: relay the dashboard's dark/light toggle into the iframe via
 *      postMessage ({source:"climate-dashboard", theme}).
 *   3. Size: fill the viewport below the topbar exactly.
 */
(function () {
  function theme() { return document.body.classList.contains("light") ? "light" : "dark"; }
  function frame() { return document.getElementById("impacts-map-frame"); }

  function post() {
    var f = frame();
    if (f && f.dataset.loaded && f.contentWindow) {
      f.contentWindow.postMessage({ source: "climate-dashboard", theme: theme() }, window.location.origin);
    }
  }
  function ensureLoaded() {
    var f = frame();
    if (!f || f.dataset.loaded) return;
    f.dataset.loaded = "1";
    f.addEventListener("load", post);
    f.src = "/elnino-map/?embed=1&theme=" + theme();
  }
  function size() {
    var f = frame(), topbar = document.querySelector(".topbar");
    if (f && topbar) {
      f.style.height = Math.max(420, window.innerHeight - topbar.getBoundingClientRect().height) + "px";
    }
  }
  function check() {
    var tab = document.getElementById("tab-content-impacts");
    if (tab && tab.style.display !== "none") { size(); ensureLoaded(); }
  }
  function init() {
    var tab = document.getElementById("tab-content-impacts");
    if (!tab) { setTimeout(init, 300); return; }  // Dash renders async
    new MutationObserver(check).observe(tab, { attributes: true, attributeFilter: ["style"] });
    new MutationObserver(post).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("resize", size);
    check();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
