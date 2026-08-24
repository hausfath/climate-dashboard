/* Warming Map tab.
 *
 * The tab embeds the standalone warming map (github.com/hausfath/warming-map,
 * served from GitHub Pages) in its ?embed=1 mode. Two jobs here:
 *   1. Lazy-load: the iframe gets its src only the first time the tab is
 *      shown, so visitors who never open the tab download nothing.
 *   2. Sync: relay the dashboard's dark/light and °C/°F toggles into the
 *      iframe via postMessage ({source:"climate-dashboard", theme, unit}),
 *      watching the body classes set by the clientside callbacks
 *      (body.light, body.fahrenheit — see units.js).
 */
(function () {
  var WM_BASE = "https://hausfath.github.io/warming-map/";
  var WM_ORIGIN = "https://hausfath.github.io";

  function state() {
    return {
      source: "climate-dashboard",
      theme: document.body.classList.contains("light") ? "light" : "dark",
      unit: document.body.classList.contains("fahrenheit") ? "F" : "C",
    };
  }

  function post() {
    var frame = document.getElementById("warming-map-frame");
    if (frame && frame.dataset.loaded && frame.contentWindow) {
      frame.contentWindow.postMessage(state(), WM_ORIGIN);
    }
  }

  function ensureLoaded() {
    var frame = document.getElementById("warming-map-frame");
    if (!frame || frame.dataset.loaded) return;
    var s = state();
    frame.dataset.loaded = "1";
    frame.addEventListener("load", post);
    frame.src = WM_BASE + "?embed=1&theme=" + s.theme + "&unit=" + s.unit;
  }

  // Full-bleed sizing: fill the viewport below the topbar exactly, even
  // when the topbar wraps to two rows on narrow screens (the CSS
  // calc(100dvh - 58px) is only the pre-JS fallback).
  function size() {
    var frame = document.getElementById("warming-map-frame");
    var topbar = document.querySelector(".topbar");
    if (frame && topbar) {
      frame.style.height =
        Math.max(400, window.innerHeight - topbar.getBoundingClientRect().height) + "px";
    }
  }

  function check() {
    var tab = document.getElementById("tab-content-map");
    if (tab && tab.style.display !== "none") { size(); ensureLoaded(); }
  }

  function init() {
    var tab = document.getElementById("tab-content-map");
    if (!tab) { setTimeout(init, 300); return; }  // Dash renders async
    new MutationObserver(check)
      .observe(tab, { attributes: true, attributeFilter: ["style"] });
    new MutationObserver(post)
      .observe(document.body, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("resize", size);
    check();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
