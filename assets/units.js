/* °C → °F for dashboard chrome text (KPI cards, heroes, captions, tables).
 *
 * Figures are converted server-side (src/units.py); this script only
 * touches text nodes OUTSIDE any Plotly graph. It is driven entirely by
 * the body.fahrenheit class, which the unit toggle's clientside callback
 * maintains — an attribute observer reacts to the flip, and a subtree
 * observer keeps late-rendered / re-rendered Dash components converted
 * while °F is active (original °C strings are remembered per text node,
 * so switching back is lossless).
 *
 * Loop safety: converted text contains no '°C', so re-processing a node
 * we just edited is a no-op; restores only happen when °F is off, and
 * the subtree observer ignores mutations while °F is off.
 *
 * Chrome values are anomalies (scale by 9/5) except the one
 * "Absolute: X°C" KPI line, detected by its "Absolute" prefix, which
 * also gets the +32 offset. ±-prefixed values are uncertainty
 * half-widths: scale only.
 */
(function () {
    'use strict';

    var originals = new WeakMap();   // text node → original °C string
    var NUM_C = /([+\-−±]?)(\d+(?:\.\d+)?)\s*°C/g;
    // "29.43 ±0.16°C" — the leading value carries no °C of its own
    var VAL_BEFORE_PM = /([+\-−]?\d+(?:\.\d+)?)(?=\s*±\s*\d+(?:\.\d+)?\s*°C)/g;

    function convert(text) {
        var absolute = /Absolute/i.test(text);
        text = text.replace(VAL_BEFORE_PM, function (_, num) {
            var dec = (num.split('.')[1] || '').length;
            var val = parseFloat(num.replace('−', '-'));
            var f = val * 1.8 + (absolute ? 32 : 0);
            var str = f.toFixed(dec);
            return (num.charAt(0) === '+' && f >= 0) ? '+' + str : str;
        });
        var out = text.replace(NUM_C, function (_, sign, num) {
            var dec = (num.split('.')[1] || '').length;
            var val = parseFloat(num);
            if (sign === '-' || sign === '−') val = -val;
            if (sign === '±') return '±' + Math.abs(val * 1.8).toFixed(dec) + '°F';
            var f = val * 1.8 + (absolute ? 32 : 0);
            var str = f.toFixed(dec) + '°F';
            return (sign === '+' && f >= 0) ? '+' + str : str;
        });
        return out.replace(/°C/g, '°F');
    }

    function skip(el) {
        if (!el || !el.closest) return true;
        return !!el.closest('.js-plotly-plot, .unit-cluster, script, style');
    }

    function eachTextNode(root, fn) {
        var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode: function (node) {
                return skip(node.parentElement)
                    ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
            }
        });
        var nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        nodes.forEach(fn);   // collect first: edits during the walk are unsafe
    }

    function toF(node) {
        if (node.nodeValue.indexOf('°C') === -1) return;
        originals.set(node, node.nodeValue);
        node.nodeValue = convert(node.nodeValue);
    }

    function toC(node) {
        var orig = originals.get(node);
        if (orig !== undefined) {
            originals.delete(node);
            node.nodeValue = orig;
        }
    }

    function active() {
        return document.body.classList.contains('fahrenheit');
    }

    function init() {
        // React to the toggle (body class flip)…
        new MutationObserver(function () {
            eachTextNode(document.body, active() ? toF : toC);
        }).observe(document.body, {
            attributes: true, attributeFilter: ['class']
        });
        // …and keep Dash re-renders converted while °F is on.
        new MutationObserver(function (mutations) {
            if (!active()) return;
            mutations.forEach(function (m) {
                if (m.type === 'characterData') {
                    if (!skip(m.target.parentElement)) toF(m.target);
                } else {
                    m.addedNodes.forEach(function (n) {
                        if (n.nodeType === Node.TEXT_NODE) {
                            if (!skip(n.parentElement)) toF(n);
                        } else if (n.nodeType === Node.ELEMENT_NODE && !skip(n)) {
                            eachTextNode(n, toF);
                        }
                    });
                }
            });
        }).observe(document.body, {
            childList: true, characterData: true, subtree: true
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
