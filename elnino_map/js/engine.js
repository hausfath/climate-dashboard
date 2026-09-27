/* Map engine, ported from the explainer video (El Nino Impacts/video/src/map.js) for interactive use.
   Equirectangular camera {lon, lat, s}: s = px per degree of latitude; x scale = s·cos(0.75·clamp(lat, ±50)).
   Land, borders and regions are Path2D objects in lon/lat drawn through a canvas transform, once per visible
   360° copy, so panning wraps around the dateline. The rainfall field and agreement dots are built in the
   browser from the same per-model grids the point readout uses (data/grid_<season>.bin). */
'use strict';
const DEG = Math.PI / 180;
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, p) => a + (b - a) * p;
const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
const wrap180 = (d) => ((d + 540) % 360) - 180;

// published-map tier fills (figures/make_impacts_map.py)
const TIER = { dry: { high: '#8C510A', medhigh: '#C4791C', medium: '#F4C154' }, wet: { high: '#0A5599', medhigh: '#3F8FC5', medium: '#7FC4EA' }, warm: { medium: '#D6604D' } };
const CONF_LABEL = { high: 'High', medhigh: 'Medium-high', medium: 'Medium' };

function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }
function mixHex(h1, h2, p) {
  const a = parseInt(h1.slice(1), 16), b = parseInt(h2.slice(1), 16);
  const r = Math.round(lerp((a >> 16) & 255, (b >> 16) & 255, p)), g = Math.round(lerp((a >> 8) & 255, (b >> 8) & 255, p)), bl = Math.round(lerp(a & 255, b & 255, p));
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
}

const Engine = (() => {
  let cv, ctx, W = 0, H = 0, DPR = 1;
  const CAM = { lon: 170, lat: 8, s: 4 };
  let PAL = {};
  const kx = (cam = CAM) => cam.s * Math.cos(clamp(cam.lat, -50, 50) * 0.75 * DEG);
  const P = (lon, lat, cam = CAM) => [W / 2 + wrap180(lon - cam.lon) * kx(cam), H / 2 - (lat - cam.lat) * cam.s];
  const unproject = (x, y, cam = CAM) => [wrap180(cam.lon + (x - W / 2) / kx(cam)), cam.lat - (y - H / 2) / cam.s];
  const view = (cam = CAM) => { const hw = W / 2 / kx(cam), hh = H / 2 / cam.s; return { w: cam.lon - hw, e: cam.lon + hw, s: cam.lat - hh, n: cam.lat + hh }; };
  const minS = () => W / 360;
  function constrain(cam = CAM) {
    cam.s = clamp(cam.s, minS(), 70);
    const hh = H / 2 / cam.s;
    cam.lat = hh >= 88 ? 0 : clamp(cam.lat, -88 + hh, 88 - hh);
    cam.lon = wrap180(cam.lon);
    return cam;
  }
  // van Wijk-style fly: log zoom with an outward bump proportional to distance
  function flyCam(a, b, p) {
    const q = ease(clamp(p)), dl = wrap180(b.lon - a.lon), dist = Math.hypot(dl, b.lat - a.lat);
    const bump = Math.max(0, Math.log(Math.max(1, dist * Math.min(a.s, b.s) / 900))) * 0.9;
    const ls = lerp(Math.log(a.s), Math.log(b.s), q) - bump * Math.sin(Math.PI * q);
    return { lon: a.lon + dl * q, lat: lerp(a.lat, b.lat, q), s: Math.exp(ls) };
  }

  // ---- geometry ----
  function ringsPath(rings, close = true) {
    const p = new Path2D();
    for (const r of rings) { p.moveTo(r[0][0], -r[0][1]); for (let i = 1; i < r.length; i++) p.lineTo(r[i][0], -r[i][1]); if (close) p.closePath(); }
    return p;
  }
  function bboxOf(rings) {
    let w = 1e9, e = -1e9, s = 1e9, n = -1e9;
    for (const r of rings) for (const [x, y] of r) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
    return [w, s, e, n];
  }
  function inRing(r, x, y) {
    let c = false;
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  // point-in-region with dateline copies (rings may use lon > 180 or < -180)
  function inRegion(R, lon, lat) {
    const [w, s, e, n] = R.bb; if (lat < s || lat > n) return false;
    for (const k of [0, 360, -360]) {
      const x = lon + k; if (x < w || x > e) continue;
      let c = false; for (const r of R.rings) if (inRing(r, x, lat)) c = !c;
      if (c) return true;
    }
    return false;
  }
  function segDist(px, py, r) {
    let d = 1e9;
    for (let i = 1; i < r.length; i++) {
      const [ax, ay] = r[i - 1], [bx, by] = r[i], dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1;
      const t = clamp(((px - ax) * dx + (py - ay) * dy) / L), qx = ax + t * dx - px, qy = ay + t * dy - py;
      d = Math.min(d, qx * qx + qy * qy);
    }
    return Math.sqrt(d);
  }
  // label anchor: the interior grid point farthest from the edge (cheap pole of inaccessibility)
  function labelPoint(R) {
    const [w, s, e, n] = R.bb; let best = null, bd = -1; const N = 28;
    for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) {
      const x = w + (e - w) * i / N, y = s + (n - s) * j / N;
      let inside = false; for (const r of R.rings) if (inRing(r, x, y)) inside = !inside;
      if (!inside) continue;
      let d = 1e9; for (const r of R.rings) d = Math.min(d, segDist(x, y, r));
      d *= Math.cos(y * DEG) ** 0.3;
      if (d > bd) { bd = d; best = [x, y]; }
    }
    return best ?? R.centroid;
  }
  // lon span of a set of bboxes, unwrapped around the first one
  function unionBB(bbs) {
    const c0 = (bbs[0][0] + bbs[0][2]) / 2;
    let w = 1e9, e = -1e9, s = 1e9, n = -1e9;
    for (const b of bbs) {
      const c = (b[0] + b[2]) / 2, k = c0 + wrap180(c - c0) - c;
      w = Math.min(w, b[0] + k); e = Math.max(e, b[2] + k); s = Math.min(s, b[1]); n = Math.max(n, b[3]);
    }
    return [w, s, e, n];
  }
  // camera that frames bbox inside the free part of the screen (pad = {l, r, t, b} px covered by panels)
  function camFor(bb, pad = { l: 0, r: 0, t: 0, b: 0 }, maxS = 26) {
    const fw = Math.max(200, W - pad.l - pad.r), fh = Math.max(160, H - pad.t - pad.b);
    const lat = (bb[1] + bb[3]) / 2, cosf = Math.cos(clamp(lat, -50, 50) * 0.75 * DEG);
    const s = clamp(Math.min(fw * 0.78 / (Math.max(bb[2] - bb[0], 4) * cosf), fh * 0.74 / Math.max(bb[3] - bb[1], 3)), minS(), maxS);
    const cam = { lon: (bb[0] + bb[2]) / 2, lat, s };
    cam.lon -= ((pad.l - pad.r) / 2) / kx(cam);
    cam.lat += ((pad.t - pad.b) / 2) / s;
    return constrain(cam);
  }

  // ---- data ----
  let GEO, REG = {}, XREG = {}, GRID = {}, FIELD = {}, DOTS = {}, SST = null;
  function setGeo(g) {
    GEO = { land: g.land.map((r) => ({ path: ringsPath([r]), bb: bboxOf([r]) })), borders: ringsPath(g.borders, false) };
  }
  function prepRegion(R) { R.path = ringsPath(R.rings); R.bb = bboxOf(R.rings); R.lp = labelPoint(R); return R; }
  function setRegions(regs) { REG = regs.regions; XREG = regs.extras; for (const k in REG) prepRegion(REG[k]); }

  // PR colormap + alpha exactly as tools/export_data.py (pr_<S>.png): colour by % of normal, alpha by |anomaly|
  const PR = [[-100, '#7A3E06'], [-60, '#B4651A'], [-30, '#E3A24A'], [-10, '#D8A057'], [0, '#F5EEDC'], [10, '#7FB6E3'], [30, '#6FB4E6'], [60, '#2F86D6'], [100, '#1D5FB8']]
    .map(([v, h]) => [v, parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  function prColor(v) {
    v = clamp(v, -100, 100);
    for (let i = 1; i < PR.length; i++) if (v <= PR[i][0]) { const a = PR[i - 1], b = PR[i], t = (v - a[0]) / (b[0] - a[0]); return [lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)]; }
    return PR[PR.length - 1].slice(1);
  }
  const prAlpha = (v) => Math.pow(clamp((Math.abs(v) - 5) / 40), 0.85) * (235 / 255);
  // grid: Int8Array [(n + 1) × 181 × 360], last layer = multi-model mean; 127 = no value
  function setGrid(season, buf, n) {
    const g = new Int8Array(buf), NL = 181 * 360;
    GRID[season] = { g, n, NL };
    // field image at 4 px per degree, bilinear in value space, soft edge where cells are missing
    const R = 4, w = 360 * R, h = 180 * R, img = new ImageData(w, h), mm = g.subarray(n * NL, (n + 1) * NL);
    const val = (j, i) => mm[clamp(j, 0, 180) * 360 + ((i % 360) + 360) % 360];
    // soft validity mask (3×3 box over has-value cells), so masked deserts get smooth edges instead of 1° steps
    const vs = new Float32Array(NL);
    for (let j = 0; j < 181; j++) for (let i = 0; i < 360; i++) {
      let c = 0; for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) c += val(j + dj, i + di) !== 127 ? 1 : 0;
      vs[j * 360 + i] = c / 9;
    }
    const vv = (j, i) => vs[clamp(j, 0, 180) * 360 + ((i % 360) + 360) % 360];
    for (let py = 0; py < h; py++) {
      const gy = (py + 0.5) / R, j0 = Math.floor(gy), fy = gy - j0;
      for (let px = 0; px < w; px++) {
        const gx = (px + 0.5) / R + 360, i0 = Math.floor(gx), fx = gx - i0;
        const v = [val(j0, i0), val(j0, i0 + 1), val(j0 + 1, i0), val(j0 + 1, i0 + 1)], wt = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
        let s = 0, ws = 0, cover = 0;
        for (let q = 0; q < 4; q++) if (v[q] !== 127) { s += v[q] * wt[q]; ws += wt[q]; cover += wt[q]; }
        if (ws < 1e-6) continue;
        const m = vv(j0, i0) * wt[0] + vv(j0, i0 + 1) * wt[1] + vv(j0 + 1, i0) * wt[2] + vv(j0 + 1, i0 + 1) * wt[3];
        const x = s / ws, [r, gg, b] = prColor(x), o = (py * w + px) * 4, e = clamp((m - 0.4) / 0.45);
        img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b;
        img.data[o + 3] = 255 * prAlpha(x) * e * e * (3 - 2 * e) * clamp(cover * 4);
      }
    }
    const c = document.createElement('canvas'); c.width = w; c.height = h; c.getContext('2d').putImageData(img, 0, 0);
    FIELD[season] = c;
    // agreement dots: ≥ 80% of models share the sign of the mean and |mean| ≥ 10% (as in the video)
    const pts = [];
    for (let j = 0; j < 181; j++) for (let i = 0; i < 360; i++) {
      const m = mm[j * 360 + i]; if (m === 127 || Math.abs(m) < 10) continue;
      let ag = 0; for (let k = 0; k < n; k++) { const v = g[k * NL + j * 360 + i]; if (v !== 127 && Math.sign(v) === Math.sign(m)) ag++; }
      if (ag / n >= 0.8 - 1e-9) pts.push(i - 180, 90 - j);
    }
    DOTS[season] = new Int16Array(pts);
  }
  function pointQuery(season, lon, lat) {
    const G = GRID[season]; if (!G) return null;
    const j = clamp(Math.round(90 - lat), 0, 180), i = ((Math.round(lon) + 180) % 360 + 360) % 360, o = j * 360 + i;
    const vals = []; for (let k = 0; k < G.n; k++) { const v = G.g[k * G.NL + o]; vals.push(v === 127 ? null : v); }
    const mm = G.g[G.n * G.NL + o];
    return { lat: 90 - j, lon: i - 180, vals, mean: mm === 127 ? null : mm };
  }

  // ---- drawing ----
  function readPalette() {
    const cs = getComputedStyle(document.documentElement), v = (k) => cs.getPropertyValue(k).trim();
    PAL = { oceanTop: v('--map-ocean-top'), ocean: v('--map-ocean'), oceanBot: v('--map-ocean-bot'), land: v('--map-land'), coast: v('--map-coast'), border: v('--map-border'),
      grat: v('--map-grat'), eq: v('--map-eq'), dot: v('--map-dot'), fillA: +v('--map-fill-a'), fieldA: +v('--map-field-a'), label: v('--map-label'), halo: v('--map-halo'),
      edge: { dry: v('--map-edge-dry'), wet: v('--map-edge-wet'), warm: v('--map-edge-warm') }, hit: v('--hit'), miss: v('--miss'), text: v('--text'), accent: v('--accent'),
      light: document.documentElement.dataset.theme === 'light' };
  }
  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    constrain();
  }
  function eachCopy(bb, fn) {
    const v = view(), k0 = kx();
    for (const k of [-360, 0, 360, -720, 720]) {
      if (bb && (bb[2] + k < v.w - 2 || bb[0] + k > v.e + 2 || bb[3] < v.s - 2 || bb[1] > v.n + 2)) continue;
      if (!bb && Math.abs(k) > 360) continue;
      ctx.save();
      ctx.setTransform(DPR * k0, 0, 0, DPR * CAM.s, DPR * (W / 2 + (k - CAM.lon) * k0), DPR * (H / 2 + CAM.lat * CAM.s));
      fn(k); ctx.restore();
    }
  }
  function drawImageWorld(im, a) {
    if (!im || a <= 0.01) return;
    for (const k of [-360, 0, 360]) {
      const xa = W / 2 + (-180 + k - CAM.lon) * kx(), xb = xa + 360 * kx();
      if (xb < 0 || xa > W) continue;
      ctx.save(); ctx.globalAlpha = a; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(im, xa, H / 2 - (90 - CAM.lat) * CAM.s, 360 * kx(), 180 * CAM.s); ctx.restore();
    }
  }
  function drawGraticule() {
    const v = view(), step = CAM.s > 16 ? 10 : 30;
    ctx.save(); ctx.strokeStyle = PAL.grat; ctx.lineWidth = 1; ctx.beginPath();
    for (let lo = Math.ceil(v.w / step) * step; lo <= v.e; lo += step) { const x = W / 2 + (lo - CAM.lon) * kx(); ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let la = Math.ceil(v.s / step) * step; la <= v.n; la += step) { const [, y] = P(0, la); ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    const [, ye] = P(0, 0); ctx.strokeStyle = PAL.eq; ctx.setLineDash([6, 10]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(0, ye); ctx.lineTo(W, ye); ctx.stroke();
    ctx.restore();
  }
  function drawDots(season, a) {
    const pts = DOTS[season]; if (!pts || a <= 0.01) return;
    const v = view(), r = clamp(CAM.s / 12, 0.9, 2.4), k0 = kx();
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = PAL.dot;
    for (let q = 0; q < pts.length; q += 2) {
      const la = pts[q + 1]; if (la < v.s - 1 || la > v.n + 1) continue;
      const x = W / 2 + wrap180(pts[q] - CAM.lon) * k0; if (x < -4 || x > W + 4) continue;
      ctx.fillRect(x - r / 2, H / 2 - (la - CAM.lat) * CAM.s - r / 2, r, r);
    }
    ctx.restore();
  }
  // L: {season, fieldA, dotsA, sstA, regions: [{R, a, fill, hi, sel, dim, dashed}]}
  function draw(L) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, PAL.oceanTop); g.addColorStop(0.5, PAL.ocean); g.addColorStop(1, PAL.oceanBot);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    drawGraticule();
    drawImageWorld(SST, L.sstA ?? 0);
    for (const Ld of GEO.land) eachCopy(Ld.bb, () => { ctx.fillStyle = PAL.land; ctx.fill(Ld.path); });
    const fz = 1 - 0.3 * clamp((CAM.s - 7) / 10);
    if (L.fieldFrom && L.fieldMix < 1) drawImageWorld(FIELD[L.fieldFrom], (L.fieldA ?? 0) * PAL.fieldA * fz * (1 - L.fieldMix));
    drawImageWorld(FIELD[L.season], (L.fieldA ?? 0) * PAL.fieldA * fz * (L.fieldFrom ? L.fieldMix : 1));
    for (const Ld of GEO.land) eachCopy(Ld.bb, () => { ctx.strokeStyle = PAL.coast; ctx.lineWidth = (CAM.s > 15 ? 1.2 : 0.9) / CAM.s; ctx.stroke(Ld.path); });
    const ba = clamp((CAM.s - 6) / 6);
    if (ba > 0) eachCopy(null, () => { ctx.globalAlpha = ba; ctx.strokeStyle = PAL.border; ctx.lineWidth = 0.8 / CAM.s; ctx.stroke(GEO.borders); });
    drawDots(L.season, L.dotsA ?? 0);
    for (const o of L.regions) drawRegion(o);
    for (const o of L.regions) if (o.glyph) drawGlyph(o);
  }
  function drawRegion(o) {
    const R = o.R; if (o.a <= 0.01) return;
    const edge = o.edge ?? PAL.edge[R.kind] ?? PAL.edge.dry;
    eachCopy(R.bb, () => {
      ctx.globalAlpha = o.a; ctx.lineJoin = 'round';
      ctx.fillStyle = hexA(o.fill ?? R.fill, o.fillA ?? PAL.fillA); ctx.fill(R.path);
      const lw = (o.lw ?? 1.6) / CAM.s;
      if (o.hi > 0) { ctx.save(); ctx.shadowColor = edge; ctx.shadowBlur = 22 * o.hi * DPR; ctx.strokeStyle = hexA(edge.startsWith('#') ? edge : '#ffffff', 0.95 * o.hi); ctx.lineWidth = lw * 2.2; ctx.stroke(R.path); ctx.restore(); }
      ctx.strokeStyle = edge; ctx.globalAlpha = o.a * (o.edgeA ?? 0.85); ctx.lineWidth = lw;
      if (R.dashed) ctx.setLineDash([7 / CAM.s, 5 / CAM.s]);
      ctx.stroke(R.path);
    });
  }
  function drawGlyph(o) {
    const [x, y] = P(o.R.lp[0], o.R.lp[1]); if (x < -30 || x > W + 30 || y < -30 || y > H + 30) return;
    const s = clamp(CAM.s * 1.6, 11, 22);
    ctx.save(); ctx.globalAlpha = o.glyphA ?? 1;
    ctx.beginPath(); ctx.arc(x, y, s * 0.85, 0, 7); ctx.fillStyle = PAL.halo; ctx.fill();
    ctx.strokeStyle = o.glyph === 'hit' ? PAL.hit : o.glyph === 'miss' ? PAL.miss : PAL.label; ctx.lineWidth = s * 0.2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath();
    if (o.glyph === 'hit') { ctx.moveTo(x - s * 0.42, y); ctx.lineTo(x - s * 0.1, y + s * 0.32); ctx.lineTo(x + s * 0.45, y - s * 0.34); }
    else if (o.glyph === 'miss') { ctx.moveTo(x - s * 0.32, y - s * 0.32); ctx.lineTo(x + s * 0.32, y + s * 0.32); ctx.moveTo(x + s * 0.32, y - s * 0.32); ctx.lineTo(x - s * 0.32, y + s * 0.32); }
    else { ctx.moveTo(x - s * 0.3, y); ctx.lineTo(x + s * 0.3, y); }
    ctx.stroke(); ctx.restore();
  }
  function init(canvas) { cv = canvas; ctx = cv.getContext('2d'); readPalette(); resize(); }
  return {
    CAM, init, resize, readPalette, draw, P, unproject, view, kx, constrain, flyCam, camFor, unionBB, minS,
    setGeo, setRegions, prepRegion, setGrid, inRegion, pointQuery, get REG() { return REG; }, get XREG() { return XREG; },
    get W() { return W; }, get H() { return H; }, hasGrid: (s) => !!GRID[s], setSST: (im) => { SST = im; }, prColor, prAlpha,
  };
})();
