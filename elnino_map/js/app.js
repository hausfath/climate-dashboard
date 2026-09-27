/* El Niño impacts map: state, interaction, and wiring between the canvas (engine.js) and the HTML cards (cards.js). */
'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const E = Engine, CAM = E.CAM;
  const NARROW = () => innerWidth <= 820;
  const SEASONS = [
    { id: 'SON', label: 'Sep–Nov', sub: '2026', win: [8, 3] },
    { id: 'OND', label: 'Oct–Dec', sub: '2026', win: [9, 3] },
    { id: 'DJF', label: 'Dec–Feb', sub: '2026–27', win: [11, 3] },
    { id: 'MAM', label: 'Mar–May', sub: '2027', win: [14, 3] },
  ];
  // blog order (Asia & Pacific, Africa, South America, North & Central America); used for prev/next and the fallback tour
  const ORDER = ['maritime', 'philippines', 'wpacific', 'cpacific', 'schina', 'yangtze', 'mekong', 'srilanka', 'seaustralia', 'hawaii',
    'horn', 'safrica', 'amazon', 'nsam', 'nebrazil', 'peru', 'altiplano', 'cchile', 'sesa',
    'gulf', 'antilles', 'swus', 'norcal', 'inlandnw', 'ccanada_warm', 'drycorridor'];
  const FALLBACK_TOUR = [
    { id: 'pacific', title: 'Asia, Australia and the Pacific', regions: ORDER.slice(0, 10), text: 'Where the rising branch of the Walker circulation normally sits, and where El Niño’s fingerprint is clearest.' },
    { id: 'africa', title: 'Africa', regions: ['horn', 'safrica'], text: 'A wet autumn in the Horn of Africa and a dry summer in southern Africa.' },
    { id: 'samerica', title: 'South America', regions: ORDER.slice(12, 19), text: 'Drought in the north and the Amazon, floods on the Peruvian coast and in the southeast.' },
    { id: 'namerica', title: 'North and Central America', regions: ORDER.slice(19), text: 'A wet southern tier, a dry and mild north, and a drier Central America.' },
  ];

  const S = {
    season: 'OND', fieldFrom: null, fieldT0: 0, field: true, dots: true, sst: false, filter: 'all', extras: false,
    replay: null, sel: null, selExtra: false, hover: null, tour: null, tourRegions: null, fly: null, pin: null, play: null, lastDismiss: -1e9,
  };
  let META, LIT = {}, REG, XREG, cur = {};

  // ------------------------------------------------------------------ loading
  const j = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
  const gridLoads = {};
  function loadGrid(id) {
    if (id === 'JJA' || E.hasGrid(id)) return Promise.resolve();
    return (gridLoads[id] ??= fetch(`data/grid_${id}.bin`).then((r) => r.arrayBuffer()).then((b) => { E.setGrid(id, b, META.seasons[id].n); render(); }));
  }
  async function boot() {
    E.init($('map'));
    const [geo, regs, meta, lit] = await Promise.all([j('data/geo.json'), j('data/regions.json'), j('data/meta.json'), j('data/lit.json').catch(() => ({}))]);
    META = meta; LIT = lit;
    E.setGeo(geo); E.setRegions(regs); REG = E.REG; XREG = E.XREG;
    for (const k of ['nindia', 'lit_ohio', 'c_uk_ceurope', 'c_scandinavia']) if (XREG[k]?.rings) prepExtra(XREG[k]);
    for (const k of ORDER) if (!REG[k]) console.warn('region missing from data:', k);
    checkByEvent();
    await loadGrid(S.season);
    const sst = new Image(); sst.onload = () => { E.setSST(sst); render(); }; sst.src = 'img/sst_DJF.png';
    buildUI(); home(false); render();
    $('loading').classList.add('done');
    setTimeout(() => SEASONS.forEach((s) => loadGrid(s.id)), 1200);
    const q = new URLSearchParams(location.search).get('region');
    if (q && REG[q]) openRegion(q, { fly: true });
  }
  function prepExtra(R) {
    // extras reuse the region machinery; neutral fill, always dashed
    E.prepRegion(R); R.fill = '#8A93A6'; R.dashed = true; R.extra = true;
  }
  // replay counts must match the published per-event tally
  function checkByEvent() {
    for (const y of META.strong_events) {
      let h = 0, n = 0;
      for (const k in REG) { const e = REG[k].hits?.events.find((x) => x.year === y); if (e && e.hit !== null && e.pct_normal !== null) { n++; if (e.hit) h++; } }
      const [H0, N0] = META.by_event[y]; if (h !== H0 || n !== N0) console.warn(`by_event mismatch ${y}: ${h}/${n} vs ${H0}/${N0}`);
    }
  }

  // ------------------------------------------------------------------ geometry helpers
  function pads() {
    const p = { l: 0, r: 0, t: 0, b: $('bar').offsetHeight + 24 };
    if (NARROW()) {
      if (!$('tour').hidden) p.b += $('tour').offsetHeight + 12;
      p.t = $('dock').offsetHeight + 16;
      if (!$('card').hidden) p.b = Math.max(p.b, $('card').offsetHeight);
    } else {
      if (!$('tour').hidden || !$('dock').classList.contains('collapsed')) p.l = $('dock').offsetWidth + 24; else p.t = $('dock').offsetHeight + 16;
      if (!$('card').hidden) p.r = $('card').offsetWidth + 24;
    }
    return p;
  }
  function flyTo(to, dur) {
    const from = { lon: CAM.lon, lat: CAM.lat, s: CAM.s };
    const d = Math.hypot(wrap180(to.lon - from.lon), to.lat - from.lat) + 40 * Math.abs(Math.log(to.s / from.s));
    S.fly = { from, to, t0: performance.now(), dur: REDUCED ? 1 : dur ?? clamp(650 + d * 7, 800, 1900) };
    render();
  }
  function frame(keys, opts = {}) {
    const bbs = keys.map((k) => (REG[k] ?? XREG[k]).bb);
    flyTo(E.camFor(E.unionBB(bbs), pads(), opts.maxS ?? 11), opts.dur);
  }
  function home(anim = true) {
    // whole world, Pacific-centred so every region is on screen at once
    const cam = E.constrain({ lon: 160, lat: 4, s: NARROW() ? Math.max(E.minS() * 2.3, E.H / 175) : E.minS() });
    if (NARROW()) cam.lon = 175;
    if (anim) flyTo(cam); else Object.assign(CAM, cam);
  }
  function inSeason(R) {
    const w = R.hits ? seasonWindow(R.hits.season) : null, sw = SEASONS.find((s) => s.id === S.season).win;
    if (!w) return true;
    return w[0] < sw[0] + sw[1] && sw[0] < w[0] + w[1];
  }
  const passes = (R) => S.filter === 'all' || kindOf(R) === S.filter;
  function regionAt(lon, lat) {
    const hits = [];
    for (const k in REG) if (passes(REG[k]) && E.inRegion(REG[k], lon, lat)) hits.push(k);
    if (S.extras) for (const k of Object.keys(XREG)) if (XREG[k].path && E.inRegion(XREG[k], lon, lat)) hits.push(k);
    // smallest first, so a small region on top of a big one stays clickable
    hits.sort((a, b) => area(REG[a] ?? XREG[a]) - area(REG[b] ?? XREG[b]));
    return hits[0] ?? null;
  }
  const area = (R) => (R.bb[2] - R.bb[0]) * (R.bb[3] - R.bb[1]);

  // ------------------------------------------------------------------ render loop
  let raf = 0;
  function render() { if (!raf) raf = requestAnimationFrame(tick); }
  function tick(now) {
    raf = 0;
    let busy = false;
    if (S.fly) {
      const p = clamp((now - S.fly.t0) / S.fly.dur);
      Object.assign(CAM, E.flyCam(S.fly.from, S.fly.to, p)); E.constrain();
      if (p >= 1) S.fly = null; else busy = true;
    }
    let mix = 1;
    if (S.fieldFrom) { mix = clamp((now - S.fieldT0) / (REDUCED ? 1 : 450)); if (mix >= 1) S.fieldFrom = null; else busy = true; }
    const layers = [];
    const all = [...Object.keys(REG).map((k) => [k, REG[k]]), ...(S.extras ? Object.keys(XREG).filter((k) => XREG[k].path).map((k) => [k, XREG[k]]) : [])];
    for (const [k, R] of all) {
      const vis = R.extra ? 1 : passes(R) ? 1 : 0;
      const seasonOk = S.replay || R.extra ? true : inSeason(R);
      const focus = S.tourRegions ? S.tourRegions.has(k) : S.sel ? S.sel === k : true;
      const ev = S.replay ? R.hits?.events.find((x) => x.year === S.replay) : null;
      const tgt = {
        a: vis,
        f: (R.extra ? 0.22 : 1) * (seasonOk ? 1 : 0.3) * (focus ? 1 : 0.42) * (S.sel === k ? 0.7 : 1),
        hi: S.sel === k ? 1 : S.hover === k ? 0.55 : S.tourRegions?.has(k) ? 0.5 : 0,
      };
      const c = (cur[k] ??= { ...tgt });
      for (const q of ['a', 'f', 'hi']) { const d = tgt[q] - c[q]; if (Math.abs(d) > 0.004) { c[q] += d * (REDUCED ? 1 : 0.2); busy = true; } else c[q] = tgt[q]; }
      let fill = R.fill, glyph = null;
      if (S.replay && !R.extra) {
        const PALc = getComputedStyle(document.documentElement);
        if (ev && ev.hit !== null && ev.pct_normal !== null) { fill = PALc.getPropertyValue(ev.hit ? '--hit' : '--miss').trim(); glyph = ev.hit ? 'hit' : 'miss'; }
        else { fill = '#8A93A6'; glyph = 'none'; }
      }
      layers.push({ R, a: c.a, fillA: (+getVar('--map-fill-a')) * c.f, hi: c.hi, fill, glyph, glyphA: c.a * (focus ? 1 : 0.5), edgeA: 0.85 * (seasonOk ? 1 : 0.55) });
    }
    E.draw({
      season: S.season === 'JJA' ? null : S.season, fieldFrom: S.fieldFrom, fieldMix: mix,
      fieldA: S.field && !S.replay && S.season !== 'JJA' ? 1 : 0, dotsA: S.dots && !S.replay && S.season !== 'JJA' ? 0.9 * clamp((CAM.s / E.minS() - 1.25) / 0.9) : 0, sstA: S.sst ? 0.85 : 0, regions: layers,
    });
    placeChips(); placePin();
    if (busy) render();
  }
  const varCache = {};
  function getVar(k) { return (varCache[k] ??= getComputedStyle(document.documentElement).getPropertyValue(k).trim()); }

  // ------------------------------------------------------------------ zoom chips: context boxes that appear as you zoom in
  const chipEls = {};
  function chipFor(k, R) {
    if (chipEls[k]) return chipEls[k];
    const el = document.createElement('button'); el.type = 'button'; el.className = 'rchip';
    const H0 = R.hits, M = R.models, kd = kindOf(R);
    el.innerHTML = `<i class="rbar ${kd === 'cold' ? 'wet' : kd}"></i><span class="t">${esc(R.title)}</span><span class="n">${H0 ? `<span class="h" title="past strong El Niños ${moreWord(kd)} than a typical neutral year">${H0.hits}/${H0.n}</span>` : ''}${M ? `<span title="forecast models leaning ${M.sign < 0 ? 'drier' : 'wetter'}">${M.agree}/${M.n}</span>` : ''}</span>`;
    el.setAttribute('aria-label', `${R.title}: open evidence`);
    el.addEventListener('click', (e) => { e.stopPropagation(); openRegion(k, { fly: false }); });
    el.addEventListener('pointerdown', (e) => e.stopPropagation());
    $('chips').appendChild(el); chipEls[k] = el; return el;
  }
  function placeChips() {
    const z = CAM.s / E.minS(), show = z >= 2.1 && !S.fly, a0 = $('app').getBoundingClientRect();
    // panels count as occupied space, so chips never hide underneath them
    const placed = ['dock', 'bar', 'legend', 'card', 'tour', 'replay', 'pop', 'layers'].map((id) => $(id)).filter((el) => !el.hidden && el.offsetParent)
      .map((el) => { const r = el.getBoundingClientRect(); return [r.left - a0.left, r.top - a0.top, r.right - a0.left, r.bottom - a0.top]; });
    const keys = Object.keys(REG).filter((k) => passes(REG[k]));
    const rank = (k) => (k === S.hover ? -1 : 0) + ({ high: 0, medhigh: 1, medium: 2 }[REG[k].conf] ?? 3) + (inSeason(REG[k]) ? 0 : 5);
    keys.sort((a, b) => rank(a) - rank(b));
    for (const k of Object.keys(chipEls)) if (!keys.includes(k)) chipEls[k].classList.remove('show');
    for (const k of keys) {
      const R = REG[k], el = chipFor(k, R);
      if (!show || k === S.sel || S.replay) { el.classList.remove('show'); continue; }
      const [x, y] = E.P(R.lp[0], R.lp[1]), w = el.offsetWidth || 150, h = el.offsetHeight || 30;
      const bx = x - w / 2, by = y - h / 2;
      const off = bx < 4 || by < 4 || bx + w > E.W - 4 || by + h > E.H - 4;
      const hit = placed.some((b) => bx < b[2] && bx + w > b[0] && by < b[3] && by + h > b[1]);
      if (off || hit) { el.classList.remove('show'); continue; }
      placed.push([bx - 6, by - 4, bx + w + 6, by + h + 4]);
      el.style.transform = `translate(${bx.toFixed(1)}px, ${by.toFixed(1)}px)`;
      el.classList.add('show'); el.classList.toggle('dim', !inSeason(R));
    }
  }

  // ------------------------------------------------------------------ cards
  function showCard(html, after) {
    const card = $('card'), body = $('card-body');
    const wasHidden = card.hidden;
    body.innerHTML = html; body.scrollTop = 0; card.hidden = false; $('app').classList.add('card-open');
    if (wasHidden && !REDUCED) { card.classList.add('enter'); requestAnimationFrame(() => requestAnimationFrame(() => card.classList.remove('enter'))); }
    after?.(body);
    $('pop').hidden = true; S.pin = null;
  }
  function closeCard() {
    if ($('card').hidden) return;
    $('card').hidden = true; $('app').classList.remove('card-open'); S.sel = null; S.lastDismiss = performance.now(); render();
  }
  function openRegion(k, o = {}) {
    const R = REG[k] ?? XREG[k]; if (!R) return;
    S.sel = k; S.selExtra = !!R.extra;
    const i = ORDER.indexOf(k), nav = i >= 0 ? { prev: ORDER[(i + ORDER.length - 1) % ORDER.length], next: ORDER[(i + 1) % ORDER.length] } : null;
    if (nav) { nav.prevTitle = REG[nav.prev].title; nav.nextTitle = REG[nav.next].title; }
    showCard(regionCardHTML(k, R, { lit: LIT, meta: META, selYear: S.replay, nav, extra: R.extra }), (body) => {
      animateCard(body);
      body.querySelectorAll('.strip .col[data-year]').forEach((g) => g.addEventListener('click', () => setReplay(+g.dataset.year === S.replay ? null : +g.dataset.year)));
      body.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => openRegion(b.dataset.go, { fly: true })));
    });
    if (o.fly) frame([k]);
    else if (!o.auto) keepVisible(R);
    render();
  }
  // after a click: if the region is tiny, frame it; if the card now covers it, pan (same zoom) so it stays in view
  function keepVisible(R) {
    const [x, y] = E.P(R.lp[0], R.lp[1]), p = pads(), wpx = (R.bb[2] - R.bb[0]) * E.kx();
    if (wpx < 40) return frame([Object.keys(REG).find((k) => REG[k] === R) ?? Object.keys(XREG).find((k) => XREG[k] === R)]);
    if (x >= p.l && x <= E.W - p.r && y >= p.t && y <= E.H - p.b) return;
    const cx = p.l + (E.W - p.l - p.r) / 2, cy = p.t + (E.H - p.t - p.b) / 2;
    flyTo(E.constrain({ lon: CAM.lon + (x - cx) / E.kx(), lat: CAM.lat - (y - cy) / CAM.s, s: CAM.s }), 650);
  }
  function openGlobal() {
    S.sel = null;
    showCard(globalHTML(META, LIT), (body) => {
      body.querySelectorAll('.byev [data-year]').forEach((d) => d.addEventListener('click', () => setReplay(+d.dataset.year)));
      requestAnimationFrame(() => body.querySelectorAll('.byev i').forEach((el, i) => setTimeout(() => { el.style.height = el.dataset.h + 'px'; }, REDUCED ? 0 : 120 + i * 80)));
    });
    render();
  }
  function openHowto() { S.sel = null; showCard(howtoHTML(META)); render(); }

  // ------------------------------------------------------------------ point readout
  function pointAt(x, y) {
    const [lon, lat] = E.unproject(x, y);
    if (Math.abs(lat) > 85) return;
    if (S.season === 'JJA') {
      const pop = $('pop'); pop.className = 'pop panel';
      pop.innerHTML = '<div class="c-kicker neutral">Jun–Aug 2027</div><h3>No forecast this far ahead</h3><p class="note">Seasonal forecasts from the September start end in spring 2027. Pick an earlier season for the model readout.</p><button class="card-close btn btn-icon" type="button" data-close aria-label="Close"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg></button>';
      pop.hidden = false; S.pin = E.unproject(x, y);
      pop.querySelector('[data-close]').addEventListener('click', () => { pop.hidden = true; S.pin = null; render(); });
      placePop(x, y); render(); return;
    }
    const season = S.season;
    const q = E.pointQuery(season, lon, lat); if (!q) return;
    // nearest mapped region (by label point), if reasonably close
    let near = null, bd = 1e9;
    for (const k in REG) { const R = REG[k], d = Math.hypot(wrap180(R.lp[0] - lon) * Math.cos(lat * DEG), R.lp[1] - lat); if (d < bd) { bd = d; near = k; } }
    const nearObj = bd < 25 ? { key: near, title: REG[near].title } : null;
    const pop = $('pop');
    pop.className = 'pop panel'; pop.innerHTML = pointHTML(q, season, META, nearObj); pop.hidden = false;
    S.pin = [lon, lat];
    animateCard(pop);
    pop.querySelector('[data-close]').addEventListener('click', () => { pop.hidden = true; S.pin = null; render(); });
    pop.querySelector('[data-go]')?.addEventListener('click', (e) => openRegion(e.currentTarget.dataset.go, { fly: true }));
    placePop(x, y); render();
  }
  function placePop(x, y) {
    const pop = $('pop'); if (NARROW()) { pop.style.left = ''; pop.style.top = ''; return; }
    const w = pop.offsetWidth, h = pop.offsetHeight, p = pads();
    let px = x + 18, py = y - h / 2;
    if (px + w > E.W - p.r - 8) px = x - w - 18;
    py = clamp(py, 12, E.H - h - 12);
    pop.style.left = px + 'px'; pop.style.top = py + 'px';
  }
  let pinEl = null;
  function placePin() {
    if (!S.pin || $('pop').hidden) { if (pinEl) pinEl.hidden = true; return; }
    if (!pinEl) { pinEl = document.createElement('div'); pinEl.className = 'pin'; $('app').appendChild(pinEl); }
    const [x, y] = E.P(S.pin[0], S.pin[1]); pinEl.hidden = false; pinEl.style.left = x + 'px'; pinEl.style.top = y + 'px';
  }

  // ------------------------------------------------------------------ replay a past event
  function setReplay(y) {
    S.replay = y;
    document.querySelectorAll('#years button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.y === y)));
    const bar = $('replay');
    if (y) {
      const [h, n] = META.by_event[y];
      const misses = Object.keys(REG).filter((k) => { const e = REG[k].hits?.events.find((x) => x.year === y); return e && e.hit === false; }).map((k) => REG[k].title);
      $('replay-year').textContent = `${y}–${String(y + 1).slice(2)}`;
      $('replay-text').innerHTML = `<b>${h} of ${n}</b> regions went the expected way${misses.length ? `<br><span class="x">✗</span> ${misses.length <= 4 ? esc(misses.join(', ')) : `${misses.length} misses`}` : '. Every region delivered.'}`;
      bar.className = 'replay panel'; bar.hidden = false;
    } else bar.hidden = true;
    if (S.sel) openRegion(S.sel, { fly: false });
    updateLegend(); render();
  }

  // ------------------------------------------------------------------ season
  function setSeason(id, fromPlay = false) {
    if (id === S.season) return;
    if (!fromPlay) stopPlay();
    if (S.season !== 'JJA' && id !== 'JJA') { S.fieldFrom = S.season; S.fieldT0 = performance.now(); }
    S.season = id;
    document.querySelectorAll('#seasons [data-s]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.s === id)));
    loadGrid(id).then(() => render());
    if (!$('pop').hidden && S.pin) { const [x, y] = E.P(S.pin[0], S.pin[1]); pointAt(x, y); }
    updateLegend(); render();
  }
  function togglePlay() {
    if (S.play) return stopPlay();
    const btn = document.querySelector('#seasons .play'); btn.classList.add('on'); btn.setAttribute('aria-label', 'Pause');
    btn.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h3v10H4zM9 3h3v10H9z"/></svg>';
    const ids = SEASONS.map((s) => s.id), stepF = () => setSeason(ids[(ids.indexOf(S.season) + 1) % ids.length], true);
    stepF(); S.play = setInterval(stepF, 1900);
  }
  function stopPlay() {
    if (!S.play) return; clearInterval(S.play); S.play = null;
    const btn = document.querySelector('#seasons .play'); btn.classList.remove('on'); btn.setAttribute('aria-label', 'Play through the seasons');
    btn.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
  }

  // ------------------------------------------------------------------ tour
  function tourStops() { return Array.isArray(LIT._tour) && LIT._tour.length ? LIT._tour : FALLBACK_TOUR; }
  function startTour() { closeCard(); setReplay(null); $('dock').classList.add('collapsed'); $('collapse-btn').setAttribute('aria-expanded', 'false'); goStop(0); }
  function goStop(i) {
    const stops = tourStops(); if (i < 0 || i >= stops.length) return endTour();
    const st = stops[i], keys = st.regions.filter((k) => REG[k] || XREG[k]?.path);
    // a stop about regions left off the map switches that layer on
    if (keys.some((k) => !REG[k]) && !S.extras) { S.extras = true; $('lyr-extra').checked = true; }
    S.tour = i; S.tourRegions = new Set(keys);
    $('tour-step').textContent = `Stop ${i + 1} of ${stops.length}`;
    $('tour-title').textContent = st.title; $('tour-text').textContent = st.text;
    $('tour-regions').innerHTML = keys.map((k) => { const R = REG[k] ?? XREG[k]; return `<button type="button" data-k="${k}"><i style="background:${R.fill}"></i>${esc(R.title)}</button>`; }).join('');
    $('tour-regions').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => openRegion(b.dataset.k, { fly: false })));
    $('tour-prev').disabled = i === 0; $('tour-next').textContent = i === stops.length - 1 ? 'Finish' : 'Next';
    $('tour').className = 'tour panel'; $('tour').hidden = false;
    $('tour').style.top = NARROW() ? '' : ($('dock').offsetHeight + 28) + 'px';
    closeCard();
    requestAnimationFrame(() => keys.length && frame(keys, { maxS: 14, dur: 1500 }));
    render();
  }
  function endTour() { S.tour = null; S.tourRegions = null; $('tour').hidden = true; home(); render(); }

  // ------------------------------------------------------------------ auto-open on zoom
  // When you zoom in with the pointer over a region until it fills a good part of the view, its card opens.
  // Never after a pan, never for a region you are not pointing at, and it never moves the camera.
  let moveTimer = 0;
  function zoomedIn(x, y) {
    clearTimeout(moveTimer);
    moveTimer = setTimeout(() => {
      if (!$('card').hidden || S.tour !== null || S.replay || performance.now() - S.lastDismiss < 5000) return;
      if (CAM.s / E.minS() < 2.6) return;
      const [lon, lat] = E.unproject(x, y), k = regionAt(lon, lat); if (!k || !REG[k]) return;
      const R = REG[k], v = E.view(), va = (v.e - v.w) * (v.n - v.s);
      const c0 = CAM.lon + wrap180((R.bb[0] + R.bb[2]) / 2 - CAM.lon), hw = (R.bb[2] - R.bb[0]) / 2;
      const ow = Math.max(0, Math.min(v.e, c0 + hw) - Math.max(v.w, c0 - hw)), oh = Math.max(0, Math.min(v.n, R.bb[3]) - Math.max(v.s, R.bb[1]));
      if (ow * oh / va >= 0.1) openRegion(k, { auto: true });
    }, 520);
  }

  // ------------------------------------------------------------------ pointer input
  const cv = $('map'), ptrs = new Map();
  let down = null, pinch = null, hoverRaf = 0;
  cv.tabIndex = 0;
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.offsetX, e.offsetY]);
    S.fly = null;
    if (ptrs.size === 1) down = { x: e.offsetX, y: e.offsetY, moved: false, t: performance.now() };
    if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), s: CAM.s }; down && (down.moved = true); }
  });
  cv.addEventListener('pointermove', (e) => {
    if (ptrs.has(e.pointerId)) {
      const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.offsetX, e.offsetY]);
      if (pinch && ptrs.size === 2) {
        const [a, b] = [...ptrs.values()], mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const f = pinch.s * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch.d / CAM.s;
        zoomAt(mx, my, f); if (f > 1) zoomedIn(mx, my);
      } else if (down) {
        if (!down.moved && Math.hypot(e.offsetX - down.x, e.offsetY - down.y) > 7) { down.moved = true; cv.classList.add('dragging'); $('tip').hidden = true; }
        if (down.moved) { CAM.lon -= (e.offsetX - prev[0]) / E.kx(); CAM.lat += (e.offsetY - prev[1]) / CAM.s; E.constrain(); render(); }
      }
      return;
    }
    if (hoverRaf) return;
    const x = e.offsetX, y = e.offsetY;
    hoverRaf = requestAnimationFrame(() => { hoverRaf = 0; hoverAt(x, y); });
  });
  const endPtr = (e) => {
    ptrs.delete(e.pointerId);
    if (ptrs.size < 2) pinch = null;
    if (ptrs.size === 0) {
      cv.classList.remove('dragging');
      if (down && !down.moved && e.type === 'pointerup') clickAt(e.offsetX, e.offsetY);
      down = null;
    }
  };
  cv.addEventListener('pointerup', endPtr); cv.addEventListener('pointercancel', endPtr);
  cv.addEventListener('pointerleave', () => { if (S.hover) { S.hover = null; render(); } $('tip').hidden = true; });
  cv.addEventListener('wheel', (e) => {
    e.preventDefault(); S.fly = null;
    const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022));
    zoomAt(e.offsetX, e.offsetY, f); if (f > 1) zoomedIn(e.offsetX, e.offsetY);
  }, { passive: false });
  cv.addEventListener('dblclick', (e) => { e.preventDefault(); const [lon, lat] = E.unproject(e.offsetX, e.offsetY); flyTo(E.constrain({ lon, lat, s: CAM.s * 2 }), 600); setTimeout(() => zoomedIn(E.W / 2, E.H / 2), 650); });
  function zoomAt(x, y, f) {
    const [lo0, la0] = E.unproject(x, y);
    CAM.s *= f; E.constrain();
    const [lo1, la1] = E.unproject(x, y);
    CAM.lon += wrap180(lo0 - lo1); CAM.lat += la0 - la1; E.constrain(); render();
  }
  function clickAt(x, y) {
    const [lon, lat] = E.unproject(x, y), k = regionAt(lon, lat);
    if (k) { $('pop').hidden = true; openRegion(k); }
    else pointAt(x, y);
  }
  function hoverAt(x, y) {
    const [lon, lat] = E.unproject(x, y), k = regionAt(lon, lat), tip = $('tip');
    cv.classList.toggle('over-region', !!k);
    if (k !== S.hover) { S.hover = k; render(); }
    if (!k || NARROW()) { tip.hidden = true; return; }
    const R = REG[k] ?? XREG[k], H0 = R.hits, M = R.models, kd = kindOf(R);
    let body = `<b>${esc(R.title)}</b><span class="m">${esc(R.extra ? 'Tested, not on the map' : R.detail)}</span>`;
    if (S.replay && H0) {
      const ev = H0.events.find((e) => e.year === S.replay);
      body += ev && ev.pct_normal !== null ? `<div class="ev-mini"><span class="${ev.hit ? 'h' : ''}">${ev.hit ? '✓' : '✗'} ${isTemp(R) ? signed(ev.pct_normal, 1) + ' °C' : Math.round(ev.pct_normal) + '% of avg'}</span></div>` : '<div class="ev-mini">no data for this event</div>';
    } else body += `<div class="ev-mini">${H0 ? `<span class="h">${H0.hits}/${H0.n} past</span>` : ''}${M ? `<span>${M.agree}/${M.n} models</span>` : ''}</div>`;
    tip.innerHTML = body; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = clamp(x + 14, 8, E.W - w - 8) + 'px'; tip.style.top = clamp(y - h - 12, 8, E.H - h - 8) + 'px';
  }
  // tooltips for model tiles and past-event bars inside the cards (data-tt = title, data-tb = body HTML we generate)
  function showDataTip(el) {
    const tip = $('tip'); tip.innerHTML = `<b>${esc(el.dataset.tt)}</b>${el.dataset.tb}`; tip.hidden = false;
    const r = el.getBoundingClientRect(), a = $('app').getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let y = r.top - a.top - h - 8; if (y < 8) y = r.bottom - a.top + 8;
    tip.style.left = clamp(r.left - a.left + r.width / 2 - w / 2, 8, E.W - w - 8) + 'px'; tip.style.top = y + 'px';
  }
  const tipEl = (e) => e.target.closest?.('[data-tt]');
  $('app').addEventListener('pointerover', (e) => { const el = tipEl(e); if (el) showDataTip(el); });
  $('app').addEventListener('pointerout', (e) => { const el = tipEl(e); if (el && !el.contains(e.relatedTarget)) $('tip').hidden = true; });
  $('app').addEventListener('focusin', (e) => { const el = tipEl(e); if (el) showDataTip(el); });
  $('app').addEventListener('focusout', (e) => { if (tipEl(e)) $('tip').hidden = true; });
  $('card-body').addEventListener('scroll', () => { $('tip').hidden = true; }, { passive: true });

  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { if (!$('pop').hidden) { $('pop').hidden = true; S.pin = null; render(); } else if (!$('card').hidden) closeCard(); else if (S.tour !== null) endTour(); else if (S.replay) setReplay(null); }
    if (document.activeElement === cv) {
      const st = 60;
      if (e.key === '+' || e.key === '=') zoomAt(E.W / 2, E.H / 2, 1.4);
      if (e.key === '-') zoomAt(E.W / 2, E.H / 2, 1 / 1.4);
      if (e.key === 'ArrowLeft') { CAM.lon -= st / E.kx(); render(); }
      if (e.key === 'ArrowRight') { CAM.lon += st / E.kx(); render(); }
      if (e.key === 'ArrowUp') { CAM.lat += st / CAM.s; E.constrain(); render(); }
      if (e.key === 'ArrowDown') { CAM.lat -= st / CAM.s; E.constrain(); render(); }
    }
    if (S.tour !== null && $('card').hidden && $('layers').hidden && (e.key === 'ArrowRight' || e.key === 'ArrowLeft') && document.activeElement !== cv) goStop(S.tour + (e.key === 'ArrowRight' ? 1 : -1));
  });
  let rsT = 0;
  addEventListener('resize', () => { clearTimeout(rsT); rsT = setTimeout(() => { E.resize(); render(); }, 60); });

  // ------------------------------------------------------------------ theme (dashboard toggle via postMessage, or OS)
  function setTheme(t) {
    if (t !== 'light' && t !== 'dark') return;
    if (document.documentElement.dataset.theme === t) return;
    document.documentElement.dataset.theme = t;
    for (const k in varCache) delete varCache[k];
    E.readPalette(); updateLegend();
    if (S.sel) { openRegion(S.sel, { fly: false }); }
    render();
  }
  addEventListener('message', (e) => { if (e.origin === location.origin && e.data?.source === 'climate-dashboard') setTheme(e.data.theme); });
  if (!new URLSearchParams(location.search).get('theme')) matchMedia('(prefers-color-scheme: light)').addEventListener('change', (m) => setTheme(m.matches ? 'light' : 'dark'));

  // ------------------------------------------------------------------ UI
  function updateLegend() {
    const sd = SEASONS.find((s) => s.id === S.season), M = META.seasons[S.season];
    const cap = $('lg-cap');
    $('lg-field').style.opacity = S.replay ? 0.4 : 1;
    if (S.replay) cap.textContent = `Replaying ${S.replay}–${String(S.replay + 1).slice(2)}: green went the expected way, red did not.`;
    else cap.innerHTML = `Model-mean rainfall change, ${esc(M.label)}, % of normal (${M.n} models)<span class="dotk"></span>dots: ≥80% agree`;
    // colour bar drawn from the same colormap and alpha as the map, over the land colour
    let c = $('lg-field').querySelector('canvas');
    if (!c) { c = document.createElement('canvas'); c.width = 220; c.height = 8; c.className = 'lg-bar'; $('lg-field').querySelector('.lg-bar').replaceWith(c); }
    const g = c.getContext('2d'); g.fillStyle = getVar('--map-land'); g.fillRect(0, 0, 220, 8);
    for (let x = 0; x < 220; x++) { const v = -60 + 120 * (x + 0.5) / 220, [r, gg, b] = E.prColor(v); g.fillStyle = `rgba(${r | 0},${gg | 0},${b | 0},${E.prAlpha(v) * (+getVar('--map-field-a'))})`; g.fillRect(x, 0, 1, 8); }
    document.querySelectorAll('.legend .tier').forEach((el) => { el.style.background = TIER[el.dataset.kind][el.dataset.conf]; });
  }
  function toggleLayers(open) {
    const L = $('layers'), b = $('layers-btn'), on = open ?? L.hidden;
    L.hidden = !on; b.setAttribute('aria-expanded', String(on));
    if (on && !NARROW()) {
      const r = b.getBoundingClientRect(), a = $('app').getBoundingClientRect();
      L.style.left = clamp(r.right - a.left - L.offsetWidth, 8, E.W - L.offsetWidth - 8) + 'px';
      L.style.top = (r.top - a.top - L.offsetHeight - 22) + 'px';
    }
  }
  function buildUI() {
    const n = META.n_map_regions, ne = META.strong_events.length, A = META.aggregate;
    $('lede').innerHTML = `<b>${n} regions</b> where the literature expects a strong El Niño to shift rainfall or temperature, each checked against the <b>${ne} strong El Niños since ${META.strong_events[0]}</b> and this year's <b>${META.seasons.OND.n} seasonal forecast models</b>. Click a region, or anywhere on the map, for the evidence.`;
    const gp = document.createElement('button'); gp.type = 'button'; gp.className = 'btn'; gp.id = 'global-btn';
    gp.innerHTML = `The global picture`;
    gp.title = `${A.hits} of ${A.n} past region-events went the expected way`;
    $('howto-btn').after(gp); gp.addEventListener('click', openGlobal);
    $('init-note').textContent = `models started ${META.init}`;
    $('seasons').innerHTML = SEASONS.map((s) => `<button type="button" role="radio" data-s="${s.id}" aria-checked="${s.id === S.season}" class="${s.nofield ? 'nofield' : ''}" title="${s.nofield ? 'Beyond the range of seasonal forecasts; shows which regions are in season' : `${s.label} ${s.sub}`}">${s.label}<small>${s.sub}</small></button>`).join('')
      + '<button type="button" class="play" aria-label="Play through the seasons"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg></button>';
    $('seasons').querySelectorAll('[data-s]').forEach((b) => b.addEventListener('click', () => setSeason(b.dataset.s)));
    $('seasons').querySelector('.play').addEventListener('click', togglePlay);
    $('years').innerHTML = META.strong_events.map((y) => `<button type="button" data-y="${y}" aria-pressed="false" title="${y}–${String(y + 1).slice(2)}: ${META.by_event[y][0]} of ${META.by_event[y][1]} regions went the expected way">${y}<small>${META.by_event[y][0]}/${META.by_event[y][1]}</small></button>`).join('');
    $('years').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => setReplay(+b.dataset.y === S.replay ? null : +b.dataset.y)));
    $('filter').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      S.filter = b.dataset.v; $('filter').querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b))); render();
    }));
    $('layers-btn').addEventListener('click', (e) => { e.stopPropagation(); toggleLayers(); });
    document.addEventListener('pointerdown', (e) => { if (!$('layers').hidden && !$('layers').contains(e.target) && e.target !== $('layers-btn')) toggleLayers(false); });
    $('lyr-field').addEventListener('change', (e) => { S.field = e.target.checked; render(); });
    $('lyr-dots').addEventListener('change', (e) => { S.dots = e.target.checked; render(); });
    $('lyr-sst').addEventListener('change', (e) => { S.sst = e.target.checked; render(); });
    $('lyr-extra').addEventListener('change', (e) => { S.extras = e.target.checked; if (!S.extras && S.selExtra) closeCard(); render(); });
    $('card-close').addEventListener('click', closeCard);
    $('replay-close').addEventListener('click', () => setReplay(null));
    $('tour-btn').addEventListener('click', startTour);
    $('tour-next').addEventListener('click', () => goStop(S.tour + 1));
    $('tour-prev').addEventListener('click', () => goStop(S.tour - 1));
    $('tour-exit').addEventListener('click', endTour);
    $('howto-btn').addEventListener('click', openHowto);
    $('collapse-btn').addEventListener('click', () => { const c = $('dock').classList.toggle('collapsed'); $('collapse-btn').setAttribute('aria-expanded', String(!c)); render(); });
    $('zoom-in').addEventListener('click', () => { zoomAt(E.W / 2, E.H / 2, 1.5); zoomedIn(E.W / 2, E.H / 2); });
    $('zoom-out').addEventListener('click', () => zoomAt(E.W / 2, E.H / 2, 1 / 1.5));
    $('zoom-home').addEventListener('click', () => { closeCard(); home(); });
    ['dock', 'bar', 'layers', 'card', 'pop', 'tour', 'legend', 'replay'].forEach((id) => $(id).classList.add('panel'));
    if (NARROW()) { $('dock').classList.add('collapsed'); $('collapse-btn').setAttribute('aria-expanded', 'false'); }
    updateLegend();
  }

  window.__map = { S, CAM, E, chips: () => Object.entries(chipEls).filter(([, el]) => el.classList.contains('show')).map(([k, el]) => [k, el.style.transform]) };
  boot().catch((err) => { console.error(err); $('loading').textContent = `The map could not load its data (${err.message}). Reload the page to try again.`; });
})();
