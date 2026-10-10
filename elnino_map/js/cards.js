/* HTML cards: region evidence (observed strip + model tiles + literature), point readout, global summary, how-to.
   Every number is read from data/*.json; nothing here is typed in by hand. */
'use strict';
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const minus = (v) => String(v).replace('-', '−');
const signed = (v, d = 0) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
// 'normal': magnitudes vs the 1991–2020 average (white line = typical neutral year).
// 'typical': magnitudes vs a typical neutral year (white line = 1991–2020 average). Set from meta.baseline.
let BASE = 'normal', BARS = 'normal';
const setBaseline = (b, bars) => { BASE = b === 'typical' ? 'typical' : 'normal'; BARS = bars === 'typical' ? 'typical' : 'normal'; };
const REF = () => (BASE === 'typical' ? 'a typical neutral year' : 'normal');

// observed-season codes from hit_rates/analyze.py ("SOND", "DJFMA", "MAM1" = year after onset, "ON0" = onset year)
// -> months since Jan of the onset year: [start, length]
function seasonWindow(code) {
  const d = code.match(/\d$/)?.[0], s = code.replace(/\d$/, '');
  if (d === '1') { const i = 'JFMAMJJASOND'.indexOf(s); return i < 0 ? null : [12 + i, s.length]; }
  const i = 'JFMAMJJASONDJFMAMJJASOND'.indexOf(s); return i < 0 ? null : [i, s.length];
}
function windowLabel(w, y0 = 2026) {
  if (!w) return '';
  const [a, n] = w, b = a + n - 1, ya = y0 + Math.floor(a / 12), yb = y0 + Math.floor(b / 12);
  return `${MON[a % 12]}–${MON[b % 12]} ${ya === yb ? ya : `${ya}–${String(yb).slice(2)}`}`;
}
const eventLabel = (y, w) => (w && Math.floor((w[0] + w[1] - 1) / 12) > Math.floor(w[0] / 12) ? `${y}–${String(y + 1).slice(2)}` : w && w[0] >= 12 ? `${y + 1}` : `${y}`);

// expected direction words
function kindOf(R) {
  if (R.kind) return R.kind;
  const H = R.hits; if (!H) return 'dry';
  return H.source === 'berkeley' ? (H.sign > 0 ? 'warm' : 'cold') : H.sign < 0 ? 'dry' : 'wet';
}
const isTemp = (R) => R.hits?.source === 'berkeley';
const moreWord = (k) => ({ dry: 'drier', wet: 'wetter', warm: 'warmer', cold: 'colder' }[k]);

function tileColor(p) {
  const cs = getComputedStyle(document.documentElement), v = (k) => cs.getPropertyValue(k).trim(), q = clamp(Math.abs(p) / 60);
  return p < 0 ? mixHex(v('--tile-dry-lo'), v('--tile-dry-hi'), q) : mixHex(v('--tile-wet-lo'), v('--tile-wet-hi'), q);
}

// ---------------------------------------------------------------- observed strip (SVG)
function stripSVG(R, selYear) {
  const H0 = R.hits, ev = H0.events, T = isTemp(R);
  const TY = BARS === 'typical';
  const bar = (e) => (TY ? (T ? e.pct_typ : e.pct_typ - 100) : (T ? e.pct_normal : e.pct_normal - 100));
  // marker: the other reference (typical neutral year on the 1991–2020 scale, or the 1991–2020 average on the typical scale)
  const mk = (e) => (TY ? (T ? e.pct_typ - e.pct_normal : e.typ_level ? 10000 / e.typ_level - 100 : null)
    : (T ? (e.pct_normal - e.pct_typ) : (e.typ_level ?? null) === null ? null : e.typ_level - 100));
  const cl = (v) => clamp(v, T ? -8 : -95, T ? 8 : 200);
  const ok = ev.filter((e) => e.pct_normal !== null && e.hit !== null);
  const vals = ok.flatMap((e) => [cl(bar(e)), ...(mk(e) === null ? [] : [cl(mk(e))])]);
  const up = Math.max(0, ...vals), dn = Math.max(0, ...vals.map((v) => -v));
  const X0 = 40, X1 = 380, plotH = 118, sc = plotH / Math.max(up + dn, T ? 3 : 40), base = 20 + up * sc, bot = base + dn * sc + 18;
  const cw = (X1 - X0) / ev.length, bw = cw * 0.56, hY = bot + 16, gY = bot + 38, Ht = gY + 14;
  const w = windowLabel(seasonWindow(H0.season));
  let s = `<svg class="strip" viewBox="0 0 390 ${Ht}" role="img" aria-label="Observed ${esc(R.title)} in the ${ev.length} strong El Niños">`;
  s += `<line class="base" x1="${X0}" x2="${X1}" y1="${base}" y2="${base}"/><text class="axis" x="${X0 - 6}" y="${base + 3.5}" text-anchor="end">${T ? '0 °C' : '100%'}</text>`;
  ev.forEach((e, i) => {
    const cx = X0 + cw * (i + 0.5), yl = `'${String(e.year).slice(2)}`;
    const evl = eventLabel(e.year, seasonWindow(H0.season)), ok = e.pct_normal !== null && e.hit !== null;
    let tb = 'No data for this event';
    if (ok) {
      const val = BARS === 'typical'
        ? (T ? `<span class="tv">${signed(e.pct_typ, 1)} °C</span> vs a typical neutral year<br>${signed(e.pct_normal, 1)} °C vs the 1991–2020 average`
          : `<span class="tv ${e.pct_typ < 100 ? 'd' : 'w'}">${Math.round(e.pct_typ)}%</span> of a typical neutral year<br>${Math.round(e.pct_normal)}% of the 1991–2020 average`)
        : (T ? `<span class="tv">${signed(e.pct_normal, 1)} °C</span> vs the 1991–2020 average<br>${signed(e.pct_typ, 1)} °C vs a typical neutral year`
          : `<span class="tv ${e.pct_normal < 100 ? 'd' : 'w'}">${Math.round(e.pct_normal)}%</span> of the 1991–2020 average<br>A typical neutral year would be ${Math.round(e.typ_level)}%`);
      tb = `${val}<br><span class="${e.hit ? 'h' : 'x'}">${e.hit ? '✓' : '✗ not'} ${moreWord(kindOf(R))} than a typical neutral year</span><br><span class="m">Click to replay ${evl} on the map</span>`;
    }
    s += `<g class="col${e.year === selYear ? ' sel' : ''}" data-year="${e.year}" style="--i:${i}" data-tt="${esc(evl)}" data-tb="${esc(tb)}">`;
    s += `<rect class="hl" x="${cx - cw / 2 + 1}" y="0" width="${cw - 2}" height="${Ht}" rx="6" fill-opacity="${e.year === selYear ? 1 : 0}"/>`;
    if (!ok) {
      s += `<text class="axis" x="${cx}" y="${base + 3.5}" text-anchor="middle">n/a</text><text class="yr" x="${cx}" y="${hY}">${yl}</text></g>`;
      return;
    }
    const d = cl(bar(e)) * sc, dir = d >= 0 ? 'up' : 'down', kd = T ? (bar(e) < 0 ? 'cool' : 'warmb') : bar(e) < 0 ? 'dry' : 'wet';
    const lab = T ? `${signed(bar(e), 1)}°` : `${Math.round(bar(e) + 100)}%`;
    const m = mk(e), my = m === null ? null : base - cl(m) * sc;
    const above = m !== null ? m < bar(e) : d >= 0, endY = base - d;
    const ly = above ? Math.min(endY, base) - 5 : Math.max(endY, base) + 12;
    s += `<rect class="bar ${kd} ${dir} pending" x="${cx - bw / 2}" y="${d >= 0 ? base - d : base}" width="${bw}" height="${Math.max(Math.abs(d), 1)}" rx="3.5"/>`;
    if (my !== null) s += `<line class="mk pending-o" x1="${cx - cw * 0.4}" x2="${cx + cw * 0.4}" y1="${my}" y2="${my}"/>`;
    s += `<text class="val pending-o" x="${cx}" y="${ly}">${lab}</text>`;
    s += `<text class="yr" x="${cx}" y="${hY}">${yl}</text>`;
    s += e.hit ? `<path class="glyph g-hit pending-g" d="M${cx - 6.5} ${gY} l4.5 4.5 l8.5 -9"/>` : `<path class="glyph g-miss pending-g" d="M${cx - 5} ${gY - 5} l10 10 M${cx + 5} ${gY - 5} l-10 10"/>`;
    s += '</g>';
  });
  s += '</svg>';
  return { svg: s, window: w };
}

function modelTilesHTML(M, members, opts = {}) {
  const n = members.length;
  return `<div class="tiles" style="--n:${n}">${members.map((m, i) => {
    const p = m.pct, agree = m.agree ?? true;
    const body = p === null ? 'No value for this cell' : `<span class="tv ${p < 0 ? 'd' : 'w'}">${signed(Math.round(p))}%</span> rainfall vs ${REF()}${opts.season ? `, ${esc(opts.season)}` : ''}<br><span class="m">${esc(opts.where ?? '')}${agree ? '' : `${opts.where ? ' · ' : ''}goes against the model majority`}</span>`;
    return `<div class="tile pending${agree ? '' : ' no'}" style="--i:${i};background:${p === null ? 'transparent' : tileColor(p)}" data-tt="${esc(m.label)}" data-tb="${esc(body)}" tabindex="0" aria-label="${esc(m.label)}: ${p === null ? 'no value' : `${signed(Math.round(p))}% rainfall vs ${REF()}`}">${p === null ? '·' : p > 0 ? '+' : '−'}</div>`;
  }).join('')}</div>`;
}

// reveal animation: bars rise one by one, the counter ticks as each check lands, then tiles pop in
function animateCard(root) {
  const cols = [...root.querySelectorAll('.strip .col')], counter = root.querySelector('[data-count="obs"]');
  const tiles = [...root.querySelectorAll('.tile.pending')], mcount = root.querySelector('[data-count="models"]');
  const step = REDUCED ? 0 : 110, t0 = REDUCED ? 0 : 220;
  let hits = 0;
  cols.forEach((c, i) => setTimeout(() => {
    c.querySelectorAll('.pending').forEach((el) => el.classList.remove('pending'));
    c.querySelectorAll('.pending-o').forEach((el) => el.classList.remove('pending-o'));
    setTimeout(() => {
      const g = c.querySelector('.pending-g'); if (!g) return;
      g.classList.remove('pending-g');
      if (g.classList.contains('g-hit') && counter) counter.textContent = ++hits;
    }, REDUCED ? 0 : 260);
  }, t0 + i * step));
  const tt = t0 + cols.length * step + (REDUCED ? 0 : 300);
  let ag = 0;
  tiles.forEach((el, i) => setTimeout(() => {
    el.classList.remove('pending');
    if (mcount && !el.classList.contains('no') && el.textContent !== '·') mcount.textContent = ++ag;
  }, tt + i * (REDUCED ? 0 : 45)));
}

function litHTML(L) {
  if (!L) return '';
  let s = '<div class="lit">';
  if (L.why) s += `<h3>Why it happens</h3><p>${esc(L.why)}</p>`;
  if (L.past) s += `<h3>In past events</h3><p>${esc(L.past)}</p>`;
  if (L.caveat) s += `<h3>What could make it miss</h3><p>${esc(L.caveat)}</p>`;
  if (L.links?.length) s += `<ul class="refs">${L.links.map((l) => `<li><a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a></li>`).join('')}</ul>`;
  return s + '</div>';
}

function regionCardHTML(key, R, ctx) {
  const L = ctx.lit?.[key], k = kindOf(R), H0 = R.hits, M = R.models, off = !!ctx.extra;
  const w = H0 ? seasonWindow(H0.season) : null;
  let s = `<div class="c-kicker ${k === 'cold' ? 'wet' : k}"><span>${esc(off ? `Tested, not on the map · ${moreWord(k)} · ${windowLabel(w)}` : R.detail)}</span></div>`;
  s += `<h2>${esc(R.title)}</h2><div class="c-chips">`;
  if (!off) s += `<span class="chip"><i style="background:${R.fill}"></i>${CONF_LABEL[R.conf]} confidence</span>`;
  if (R.dashed && !off) s += '<span class="chip dashed">Dashed: underperformed in a recent strong event</span>';
  s += '</div>';
  if (L?.headline) s += `<p class="c-headline">${esc(L.headline)}</p>`;
  if (H0) {
    const st = stripSVG(R, ctx.selYear);
    const lan = H0.lanina_same_dir?.split('/').map(Number);
    const y0 = H0.events.find((e) => e.pct_normal !== null && e.hit !== null)?.year ?? H0.events[0].year;
    s += `<section class="ev"><div class="ev-head"><div class="ev-lbl">${H0.n === H0.events.length ? `The ${H0.n} strong El Niños since ${y0}` : `The ${H0.n} strong El Niños with data (since ${y0})`} · ${esc(st.window)}</div>`;
    s += `<div class="count"><b data-count="obs">${REDUCED ? H0.hits : 0}</b><span>/${H0.n}</span><small>${moreWord(k)} than a typical neutral year</small></div></div>`;
    s += st.svg;
    s += BARS === 'typical'
      ? `<div class="strip-note"><span>Bars: ${isTemp(R) ? '°C vs a typical neutral year' : '% of a typical neutral year'}</span><span><i class="mkk"></i>1991–2020 average</span></div>`
      : `<div class="strip-note"><span>Bars: ${isTemp(R) ? '°C vs the 1991–2020 average' : '% of the 1991–2020 average'}</span><span><i class="mkk"></i>typical neutral year</span></div>`;
    if (lan) s += `<p class="spec">La Niña years were also ${moreWord(k)} than a typical neutral year in <b>${lan[0]} of ${lan[1]}</b>. ${H0.hits / H0.n < 0.6 ? '' : lan[0] / lan[1] <= 0.25 ? 'That makes this a signal specific to El Niño.' : lan[0] / lan[1] >= 0.5 ? 'So part of this record may reflect variability that shows up in other years too, not El Niño specifically.' : ''}</p>`;
    s += '</section>';
  }
  if (M) {
    const S = ctx.meta.seasons[M.season], dirw = M.sign < 0 ? 'drier' : 'wetter', cls = M.agree / M.n >= 0.8 ? (M.sign < 0 ? '' : ' wet') : ' mixed';
    const src = M.ens === 'nmme' ? `NMME only (Copernicus runs end in ${ctx.meta.model_horizon?.all?.split(' ')[0] ?? 'Feb'})` : (S?.source ?? 'NMME + Copernicus C3S');
    s += `<section class="ev"><div class="ev-head"><div class="ev-lbl">This year's ${M.n} forecast models · ${esc(M.window ?? S?.label ?? M.season)}</div>`;
    s += `<div class="count models${cls}"><b data-count="models">${REDUCED ? M.agree : 0}</b><span>/${M.n}</span><small>lean ${dirw}</small></div></div>`;
    s += modelTilesHTML(M, M.members, { season: S?.label, where: 'averaged over the region' });
    s += `<div class="tiles-foot"><span>Model average <b>${signed(Math.round(M.mmm_pct))}%</b> vs ${REF()}</span><span>${esc(src)}, ${esc(ctx.meta.init)} start</span></div>${M.note ? `<p class="nomodels" style="margin-top:8px;font-size:12px">${esc(M.note)}</p>` : ''}</section>`;
  } else if (!off) {
    s += `<section class="ev"><div class="ev-lbl">This year's forecast models</div><p class="nomodels">${isTemp(R) ? 'Temperature region: the model check on this map covers rainfall only.' : 'This window is beyond the range of current seasonal forecasts.'}</p></section>`;
  }
  s += litHTML(L);
  if (ctx.nav) s += `<div class="c-nav"><button class="btn" type="button" data-go="${ctx.nav.prev}">← ${esc(ctx.nav.prevTitle)}</button><button class="btn" type="button" data-go="${ctx.nav.next}">${esc(ctx.nav.nextTitle)} →</button></div>`;
  return s;
}

function pointHTML(q, season, meta, near) {
  const S = meta.seasons[season], latS = `${Math.abs(q.lat)}°${q.lat >= 0 ? 'N' : 'S'}`, lonS = `${Math.abs(q.lon)}°${q.lon >= 0 ? 'E' : 'W'}`;
  let s = `<div class="c-kicker neutral">Model forecast · ${esc(S.label)}</div><h3>${near?.inside ? esc(near.title) : 'At this point'}</h3><div class="coord">${latS}, ${lonS} · 1° grid cell</div>`;
  if (q.mean === null) {
    s += '<p class="note">Normal rainfall here is below 0.5 mm/day in this season, so a percentage change is not meaningful.</p>';
  } else {
    const dry = q.mean < 0, agree = q.vals.filter((v) => v !== null && Math.sign(v) === Math.sign(q.mean)).length, n = q.vals.filter((v) => v !== null).length;
    const members = S.models.map((m, i) => ({ label: m.label, pct: q.vals[i], agree: q.vals[i] === null || Math.sign(q.vals[i]) === Math.sign(q.mean) }));
    s += `<div class="ev-head" style="margin-top:10px"><div class="ev-lbl">${n} models</div><div class="count models${agree / n >= 0.8 ? (dry ? '' : ' wet') : ' mixed'}"><b data-count="models">${REDUCED ? agree : 0}</b><span>/${n}</span><small>lean ${dry ? 'drier' : 'wetter'}</small></div></div>`;
    s += modelTilesHTML(null, members, { season: S.label, where: 'at this 1° grid cell' });
    s += `<div class="tiles-foot"><span>Model average <b>${signed(q.mean)}%</b> vs ${REF()}</span></div>`;
    if (q.mean > 200) s += '<p class="note">Normal rainfall here is low in this season, so a modest change in millimetres shows up as a very large percentage.</p>';
  }
  s += `<p class="note">A raw readout of this year's models for one grid cell. ${near?.inside ? 'The region card adds the observed record and the literature.' : 'There is no observed track record behind it, and single cells are noisy. The shaded regions are where history and the literature back the forecast.'}</p>`;
  if (near) s += `<button class="btn near" type="button" data-go="${near.key}">${near.inside ? 'Open region card' : `Nearest mapped region: ${esc(near.title)}`}<span>→</span></button>`;
  return s + '<button class="card-close btn btn-icon" type="button" data-close aria-label="Close"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg></button>';
}

function globalHTML(meta, lit) {
  const A = meta.aggregate, pct = Math.round(100 * A.hits / A.n), p = meta.p2027_warmest;
  const lo = Math.round(100 * Math.min(p.ONI, p.RONI)), hi = Math.round(100 * Math.max(p.ONI, p.RONI));
  const L = lit?._global;
  let s = '<div class="c-kicker neutral">All regions</div><h2>The global picture</h2>';
  if (L?.headline) s += `<p class="c-headline">${esc(L.headline)}</p>`;
  s += `<div class="bignum"><div><b>${pct}%</b><span>of region-events went the expected way (${A.hits} of ${A.n}), across the ${meta.n_map_regions} map regions and ${meta.strong_events.length} strong El Niños. Chance would give about half.</span></div>`;
  s += `<div><b>${lo === hi ? lo : `${lo}–${hi}`}%</b><span>chance that 2027 sets a new global temperature record, from the Climate Brink's September 2026 forecast update (range across the ONI and relative-ONI versions).</span></div></div>`;
  s += `<section class="ev"><div class="ev-lbl">Share of regions that went the expected way, by event</div><div class="byev">${meta.strong_events.map((y) => {
    const [h, n] = meta.by_event[y]; return `<div data-year="${y}" title="${y}–${String(y + 1).slice(2)}: ${h} of ${n} regions · click to replay"><em>${Math.round(100 * h / n)}%</em><i style="height:${REDUCED ? 64 * h / n : 0}px" data-h="${64 * h / n}"></i><small>'${String(y).slice(2)}</small></div>`;
  }).join('')}</div></section>`;
  s += '<p class="spec" style="border:0;padding:0">Regions within one event are not independent, so these are not separate tests. The canonical teleconnection maps were also built partly from these same events, especially 1982–83 and 1997–98.</p>';
  s += litHTML(L);
  return s;
}

function howtoHTML(meta) {
  const ev = meta.strong_events.map((y) => `${y}–${String(y + 1).slice(2)} (${meta.oni_ndj_strong[y].toFixed(1)} °C)`).join(', ');
  return `<div class="c-kicker neutral">Method</div><h2>How to read this map</h2><div class="howto"><dl>
  <dt>Shaded regions</dt><dd>Places where the published literature expects a strong El Niño to shift rainfall or temperature, assessed with IPCC-style confidence. Darker shades are higher confidence. A dashed outline means the signal underperformed in a recent strong event.</dd>
  <dt>The ${meta.strong_events.length} strong El Niños</dt><dd>Events with a Nov–Jan Oceanic Niño Index of at least 1.5 °C: ${ev}.</dd>
  <dt>Bars, line and checks</dt><dd>${BARS === 'typical'
    ? 'Each bar is that event\'s rainfall in the region\'s season as a % of a typical ENSO-neutral year (the neutral-year trend plus the median neutral residual), so a check means the bar goes the expected way from 100%. The line marks the 1991–2020 average for comparison. By chance you would expect about half.'
    : 'Each bar is that event\'s rainfall in the region\'s season as a % of the 1991–2020 average. The line marks a typical ENSO-neutral year (the neutral-year trend plus the median neutral residual). A check means the event landed on the expected side of that line. By chance you would expect about half.'}</dd>
  <dt>La Niña check</dt><dd>How often La Niña years went the same way. A low count means the signal really is tied to El Niño.</dd>
  <dt>Why two reference points?</dt><dd>${BARS === 'typical' && BASE === 'normal'
    ? 'The past-event bars are measured against a typical neutral year, so a bar above or below 100% always matches its check mark. That reference removes long-term trends and is not pulled around by a few extreme years, and it is estimated from GPCC rain gauges since 1951. The forecast shading and model tiles stay relative to the 1991–2020 average. Each model forecasts a departure from its own long-run average, not from a typical neutral year. Re-expressing the forecasts that way would need a typical neutral year for every grid cell, which the satellite-era record (only 13 neutral years since 1979) pins down poorly: even a forecast of exactly normal rainfall would shade most of the map. So the shading shows what the models predict, and the bars show how past events compared with the same test the counts use.'
    : 'The counts compare each event with a typical neutral year; the percentages are relative to the 1991–2020 average. These usually agree, but in skewed climates a few very wet or dry years pull the average away from a typical year.'}</dd>
  <dt>Model tiles</dt><dd>One tile per forecast system, from the ${esc(meta.init)} start, averaged over the region's whole season up to ${esc(meta.model_horizon?.all ?? 'Feb 2027')}, the last month all ${meta.n_all ?? 13} systems (${meta.n_nmme ?? 6} NMME and ${(meta.n_all ?? 13) - (meta.n_nmme ?? 6)} Copernicus C3S) cover. Seasons that fall mostly after that use the ${meta.n_nmme ?? 6} NMME models, which run to ${esc(meta.model_horizon?.nmme ?? 'May 2027')}. Colour shows the model's regional-mean rainfall change. A red outline marks a model that goes the other way.</dd>
  <dt>Background shading and dots</dt><dd>${BASE === 'typical'
    ? 'The multi-model mean forecast rainfall for the selected season as a % of a typical neutral year for this event (GPCP neutral years since 1979, trend extended to 2026). Dots mark cells where at least 80% of models agree on the direction and the mean change is at least 10%.'
    : 'The multi-model mean rainfall change for the selected season, as % of the GPCP 1991–2020 normal. Dots mark cells where at least 80% of models agree on the sign and the mean change is at least 10%.'}</dd>
  <dt>Click anywhere</dt><dd>Outside the regions you get a raw model readout for one 1° grid cell. It has no observed track record behind it.</dd>
  <dt>Replay</dt><dd>Pick a past event to colour every region by whether it went the expected way that year.</dd>
  </dl></div>`;
}
