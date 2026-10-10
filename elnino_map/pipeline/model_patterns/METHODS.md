# Dynamical-model check of the El Niño 2026–27 impacts map: methods

Purpose: test the literature-assessed regions on `figures/elnino_2026_impacts_map` against the September 2026 dynamical seasonal forecasts. Where the models support a region, reshape its polygon to where they agree. Separately list model-robust signals that are missing from the map. The fill colour on the map remains the **literature** confidence tier. Model output decides only a polygon's shape, plus any region edits the author approves (see `derived/region_table.csv`).

Pipeline: `fetch_cds.py` (C3S and ERA5 via the CDS) → NMME files via curl (see Data) → `process.py` → `regions.py` → `reshape.py`; `diag_maps.py` makes the figures. Derived files in `derived/` (72 MB, zlib-compressed) regenerate every figure. Raw inputs other than ERA5 (~90 MB) are kept. The ERA5 file (57 MB) was deleted after processing; `fetch_cds.py` re-downloads it if `process.py` needs to rerun. Literature polygons as of map v4 are frozen in `lit_polygons_v4.json`, so the tests don't depend on the current map script.

## Data (all September 2026 initializations)

| Source | Models | Variables | Leads used | Anomaly baseline |
|---|---|---|---|---|
| NMME, CPC `ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/2026090800/` | CFSv2, CanESM5, GEM5.2-NEMO, NASA-GEOS5v2, NCAR-CCSM4, NCAR-CESM1 (each model's ensemble mean) | `prate`, `tmp2m` | Sep 2026 – May 2027 | Each model's hindcast climatology (see below) |
| NMME calibrated tercile probabilities, `NMME/prob/netcdf/prate.202609.prob.adj.seas.nc` | NMME multi-model | precipitation | 7 overlapping 3-month seasons | CPC standard |
| C3S, `seasonal-postprocessed-single-levels`, product `ensemble_mean` | ECMWF s51, UKMO s610, Météo-France s9, DWD s22, CMCC s4, JMA s4, BOM s2 | `2m_temperature_anomaly`, `total_precipitation_anomalous_rate_of_accumulation` | leadtime months 1–6 (Sep 2026 – Feb 2027) | 1993–2016 |
| C3S NCEP s2 and ECCC s5 | fetched only to check consistency with NMME | same | same | 1993–2016 |
| GPCP v2.3 monthly (NOAA PSL) | observations | precipitation | 1991–2020 | used for the interannual SD and climatology |
| ERA5 monthly means, 1° (CDS) | reanalysis | 2 m temperature | 1979–2025 | used for the trend and SD |

**Deduplication.** C3S NCEP is CFSv2 and C3S ECCC is the CanSIPS pair (CanESM5 + GEM5.2-NEMO), so both come from NMME. That leaves **13 models from 13 centres** for SON, OND and DJF, and **6 NMME models** for JFM and MAM. The models are still not fully independent (NCAR CESM1/CCSM4 share lineage, and several C3S systems share NEMO or IFS components). The effective sample is therefore smaller than 13 and the sign-test p-values below are optimistic.

**Lagged ensembles.** In the C3S files, BOM, JMA and NCEP carry no `forecast_reference_time`. They are lagged ensembles whose nominal start is September 2026, as requested. Check [2] below confirms the NCEP lead alignment.

## Temperature baselines and trend adjustment

The NMME realtime anomalies are **not** all on one baseline. The periods below come from the `long_name` in CPC's `/NMME/clim/` files. For GEM5.2-NEMO and CFSv2, I checked that realtime anomaly = forecast − that climatology to within 1e-4 K:
- CFSv2, NASA-GEOS5v2, NCAR-CCSM4: 1992–2019
- GEM5.2-NEMO: 1982–2010
- CanESM5, NCAR-CESM1: **no climatology file on CPC; baseline unknown.** They are excluded from the temperature agreement but kept for precipitation.

Seasonal-forecast temperature anomalies relative to hindcast climatologies centred on about 1996–2006 are dominated by warming since then. Before adjustment, 90–95% of land between 60°S and 60°N had ≥80% of models warm. I used three treatments:
1. **Main:** subtract the ERA5 1979–2025 linear trend for that cell and season (Gaussian-smoothed, σ = 1.5 cells), multiplied by (target time − baseline midpoint). The land-mean offset removed is 0.6–0.75 °C for 1992–2019 and 1993–2016 models, and 0.9–1.0 °C for GEM5.2.
2. As (1), but with the 1991–2025 trend.
3. Subtract each model's own 60°S–60°N land-mean anomaly. This does not depend on the baseline, but it also removes El Niño's own contribution to the global mean.

A cell counts as robust for temperature only if all three agree. Robustly warm land is 44–61% under (1) and (2), 17% under (3), and 16–17% when all three must agree. The linear trend understates recent warmth: ERA5 land residuals were +0.26 °C in 2023 and +0.44 °C in 2024. That is why the trend-only version overstates the El Niño part.

**Baseline adjustment for precipitation:** none. The 1993–2016 vs 1991–2020 difference is second order for precipitation.

## Metrics

- Seasonal means: SON = months 1–3 (Sep–Nov), OND = 2–4, DJF = 4–6, JFM = 5–7 (NMME only), MAM = 7–9 (NMME only).
- C3S data are bilinearly regridded from the half-degree-offset 1° grid to the NMME integer 1° grid, periodic in longitude.
- **Standardized anomaly** z = anomaly / observed detrended 1991–2020 interannual SD (GPCP for precipitation, floor 0.05 mm/day; ERA5 for temperature, floor 0.1 °C). Grey masks on maps mark cells with seasonal climatology below 0.5 mm/day.
- **Robust cell:** ≥80% of models agree on the sign (≥5/6 for NMME-only seasons) and |multi-model-mean z| ≥ 0.5.
- **NMME calibrated tercile probabilities** are CPC's skill-adjusted product. The file is stored as fractions despite its `units="percent"` attribute (above + normal + below = 1.0). Target 800 = SON, 801 = OND, 803 = DJF, 804 = JFM, 806 = MAM (see check [3]).

## Region tests (`regions.py`)

Regions, expected signs and seasons were fixed in the script before any region results were computed:
- **Existing regions:** the literature polygons, buffered as drawn on the map.
- **Candidate regions:** boxes from the ENSO reviewer's list.

For each test:
- area-weighted regional-mean z per model; land cells only, except for island and ocean regions
- number of models with the expected sign
- one-sided binomial sign test
- **Benjamini–Hochberg FDR across all 51 tests** (q ≤ 0.10)
- NMME calibrated probability of the expected tercile
- fraction of the region's cells that are robust

Verdict rules:

| Verdict | Criteria |
|---|---|
| SUPPORTED (robust) | q ≤ 0.10, \|z\| ≥ 0.5, robust fraction ≥ 0.25 |
| supported (moderate) | q ≤ 0.10, \|z\| ≥ 0.25 |
| contradicted | majority of models have the opposite sign, \|z\| ≥ 0.25 |
| weak / no signal | otherwise |

## Polygon proposals (`reshape.py`)

Robust cells in the literature direction, across any of the seasons the label covers, within a search window: the literature polygon buffered by 2–5°, or a candidate box. Land regions are clipped to land dilated by 1 cell; island regions (Maritime Continent, Philippines) to land dilated by 2 cells. The mask is then smoothed (Gaussian σ = 0.8 cells), contoured at 0.45, and fragments under 10 deg² are dropped. Last comes a morphological close (±0.8°) and a 0.3° simplify. Contours only trim or extend a region inside its window; they never create a region on their own.

**v5.1 additions.**
- Inland Northwest dry: robust DJF dry cells in window 134°W–98°W, 40°N–64°N.
- Central Canada warm: DJF cells robust-warm under all three trend treatments in window 112°W–55°W, 42°N–66°N. The dry shape is removed from it, and it is trimmed east of 79.5°W.
- SE Australia: robust SON dry cells in window 139–153°E, 45–33°S.
- Sri Lanka: robust OND wet cells, land dilated 2 cells.

The last two windows were chosen after inspecting the maps.

**Europe (`europe.py`).**
- Tests fixed a priori: 6 regions × (Oct–Dec, Jan–Feb) × (precipitation, temperature) with literature-expected signs, plus exploratory DJF.
- Two-sided sign test, with BH FDR across the 36 tests.
- A Jan–Feb season was added for this (NMME months 5–6, C3S forecastMonth 5–6; 13 models). The existing region table was unchanged after the rerun.

## Validation checks (`derived/validation.txt`)

1. **Reproducing CPC:** my equal-weight mean of the 6 NMME model ensemble means reproduces CPC's `NMME.ENSMEAN` anomaly file. Pattern r = 1.0000 on all 9 leads; max difference 3e-6 mm/day for precipitation and 0 for T2m. This confirms that CPC's NMME mean weights models equally and that my lead indexing is right.
2. **Same model via C3S and NMME:** SON/DJF pattern r = 0.97/0.99 for precipitation and 0.95/0.97 for T2m (NCEP vs CFSv2); 0.97–0.98 for precipitation and 0.92–0.95 for T2m (ECCC vs CanESM5+GEM5.2). This confirms C3S lead alignment, including for the lagged NCEP ensemble.
3. **Tercile target convention:** target 800's (P_above − P_below) correlates best with lead months 0–2 (r = 0.75), so target = first month of the season.

## California hindcast / teleconnection test

Summary only; full details are in `hindcast/METHODS.md`, with figure `hindcast/fig_ca_precip_vs_nino34_nmme.png`.
- **Observations:** NOAA nClimDiv CA division 6 (SoCal) and division 2 (NorCal), 1950–2025, regressed on CPC NDJ ONI and RONI.
- **Models:** NMME September-start hindcasts via the IRI Data Library, 11 systems, each model regressed on its own predicted Niño3.4.
- **Validation:** ONI NDJ 1997 = 2.37 and NDJ 2015 = 2.59, matching the published 2.4 and 2.6. The 1991–2020 normals match NCEI's file to 0.01 in.
- **SoCal JFM:** observed slope +24 pp per °C (95% CI 11–37), r = 0.39. Model/observed slope ratio has a median of 0.97.
- **2015–16:** 9 of 10 models forecast wet; observed was 60–68% of normal.

## Caveats

- **Amplitude:** a +3.9 °C ONI event (≈ +3.3 relative ONI) is outside every hindcast set. Model teleconnections are being extrapolated in amplitude and possibly in flavour.
- **Circularity:** dynamical models build in the canonical teleconnections, so agreement with the literature is partly expected. Model support constrains **where** the signal sits this year more than **how confident** to be. Hindcast skill for strong events speaks to confidence (see `hindcast/`).
- **One snapshot:** everything is from the September 2026 initialization and will change with later ones.
- **Resolution:** the 1° grid does not resolve the Peru/Ecuador coastal strip or coastal ranges; the Southern California signal spans only about 2–3 cells.
- **Out of range:** summer-2027 impacts (Yangtze, N/Central India monsoon, the Central America midsummer drought) and 2027 fire seasons are beyond all leads and stay literature-only.
- **Latent issue for the Climate Dashboard:** its NMME Niño3.4 stream keeps NCAR-CESM1, NCAR-CCSM4 and NASA-GEOS. CCSM4 and GEOS use a 1992–2019 climatology, whose midpoint is 0.5 yr from 1991–2020, so the effect is negligible. CESM1's climatology period is undocumented on CPC, so its offset relative to 1991–2020 cannot be quantified from the available files. Flagged, not fixed.

## Full-window model checks (27 Sep 2026)

The per-region model checks shown in the post, the hit grid, the figures and the dashboard now average each model over the region's whole observed window, instead of the closest 3-month season. Source: `windows.py` (the rule and the monthly data) and `window_tests.py` (the tests), with output in `derived/window_tests.csv`.

- **Rule.** The model window is the observed window (the `hit_rates` season code) cut to the months all 13 systems cover. C3S runs 6 months from the start month, so a September start ends in Feb and an October start in Mar. If that covers less than half the window, the window is cut to the NMME horizon instead (9 months, NMME only). If that also covers less than half, there is no model check. The start month and both horizons are read from the raw folders.
- **Effect of the half-coverage rule.** It keeps a season that is mostly past, or mostly beyond the horizon, from being judged on one or two months. The 2026 Sahel monsoon (Jul–Sep) overlaps the forecast only in September, so it stays "beyond range".
- **Data.** Monthly model anomalies are in `derived/monthly_anoms.nc`, built by the same readers as `process.py` (lead 0 = start month; C3S `forecastMonth` 1 = start month).
- **Normals.** Window normals and detrended SDs come from GPCP monthly 1991–2020. Windows are labelled by the year of their last month, as `season_series` does for DJF.
- **Methods are unchanged; only the window is new.** Pre-registered literature regions and candidate boxes keep the standardized per-cell method (`regions.py`). Final map polygons keep the raw regional-mean method (`hit_rates/model_counts.py`). Temperature rows keep their 3-month seasons, because `t2m_adj` is trend-adjusted per season.
- **Reproduction check.** With each window forced to the region's old 3-month season, `window_tests.py` reproduces all 29 polygon counts in `model_counts.csv` and all 46 precipitation box tests in `region_table.csv`, including the cell counts and the clim/SD fields (to 1e-4). It asserts this on every run. The old tables are kept; the released video was built from them.
- **Result.** Every map-region count is unchanged. Indonesia Sep–Dec, W and Central Pacific Sep–Feb, Amazon Oct–Feb, SE South America Sep–Feb and Hawaii Nov–Feb are all still 13/13. Central Chile moves to Oct–Nov (13/13) and southern China stays at 12/13 on its box. Off the map, S-central Chile moves from 2/6 NMME (Jan–Mar) to 8/13 (Jan–Feb). The model averages shown on the dashboard change for the regions whose window changed. For example, Indonesia is −32% for Sep–Dec, against −46% for Sep–Nov, because the dry signal fades toward December.

## October 2026 refresh (10 Oct 2026)

The pipeline now lives in the Climate Dashboard repo (`elnino_map/pipeline/`) and is parametrized by
initialization (`model_patterns/initcfg.py`, `ELNINO_INIT=YYYYMM`). Before the first October run it reproduced the
September bundle exactly (seasonal fields and metrics to 0, grid files and `regions.json` byte-identical).

- **Data.** NMME 1 Oct 2026 initializations (CPC folder `2026100800`), all 9 C3S systems' October runs
  (ECMWF posted 7 Oct, the rest 10 Oct), NMME calibrated tercile probabilities `prate.202610.prob.adj.seas.nc`.
  Observations, polygons, confidence tiers and the hit record are unchanged.
- **Seasons.** An October start has no Sep–Nov; the C3S horizon moves to March 2027, so Jan–Mar becomes a
  13-system season and Mar–May stays NMME-only (6). The map shows OND, DJF, JFM and MAM.
- **Tercile targets.** The October probability file's target codes run 801–807 (the September file's ran
  800–806). Check [3] shows target 801 correlates best with lead months 0–2 (r = 0.75 vs 0.69, 0.61), so the file's
  first code is the season starting at the start month; `process.py` now maps targets from that rather than
  assuming 800.
- **Validation.** [1] NMME mean reproduces CPC's file (r = 1.0000, max |diff| 4e-6 mm/day); [2] C3S NCEP vs
  CFSv2 precip r = 0.98 (OND) / 0.99 (DJF), ECCC vs CanSIPS 0.98 / 0.97.
- **Region checks (full windows, 13 systems unless noted).** 20 of 24 checked regions keep every model on
  the literature side. Moves: Horn of Africa 13→12/13 (one system dry, mean still +34%), southern Africa
  13→12/13, Inland Northwest 13→12/13, SE Australia 12→11/13 with the mean dry signal shrinking to −6%,
  Sri Lanka 12→13/13, southern China 11→12/13, Mekong (NMME) 4→5/6. Largest amplitude changes: central Chile
  +90% → +116%, central Pacific +179% → +168% (now Oct–Mar), Hawaii −66% → −71%, N. South America −45% → −40%.
- **Not changed.** `data/forecast_2027_extract.json` (P(2027 warmest)) remains the September extract from the
  global temperature forecast workflow. Region polygons are the published map's (author-approved shapes are not
  re-derived from the October fields).
