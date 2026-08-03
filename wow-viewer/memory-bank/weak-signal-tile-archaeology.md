# Weak-Signal Tile Archaeology — capability note

Recorded: 2026-08-03. Status: tooling built, validated, and parked. Not a training dependency.

## What this is

A pipeline that finds and visualizes terrain the corpus throws away: tiles whose relief is
compressed to near-nothing, or absent entirely. Built on 0.5.3, then run against 4.0.0.11927
(Azeroth, Kalimdor, Deephome, Gilneas2, LostIsles). Everything below is measured, not estimated.

## The tools (all `data-harvester`, all era-agnostic)

| script | what it does |
|---|---|
| `v50_tile_inventory.py` | per-tile record: classification, height range, `surviving_height_levels`, MCNR tilt, 4 neighbour keys + their real range, suggested amplification factor. CSV + JSON. Filters nothing. |
| `v50_synthesize_weak_tiles.py` | 4-panel sheet per degenerate tile (autostretched height, raking hillshade, XY-amplified normals, minimap) + a grid mosaic |
| `v50_tile_composite.py` | whole-map renders in 4 modes from ONE read pass: `absolute` / `autostretch` / `restored` / `liquid` |
| `v50_tile_version_diff.py` | pairs two builds' inventories by tile coord; transition matrix + side-by-side pair renders |

Run order: inventory -> synthesize -> composite. Version diff needs two inventories.

**Always pass `--near-zero-band inf` on any non-alpha client.** See the gates section.

## The measurement that matters: `surviving_height_levels`

Count of distinct height values in a tile. Amplitude alone cannot tell a squeezed landscape from a
squeezed nothing — two 0.5.3 tiles both measure ~0.512 range, one holds 2 distinct values and one
holds 27,132. Only the level count separates them.

Bands: `<=1` bit_exact_flat, `<=8` trace, `<=64` coarse_terrain, `65+` rich_terrain.

## Findings

- **0.5.3**: 1756 tiles, 361 degenerate. 205 carry non-zero relief; only 156 are bit-exact flat.
  33 are rich_terrain — shape intact, amplitude crushed. Sub-millimetre relief (5.19e-4 at world
  Z -501) is real: it **continues across tile boundaries**, which per-tile noise cannot do.
- **4.0.0 Azeroth**: 174 degenerate tiles, **zero** bit-exact flat. The weak-signal tiles are almost
  all ocean — seafloor is low-relief by design. The `liquid` vs `absolute` pair is what shows this:
  flat because water hides it, not flat because nothing is there.
- **The alpha squeezed tiles are NOT previews of the shipped game.** Tested on 62 Azeroth tiles that
  were trace/coarse in 0.5.3 and gained detail by 4.0.0: same-tile |r| median 0.119 vs a
  mismatched-pair control of 0.083 — indistinguishable. Power control (rich in both builds) fires at
  **|r| = 0.959** vs 0.248 shuffled, so the test works and the null is real.
  Conclusion: alpha terrain that existed survived nearly intact into Cataclysm; the degenerate tiles
  were built from scratch later. The squeezed data is genuine authored content of something that was
  **abandoned** — lost work, not an early draft.

## Three gates that each silently excluded the data we were hunting

Same failure mode three times. Worth checking for a fourth before trusting any null result here.

1. `RANGE_FLOOR = 1.0` (`height_relative_model.py`) — 149 tiles with real relief below it get a
   training target occupying a median 9.2e-05 of [0,1].
2. `nearZeroBand = 50.0` (`WeakSignalDetector.cs`, ported) — assumes alpha terrain near sea level.
   4.0.0 Kalimdor sits at |Z| p50 = 441, so **all 71** of its compressed tiles were rejected and the
   first run reported zero. Now `is_compressed_range` (band-free) is recorded alongside
   `is_weak_signal` so this can never hide again.
3. `usable = Height257 != null && MinimapRgb256 != null` (`Program.cs:3378`) — `discover-maps`
   excludes maps for lacking a **minimap**, not terrain. Drops Uldum (25 tiles), MaelstromZone (25),
   DeepholmeDungeon (16). This analysis needs only height + normals. The gate is advisory; `build`
   takes `--map` directly, untested whether harvest-stream enforces it.

## Known limitation if this is ever resumed

Post-alpha clients decode height as `float32(rawDelta + baseHeight)`; alpha stores absolute with no
addition. That sum destroys relief below the float32 ULP at altitude (6.1e-05 at |Z|=515, measured
on 4.0.0 Kalimdor). **`surviving_height_levels` therefore undercounts on every non-alpha client** —
false negatives possible, false positives not. To measure a later client faithfully, count levels on
the raw MCVT deltas before the base is added. NOT IMPLEMENTED.

## Why a full-corpus campaign was dropped

Measured 1.4 MB/tile. One late client = 69 harvestable maps, 6,178 tiles, ~8.4 GB. 28 clients
~237 GB against 148.5 GB free on I:. Does not fit, and the ULP issue above would bias almost every
client in it. Revisit only with the delta-based level counting done first.

## Related

- Spec 127 (`127-weak-tile-explorer`) — viewer explorer + wiring
  `WeakSignalDetector.EstimateFactorFromRanges`, which exists with **zero callers**; the viewer's
  auto factor uses a WDL guide and its `epsilon = 0.001` refuses exactly these tiles. Spec drafted,
  unimplemented.
- The one training-relevant output is in `progress.md`: gate curation on `surviving_height_levels`.
