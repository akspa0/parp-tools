# Product Context — WoWRollback (Rewrite)

## Why this exists
- **Quick visual QA** across branches/versions using per‑tile overlays and a static viewer.
- **Deterministic artifacts** (JSON/CSV) that other tools or branches can consume without coupling to the current code tree.
- **Cherry‑picking ready**: stable paths/schemas so working parts can be lifted into older baselines.

## Problems it solves
- **Inconsistent filenames** (e.g., `placements.csv` vs `<map>_placements.csv`) blocked overlays. We standardize probe order and outputs.
- **Viewer path mismatches** (`data/overlays` vs `overlays`). We align overlay paths to what the viewer fetches.
- **Analyze‑only gaps**: Missing terrain/shadow shouldn’t block object overlays or the viewer index.

## How it should work (happy path)
1. Inputs discovered:
   - Converted LK ADTs dir; optional `analysis/index.json`.
   - Placements CSV `<map>_placements.csv` (preferred locations), or master index `<map>_master_index.json`.
2. Overlays generated:
   - Emit `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json` (layered schema; `M2` and `WMO` points; `world` and `pixel`).
3. Index built:
   - Scan overlays on disk to write root `05_viewer/index.json` with versions/maps/tiles.
4. Serve/view:
   - Viewer loads defaultVersion, first map, and tile overlays using the `combined` variant.

## User experience goals
- **Fast load**: Minimal index listing only available tiles.
- **Graceful degradation**: Missing terrain/shadow is non‑fatal; objects still render.
- **Stable contracts**: Overlay/CSV schema and paths do not change without an explicit plan.
