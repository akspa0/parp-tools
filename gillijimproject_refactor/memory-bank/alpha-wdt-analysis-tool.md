# AlphaWDTAnalysisTool — Viz and Data Notes

## Purpose
Capture the visualization/static index pipeline and data files under `053-viz/` and current gaps.

## Current State
- `053-viz/index.json` lists tiles from `.\\test_data\\0.5.3\\tree\\World\\Maps\\Azeroth\\...`.
- `053-viz/viz/maps/Azeroth/index.html` is a pre-generated static SVG page; no dynamic CSV/JSON loading.
- Legend shows only AreaID 0, suggesting the generator’s source had no meaningful AreaIDs (likely Alpha ADTs lacking area metadata) or parsing defaulted to 0.
- `053-viz/csv/` contains:
  - `unique_ids_all.csv`, `placements.csv` with `type,asset_path,map,tile_x,tile_y,unique_id`.
  - `assets_*.csv` enumerations; `id_range_summary*.csv` and `id_ranges*.csv` for summaries.

## Observations
- The static `index.html` is not wired to DBCTool outputs; any analysis relying on ADT-embedded area fields will collapse to 0 in Alpha contexts.

## Next Steps
- Rewire visualization to consume `DBCTool.V2` outputs (`mapping.csv`/`trace.csv`) and color by `tgt_id_335` or `path`.
- Add generator diagnostics: top-N AreaID histogram, zero/NaN counters during parsing.
- Verify instance maps (e.g., `DeadminesInstance`) via targeted slice renders using DBCTool CSVs.
