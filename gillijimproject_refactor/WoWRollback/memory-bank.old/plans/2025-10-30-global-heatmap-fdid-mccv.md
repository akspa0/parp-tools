# 2025-10-30 Plan — Global Heatmap, FDID Pipeline, MCCV Overlay, Empty Tiles, Recompile Modes

## Scope
Deliver a richer analysis and visualization flow in GUI and CLI:
- Global heatmap (per‑build; epoch option next)
- Layers scope control (Tile | Selection | Map)
- FDID resolution + CSV enrichment + diagnostics
- MCCV analyzer + per‑tile PNG overlay; detect hidden-by-holes
- Empty‑tile visualization (gray gridlines) powered by tile_presence.csv
- Robust CASC → recompile routing (prefer LK WDT; manual picker; future synthetic Alpha WDT)

## Phases and Acceptance Criteria

### Phase 1: Global Heatmap (Build Scope)
- Implement StatsService to scan `<cacheBuild>/*/tile_layers.csv`.
- Persist `heatmap_stats.json` with: `minUnique`, `maxUnique`, `perMap` min/max, `generatedAt`.
- GUI: Heatmap Scope toggle (Local | Global). Global uses build stats for color scaling.
- AC:
  - Stats recomputed when any tile_layers.csv is newer than stats file.
  - Global view produces identical color scaling across different maps of same build.

### Phase 2: Layers Scope Control
- Add `Layer Scope` ComboBox: Tile | Selection | Map.
- Tile: original per‑tile layer list (always visible regardless of selection count).
- Selection: current band summary across selected tiles.
- Map: aggregate bands across entire map; checkbox interactions remain consistent.
- AC: Switching scope updates panel instantly without requiring reselecting tiles.

### Phase 3: FDID Resolver & CSV Enrichment
- FdidResolver service:
  - Inputs: community listfile (CSV), listfile JSON snapshots (from `snapshot-listfile`).
  - Normalizes paths; supports path↔FDID resolution.
- Integrate into placements/layers generation so CSVs include: `fdid`, `asset_path`, `source`.
- Diagnostics: emit `unresolved_paths.csv` with reason categories.
- AC:
  - Resolution rate logged; unresolved CSV created on misses.
  - No regressions in CSV schema (additive columns only).

### Phase 4: MCCV Analyzer & Overlay
- CLI `analyze-mccv --client-path <dir> (--map <m>|--all-maps) --out <dir>`.
- Outputs:
  - `<map>/mccv_presence.csv`: `tile_x,tile_y,chunk_idx,has_mccv,holes_set`.
  - Optional `<map>/mccv/<map>_<x>_<y>.png` from decoded MCCV BGRA (9*9 + 8*8) per MCNK stitched by tile.
- GUI overlay toggle: Show MCCV (like minimap loader).
- AC:
  - Tiles with MCCV but HOLES set are flagged "hidden by holes"; PNG still generated.
  - UI loads MCCV PNG when present.

### Phase 5: Tile Presence & Empty‑Tile Gridlines
- CLI emits `<map>/tile_presence.csv` with booleans: `has_minimap,has_placements,has_terrain,has_mccv`.
- GUI draws gray gridlines for tiles that exist but have no placement rows.
- AC: Visually distinct empty tiles without impacting heatmap fill.

### Phase 6: Recompile Modes (CASC/LK)
- GUI Recompile routing:
  - If dataset = CASC: prefer LK Client for `World/Maps/<map>/<map>.wdt`; else prompt for manual WDT.
  - (Planned) If no LK WDT exists: synthesize Alpha WDT from `tile_layers.csv` (new CLI `pack-alpha-wdt-from-tiles`).
- AC: Recompile path selectable and logged; manual WDT picker available; future synthetic path spec written.

## Risks & Mitigations
- FDID ambiguity / missing coverage → diagnostics CSV + path normalization + snapshot support.
- Performance on large builds → cache `heatmap_stats.json`; lazy recompute.
- CASC-only modern maps without LK WDT → synthetic Alpha WDT path (documented; implement after core phases).
- MCCV PNG size/time → low‑res first; allow opt‑in export flag.

## Test Plan
- Stats: two builds with different distributions; verify consistent global color scale per build.
- Layers scope: switch Tile/Selection/Map; verify lists and bands match expected counts.
- FDID: unit tests for resolver normalization; golden CSVs for enriched columns; unresolved report written.
- MCCV: sample map known to use MCCV + holes; presence CSV and PNGs match expectations.
- Tile Presence: empty tiles gridlines visible when no placements; disappears when placements exist.
- Recompile: CASC map with LK client set (auto WDT), unset (manual picker), and (future) synthetic path spec verified.

## Deliverables (this iteration)
- StatsService + `heatmap_stats.json` + UI toggle.
- Layer Scope control + restored per‑tile lists.
- FdidResolver + CSV enrichment + `unresolved_paths.csv`.
- MCCV analyzer + optional PNG export + overlay toggle.
- tile_presence.csv + empty‑tile gridlines in UI.
- Recompile routing improvements (LK client preferred + manual picker).

## Out of Scope (for next chat)
- Epoch‑level global scaling (planned after build‑level global).
- Synthetic Alpha WDT from tiles (spec only in this iteration).
- Timeline SQLite (first/last seen) across multiple builds.

## Next Chat Action Items
- Confirm Heatmap Scope UX and color legend for Global.
- Approve CSV schema additions (`fdid`, `asset_path`).
- Confirm MCCV PNG resolution and toggle placement.
- Provide LK Client path(s) for WDT lookup in CASC mode or confirm manual picker workflow.
