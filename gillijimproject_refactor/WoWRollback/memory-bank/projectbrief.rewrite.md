# Project Brief (Rewrite)

## Purpose
Reconstruct a reliable analysis + viewer pipeline for WoW rollback data that produces actionable overlays and CSV diagnostics to support visual inspection and cherry‑picking across branches.

## Core Outcomes
- Objects overlays per tile at `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json` with layered schema suitable for the web viewer.
- Root viewer `index.json` listing versions, defaultVersion, and per‑map tiles to enable the UI to load overlays/minimaps.
- UniqueID analysis CSVs and layers JSON driven from the richest available source (master index preferred, then placements CSV).

## Inputs
- Converted LK ADTs directory (supports analyze‑only).
- `analysis/index.json` (optional) containing placements.
- Generated placements CSVs named `<map>_placements.csv`.
- Master index JSON `<map>_master_index.json`.

## Outputs
- `04_analysis/<version>/objects/<map>_placements.csv`
- `04_analysis/<version>/master/<map>_master_index.json`
- `04_analysis/<version>/uniqueids/*`
- `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`
- `05_viewer/index.json`

## High‑Value Modules to Cherry‑Pick
- Analysis orchestrator stage wiring and CSV probing rules for placements.
- Overlay generation (index‑based and CSV‑based) with tile grouping and layered schema.
- Master‑index fallback for overlays when CSV is absent.
- Minimal viewer index builder based on overlays present on disk.
- World→pixel conversion aligned to viewer constants.
