# Parp Tools — Gillijim Project (C# refactor)

This repository contains tools and libraries to restore and analyze AreaTable mappings across early World of Warcraft builds, with a preservation-first philosophy. The current focus is strict, numeric, auditable mapping from Alpha-era data (0.5.x) to LK 3.3.5 — without editing DBC files.

Key docs:

- AreaID Restoration Approach: `DBCTool.V2/docs/areaid-restoration-approach.md`
- Input Data Preparation (test_data layout and Alpha WDT extraction): `DBCTool.V2/docs/input-data-prep.md`

---

## Repository structure (what each folder is for)

- `AlphaWDTAnalysisTool/`
  - `AlphaWdtAnalyzer.Cli/` — CLI to remap LK ADTs using numeric crosswalks and Alpha-captured area values; writes verify CSVs and (optionally) visualizations.
  - `AlphaWdtAnalyzer.Core/` — core exporter/patcher, CSV schemas, visualization helpers.
  - Uses per-map numeric crosswalks from `DBCTool.V2` and enforces strict, map-locked, non-heuristic behavior.

- `DBCTool.V2/`
  - CLI that generates crosswalks (`Area_patch_crosswalk_via060_map*_*.csv`), dumps, and audits under `DBCTool.V2/dbctool_outputs/session_*/compare/v2/`.
  - Rules are strict and map-locked; only via060 rows with non-zero targets are used for patching.
  - See `DBCTool.V2/README.md` and the docs linked above.

- `DBCTool.V2.Core/`
  - Shared types and helpers used by `DBCTool.V2`.

- `DBCTool/` (legacy — unmaintained; use `DBCTool.V2`)
  - Earlier implementation with issues we could not resolve; kept for historical reference. Please use `DBCTool.V2`.

- `lib/`
  - Vendored reference libraries and tooling used by this project and experiments:
    - `wow.tools.local/` (e.g., `WoWFormatLib`, `DBCD`, `TACTSharp`) — CASC/TACT access, format parsers, and DBC helpers.
    - `Warcraft.NET/` — WoW format domain models/utilities.
    - `WoWDBDefs/` — DBDefs parsing and utilities.
    - `MapUpconverter/` — reference materials for map/terrain conversions.
  - Not all subprojects are required at runtime; treat as a toolbox/reference.

- `test_data/`
  - Standard input tree: `test_data/<version>/tree/...`. Examples:
    - `test_data/0.5.3/tree/World/Maps/Azeroth/Azeroth.wdt`
    - `test_data/3.3.5/tree/DBFilesClient/AreaTable.dbc`
  - See `DBCTool.V2/docs/input-data-prep.md` for full layout and Alpha WDT extraction notes (MPQEditor from Zezula.net).

- `tools/`
  - Utility scripts like `tools/agg-area-verify.ps1` to summarize verify CSVs (per-map `alpha_areas_used_*.csv` and `unknown_area_numbers_*.csv`).

- `reference_data/`
  - Local copies of WoWDev wiki pages used during development. Treat as reference only; can be out of date and is not authoritative.

- `053-chain/`, `053-viz/`, `053-viz-36/`
  - Example output trees generated during development from prior commands; not required by tools (which choose sensible output locations by default).

- `next/`, `refactor/`, `src/`
  - Experimental or in-progress tools and refactors outside the current preservation workflow.

- `out/`, `docs/`, `memory-bank/`
  - `out/` — ad hoc outputs (varies by experiment).
  - `docs/` — planning or top-level documentation.
  - `memory-bank/` — working context notes.

---

## Quick workflows (high level)

1) Generate crosswalks and audits (DBCTool.V2)

```bash
dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s53
```

- Inputs come from `test_data/<version>/tree` (see docs). Outputs land under `DBCTool.V2/dbctool_outputs/session_*/compare/v2/`.

2) Remap ADTs and verify (AlphaWDTAnalysisTool)

- Use a patch directory that contains the via060 per-map crosswalks (and optional overrides), and provide the LK 3.3.5 DBC directory for names in verify.
- After export, run `tools/agg-area-verify.ps1` on the export root to see which `alpha_raw` values were present and whether they mapped.

---

## Policies

- Strict numeric mapping only; map-locked; no heuristics.
- Overrides contain exact numeric rows and never override explicit crosswalk mappings.
- We do not edit DBCs in this preservation track.

---

## Housekeeping

- Legacy: `DBCTool/` is unmaintained and kept for historical reference. Please use `DBCTool.V2` for all current workflows.
- License: undecided — pending review of third-party library licenses in `lib/`.

