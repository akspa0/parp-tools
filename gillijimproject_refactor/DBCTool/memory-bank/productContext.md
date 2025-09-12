# Product Context

- Why: Provide a reliable, scriptable way to export WoW DBC tables to CSV across different builds for analysis and to generate deterministic AreaID remap definitions for downstream tools.
- Users:
  - Internal devs maintaining map/tools pipelines.
  - Data analysts comparing DBC content across client builds.
- UX Goals:
  - Simple CLI with clear flags (`--table`, `--input`, `--compare-area`, etc.).
  - Helpful logging and clear output reports.
  - Deterministic output layout per build in `out/<build_alias>/` (e.g., `out/0.5.3/`).
- Constraints / Principles:
  - Use upstream DBCD from `lib/wow.tools.local/DBCD` via ProjectReference (no vendoring).
  - Default WoWDBDefs path: `lib/WoWDBDefs/definitions`.
  - Default locale: `enUS`.
  - Keep changes self-contained to the tool; avoid modifying shared core libraries.
