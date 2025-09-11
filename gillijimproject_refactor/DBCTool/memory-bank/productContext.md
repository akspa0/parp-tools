# Product Context

- Why: Provide a reliable, scriptable way to export WoW DBC tables to CSV across builds for analysis, porting, and downstream tooling.
- Users:
  - Internal devs maintaining map/tools pipelines
  - Data analysts comparing DBC content across client builds
- UX Goals:
  - Simple CLI with clear flags (`--table`, `--input`, `--mpq-root`, `--mpq-archive`, etc.)
  - Helpful logging and diagnostics (`--mpq-verbose`, debug probes) when something fails
  - Deterministic output layout per build in `out/dbcTool_<build>_<timestamp>/`
- Constraints / Principles:
  - Use upstream DBCD from `lib/wow.tools.local/DBCD` via ProjectReference (no vendoring)
  - Default WoWDBDefs path: `lib/WoWDBDefs/definitions`
  - Default locale: `enUS`
  - Keep changes self-contained to the tool; avoid modifying shared core libraries
