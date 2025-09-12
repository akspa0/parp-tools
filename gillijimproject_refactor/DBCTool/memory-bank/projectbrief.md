# Project Brief

- Name: DBCTool — DBC → CSV exporter
- Status: CLI runs; DBCD wired via ProjectReference; MPQ pipeline under active debugging
- Current Goal: Reliable export of selected DBC tables from either filesystem trees or MPQ archives, for multiple builds
- Scope:
  - Builds: Classic/WotLK friendly (initially 3.3.5.12340)
  - Inputs: DBFilesClient directory OR MPQ root/archives
  - Outputs: CSV per table in `out/dbcTool_<build>_<timestamp>/`
- Philosophy:
  - Use upstream DBCD (wow.tools) via ProjectReference (no vendoring)
  - Small, observable CLI; robust diagnostics for MPQ reads
  - Keep logic self-contained in tool; do not mutate core libraries
- Deliverables:
  - net9.0 console app + README with flags
  - Verified exports for key tables (e.g., Map, AreaTable) across target builds
