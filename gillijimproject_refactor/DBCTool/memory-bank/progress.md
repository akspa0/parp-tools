# Progress

- Works:
  - CLI scaffolding and argument parsing.
  - Filesystem-only DBC export via DBCD and a CSV writer.
  - Area comparison pipeline (`--compare-area`) with map crosswalk and robust name matching.
  - Alias/variant matching and map-biased → global fallback logic.
  - Deterministic remap workflow: discover, export to JSON (`--export-remap`), and apply from JSON (`--apply-remap`).
  - Filtering of development placeholders and exclusion of "DO NOT USE" targets by default.
  - Support for multiple source builds (`0.5.3`, `0.5.5`, `0.6.0`) targeting `3.3.5`.

- Pending:
  - No major features pending. Core functionality is complete.

- Next:
  - Improve documentation for external consumers.
  - Ensure the tool remains easy to integrate into larger workflows.

- Known Issues / Follow-ups:
  - None. The tool is stable for its intended filesystem-based workflow.
