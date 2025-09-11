# Progress

- Works:
  - CLI scaffolding and arg parsing
  - Provider selection (filesystem vs MPQ)
  - DBCD integration with filesystem provider and CSV writer
  - MPQ diagnostics: verbose logging, `--mpq-list`, `--mpq-test-open`, `--debug-mpq-file`
  - WDBC header validation and unknown-size read fallback path added

- Pending:
  - Confirm StormLib read success for DBCs via composite patching
  - Validate correct StormLib.dll (x64) is loaded; avoid 32-bit shadowing
  - Finalize patch attach order and locale prefix handling in all paths
  - Successful exports for `Map` and `AreaTable` from MPQs

- Known Issues / Follow-ups:
  - `SFileGetFileSize` often returns `err=6`; unknown-size read returned 0 for `locale-enUS.MPQ: DBFilesClient/Map.dbc`
  - Listing without a listfile may fail (expected); rely on direct open and composite view
  - Consider optional external listfile support if enumeration becomes necessary
