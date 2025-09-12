# Progress

- Works:
  - CLI scaffolding and arg parsing
  - Provider selection (filesystem vs MPQ)
  - DBCD integration with filesystem provider and CSV writer
  - MPQ diagnostics: verbose logging, `--mpq-list`, `--mpq-test-open`, `--debug-mpq-file`
  - WDBC header validation and unknown-size read fallback path added
  - Filesystem DBC export via DBCD; CSV writer
  - Compare Areas pipeline with map crosswalk and name matching
  - Alias/variant matching and map-biased → global fallback
  - Remap JSON export/apply for deterministic runs
  - Dev placeholder filtering; default exclusion of 3.3.5 "DO NOT USE" targets

- Pending:
  - Confirm StormLib read success for DBCs via composite patching
  - Validate correct StormLib.dll (x64) is loaded; avoid 32-bit shadowing
  - Finalize patch attach order and locale prefix handling in all paths
  - Successful exports for `Map` and `AreaTable` from MPQs
  - Generalized source selection (0.5.3 or 0.5.5) with `--src-alias`/`--src-build`

- Current results (0.5.3 → 3.3.5):
  - Summary: name=477, unmatched=1, ambiguous=0, skipped_dev=10
  - Remap exported: `defs/053_to_335.remap.json`

- Next:
  - Fix `MapId_to_Name_{0.5.3|0.5.5}.csv` generation; ensure Directory → InternalName → MapName_lang fallback is correct and consistent
  - Re-run compare for 0.5.5; target unmatched ≤ 2
  - Integrate `Area_patch_crosswalk_{053|055}_to_335.csv` into ADT conversion pipeline (join on `src_areaNumber` → write `tgt_areaID`)
  - Optional: proximity-enriched suggestions using 3.3.5 WorldMapArea centroids

- Known Issues / Follow-ups:
  - `SFileGetFileSize` often returns `err=6`; unknown-size read returned 0 for `locale-enUS.MPQ: DBFilesClient/Map.dbc`
  - Listing without a listfile may fail (expected); rely on direct open and composite view
  - Consider optional external listfile support if enumeration becomes necessary
  - Some MapIDs produce empty or wrong map names in Map.dbc reports; fix in progress (affects parent map labels downstream)
