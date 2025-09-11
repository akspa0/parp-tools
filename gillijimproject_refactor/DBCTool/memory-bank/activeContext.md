# Active Context

- Current Focus:
  - Make MPQ reading reliable for DBCs (`Map`, `AreaTable`) using composite patch chains and robust StormLib IO.

- Recent Findings:
  - `SFileOpenFileEx` can return a handle ("open OK") but `SFileGetFileSize` returns `err=6`.
  - Unknown-size read fallback currently returns 0 bytes for `locale-enUS.MPQ: DBFilesClient/Map.dbc`.
  - Listing without a listfile fails (expected on many MPQs; `find-first err=0`).
  - Composite attach order improved; locale prefix passed to `SFileOpenPatchArchive`.

- Hypotheses:
  - Some paths in locale MPQs are patch fragments that require a composite (base + core patches + locale overlays) to materialize.
  - Verify StormLib.dll bitness (x64) and load path to exclude shadowing by a 32-bit DLL.
  - Path normalization/prefix nuances may matter for patched views.

- Next Steps (Morning Checklist):
  1. Confirm StormLib.dll x64 is used by the x64 process (ensure it sits next to `DBCTool.dll` or on PATH; no 32-bit shadow).
  2. `--mpq-test-open` probes:
     - `base-enUS.MPQ` → `DBFilesClient/Map.dbc`
     - `patch.MPQ` → `DBFilesClient/Map.dbc`
     - `locale-enUS.MPQ` → `DBFilesClient/Map.dbc`
     Try both `\` and `/` path separators.
  3. If single-file reads 0 bytes, try a smaller DBC (e.g., `LightParams.dbc`) to compare behavior.
  4. Re-run export with composite patching:
     - `dotnet run -- --dbd-dir ..\lib\WoWDBDefs\definitions\ --out out --locale enUS --table Map --mpq-root H:\WoWDev\modernwow --input 3.3.5.12340=mpq --mpq-locale-only enUS --mpq-verbose`
  5. If still failing, extract one DBC with Ladik’s to filesystem and confirm DBCD load via filesystem provider.

- Decisions:
  - Keep DBCD as upstream ProjectReference; diagnostics live in the tool; no changes to shared libraries.
