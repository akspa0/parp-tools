# Alpha WDT → UniqueID CSVs → Crosswalks → LK ADTs (+WDT) → Viewer

This guide shows the complete workflow using WoWRollback.Cli and built-in modules.

## 0) Inputs
- Alpha loose files (recommended):
  - World/Maps/<Map>/<Map>.wdt
  - DBFilesClient/AreaTable.dbc, Map.dbc
- LK 3.3.5 DBCs: DBFilesClient/AreaTable.dbc, Map.dbc
- Optional: Client roots for MPQ-backed extraction

## 1) Build UniqueID CSVs from Alpha WDT
Extract UniqueID ranges, timeline, and asset ledger.

```powershell
cd WoWRollback

dotnet run --project WoWRollback.Cli -- analyze-alpha-wdt \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --out outputs\Azeroth-uniqueid
```

Outputs (under the session directory):
- alpha_Azeroth_ranges.csv (per-map ranges)
- alpha_Azeroth_timeline.csv (optional)
- alpha_Azeroth_asset_ledger.csv (optional)

## 2) Generate AreaID crosswalks
Use loose DBCs (preferred) or extract from clients. Crosswalks are used to map Alpha AreaIDs to LK AreaIDs.

```powershell
dotnet run --project WoWRollback.Cli -- alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --dbd-dir ..\lib\WoWDBDefs\definitions \
  --src-dbc-dir ..\test_data\0.5.3\tree\DBFilesClient \
  --lk-dbc-dir  ..\test_data\3.3.5\tree\DBFilesClient \
  --auto-crosswalks --copy-crosswalks --report-areaid \
  --strict-areaid
```

Notes:
- Pivoting via 0.6.0 is OFF by default; enable only with `--chain-via-060`.
- If you pass `--src-client-path` or `--lk-client-path`, Map/AreaTable DBCs may be auto-extracted.

## 3) Convert Alpha → LK ADTs and write LK WDT
Include burial and terrain passes. LK ADTs and a fresh LK WDT (<Map>.wdt) will be written to the LK output folder.

```powershell
dotnet run --project WoWRollback.Cli -- alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 50000 \
  --fix-holes --holes-scope self --holes-wmo-preserve true \
  --out outputs\Azeroth-session \
  --lk-out outputs\Azeroth-session\lk_adts\World\Maps\Azeroth \
  --dbd-dir ..\lib\WoWDBDefs\definitions \
  --src-dbc-dir ..\test_data\0.5.3\tree\DBFilesClient \
  --lk-dbc-dir  ..\test_data\3.3.5\tree\DBFilesClient \
  --auto-crosswalks --copy-crosswalks --report-areaid --strict-areaid
```

Outputs:
- LK ADTs under `lk_adts/World/Maps/<Map>/`
- `lk_adts/World/Maps/<Map>/<Map>.wdt` (fresh WDT synthesized from Alpha)
- `reports/areaid_patch_summary_<Map>.csv`

## 4) Analyze LK ADTs (UniqueID ranges, assets) and generate viewer
```powershell
dotnet run --project WoWRollback.Cli -- analyze-map-adts \
  --map Azeroth \
  --input-dir outputs\Azeroth-session\lk_adts\World\Maps\Azeroth \
  --out analysis_output
```

Outputs:
- `<map>_placements.csv`, `<map>_terrain.csv`, `<map>_uniqueID_analysis.csv`, mesh GLBs
- `analysis_output/viewer/` web viewer assets (overlays, minimaps, config)

## 5) Serve the viewer
```powershell
dotnet run --project WoWRollback.Cli -- serve-viewer --viewer-dir analysis_output\viewer
# or auto-detect
# dotnet run --project WoWRollback.Cli -- serve-viewer
```

Opens in browser (default http://localhost:8080). For a simple static server you can also use:

```powershell
python -m http.server 8080 --directory analysis_output\viewer
```

## Presets (Optional)
Bundle recommended UniqueID thresholds/ranges in JSON and pass via `--preset-json` (planned). Today you can replicate presets with `--max-uniqueid`.

Samples: see `WoWRollback/presets/`.

## Asset taxonomy (Optional)
Use path heuristics to enrich CSVs and viewer filters:
- M2 types: trees, bushes, rocks, props (by directory/name patterns)
- WMO types: buildings, caves, dungeons

See `docs/analysis/asset-taxonomy.md` for proposed patterns and columns.

## LK → Alpha status
See `docs/lk-to-alpha/status.md` for implemented features and roadmap.
