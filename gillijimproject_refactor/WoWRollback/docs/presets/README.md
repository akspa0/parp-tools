# Presets for UniqueID Filtering

Presets capture recommended UniqueID thresholds and optional labeled ranges for analysis and viewer filtering.

## Schema (minimal)

```json
{
  "name": "Westfall 2001 Alpha Layer",
  "description": "Early pass focused on buildings and props.",
  "uniqueId": {
    "max": 50000,
    "ranges": [
      { "min": 0,     "max": 20000, "label": "Early scaffolding" },
      { "min": 20001, "max": 35000, "label": "Buildings" },
      { "min": 35001, "max": 50000, "label": "Props/Trees" }
    ]
  },
  "filters": {
    "assetTypes": ["tree", "bush", "rock", "prop", "building"],
    "maps": ["Azeroth"]
  }
}
```

- `uniqueId.max` maps today to `--max-uniqueid` in CLI.
- `uniqueId.ranges` are for viewers and reports (planned: label-aware overlays).
- `filters.assetTypes` and `filters.maps` are optional.

## Using a preset today

```powershell
# Use the max value from the preset as --max-uniqueid
$preset = Get-Content .\WoWRollback\presets\Westfall2001.json | ConvertFrom-Json
$max = $preset.uniqueId.max

dotnet run --project WoWRollback.Cli -- alpha-to-lk `
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt `
  --max-uniqueid $max `
  --fix-holes --holes-scope self --holes-wmo-preserve true `
  --out outputs\Azeroth-session \
  --lk-out outputs\Azeroth-session\lk_adts\World\Maps\Azeroth \
  --dbd-dir ..\lib\WoWDBDefs\definitions \
  --src-dbc-dir ..\test_data\0.5.3\tree\DBFilesClient \
  --lk-dbc-dir  ..\test_data\3.3.5\tree\DBFilesClient \
  --auto-crosswalks --copy-crosswalks --report-areaid --strict-areaid
```

## Planned integration
- CLI `--preset-json <file>` to load `uniqueId.max` and pass ranges to analysis/viewer.
- Viewer overlay filters by labeled ranges and by `assetTypes` (see asset taxonomy).

## Samples
- See files under `WoWRollback/presets/` to get started.
