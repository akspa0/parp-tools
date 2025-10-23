# WoWRollback.Cli

## Overview
Command-line interface for the WoWRollback toolkit. Provides:
- analyze-map-adts (analyze loose ADTs, generate viewer)
- alpha-to-lk (Alpha WDT → LK ADTs with burial, holes, MCSH, AreaIDs)
- lk-to-alpha (patch existing LK ADTs with same terrain logic)
- serve-viewer (Kestrel-based static server)
- probe-archive / probe-minimap (diagnostics)

## Prerequisites
- .NET SDK 9.0
- Optional: LK client root for AreaTable mapping (`--lk-client-path`)
- Optional: Crosswalk CSVs (`--crosswalk-dir`/`--crosswalk-file`)

## Build
```powershell
dotnet build ..\WoWRollback.sln -c Release
```

## Usage
### Alpha → LK
```powershell
dotnet run --project . -- \
  alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow" \
  --default-unmapped 0
```

### LK → Alpha Patcher
```powershell
dotnet run --project . -- \
  lk-to-alpha \
  --lk-adts-dir .\wrb_out\lk_adts\World\Maps\Azeroth \
  --map Azeroth \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out .\patched_lk_az
```

### Analyze Loose ADTs
```powershell
dotnet run --project . -- analyze-map-adts \
  --map <name> \
  --map-dir <dir> \
  --out analysis_output
```

### Serve Viewer
```powershell
dotnet run --project . -- serve-viewer --viewer-dir analysis_output\viewer --port 8080
```

## Notes
- Neighbor-aware MCNK holes clearing uses pre-scan of buried placement refs and clears holes in self+8-neighbors.
- Shadows (MCSH) can be zeroed per-chunk with `--disable-mcsh`.
- Crosswalk flags: prefer `--crosswalk-dir` / `--crosswalk-file`; legacy `--dbctool-patch-*` still work.
