# WoWRollback - World of Warcraft Map Analysis & Rollback Toolkit

**Digital archaeology + conversion toolkit** focused on:
## Condensed Overview (2025-11-08)

- Main issue: LK ADT positions → Alpha WDT writeout (compute MAIN offsets, embed ADT payloads, validate).
- Implemented: MPQ overlay precedence (FS > root-letter > locale-letter > root-numeric > locale-numeric > base), DBC locale‑first for `DBFilesClient/*`, plain patch support, WDT tile presence fallback, tee logging (`--log-dir`/`--log-file`).

### Quick Start (CLI)

Alpha → LK (minimal):
```powershell
dotnet run --project WoWRollback.Cli -- alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 75000 --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow"
```

Pack monolithic Alpha WDT from MPQ client (diagnostics):
```powershell
dotnet run --project WoWRollback.Cli -- pack-monolithic-alpha-wdt \
  --client-path "H:\\WoWDev\\modernwow" \
  --map "CataclysmCTF" \
  --out out\CataclysmCTF.wdt \
  --verbose --verbose-logging --log-dir .\logs\
```

### Links
- memory-bank/activeContext.md — Current focus and TODOs
- memory-bank/progress.md — Snapshot of progress
- memory-bank/systemPatterns.md — Overlay precedence, WDT fallback
- memory-bank/techContext.md — Runtime/env and modules

---
## Requirements
- .NET SDK 9.0 (x64)
- Alpha data (extracted WDT/ADT/DBC) as needed
- LK 3.3.5 client or DBCs when required for AreaTable/Map guards
- Optional: MPQ client install for minimap extraction

## License
See LICENSE in repository root.
