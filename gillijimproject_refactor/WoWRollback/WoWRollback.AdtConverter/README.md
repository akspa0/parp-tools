# WoWRollback.AdtConverter

## Overview
Standalone CLI for LK ⇄ Alpha terrain workflows: pack/unpack monolithic Alpha WDTs, convert LK ADTs, validate/inspect.

## Quick Start
```powershell
# Build
dotnet build ..\WoWRollback.sln -c Release

# Pack a monolithic Alpha WDT with converted terrain-only ADTs
dotnet run --project . -- pack-monolithic ^
  --lk-dir <path-to-lk-map-dir> ^
  --lk-wdt <path-to-lk-wdt> ^
  --map <MapName> ^
  --out .\project_output ^
  --verbose-logging

# Inspect Alpha
dotnet run --project . -- inspect-alpha --wdt <path> --out .\inspect
```

## Notes
- Outputs under `project_output/<map>_<timestamp>/` by default.
- Verbose mode dumps MCAL-related debug to `debug_mcal/YY_XX/`.
- See root README “WoWRollback.AdtConverter” section for more commands.

## See Also
- `docs/Alpha-WDT-Conversion-Spec.md`
- `docs/Alpha-Conversion-Quick-Reference.md`
- `docs/MCNK-SubChunk-Audit.md`
