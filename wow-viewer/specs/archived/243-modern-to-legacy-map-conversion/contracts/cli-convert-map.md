# Contract — CLI `convert-map` (batch driver)

Thin wrapper over `ModernToLegacyMapConversionService` (Constitution II). Lives in the harvest tool
(`WowViewer.Tool.Harvest`), alongside the existing `synthetic-minimap` command.

```text
convert-map
  --client-root <dir>            modern client root (CASC install or MPQ root)
  --map <name>                   repeatable; one or more modern map names
  --target <lk-adt-v18|alpha-wdt-0.5.3>
  --output-dir <dir>             generated project folder (never a client root)
  [--include-assets]             copy referenced textures/models/minimaps + manifest
  [--casc-product <p1,p2>]       CASC products (when --client-root is a CASC install)
  [--cdn-fill]                   allow CDN fetch for missing files
  [--verbose]
```

## Behaviour

- Converts every `--map` in one run; each map succeeds or fails independently (FR-004).
- Writes LK v18 ADT/WDT or Alpha 0.5.3 WDT under `--output-dir` with provenance (FR-006).
- Emits a per-tile merge report and, with `--include-assets`, an asset manifest (FR-003/FR-007).
- Exit code 0 when at least one map succeeded; non-zero only when every map failed.

## Example

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest `
  -- convert-map `
  --client-root "I:\wow12\World of Warcraft" `
  --map Azeroth `
  --target lk-adt-v18 `
  --output-dir "I:\parp\parp-tools\wow-viewer\output\convert\azeroth-lk"
```
