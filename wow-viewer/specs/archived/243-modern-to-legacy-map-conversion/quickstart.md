# Quickstart — Modern-to-Legacy Map Conversion

Date: 2026-09-18

## Prerequisites

- A modern client root readable by the viewer (local CASC install, or an MPQ root). Client roots are
  configuration, never hardcoded (Constitution VI).
- The viewer/harvest tool built: `dotnet build wow-viewer/WowViewer.slnx -c Debug`.

## Convert one modern map to LK v18

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest `
  -- convert-map `
  --client-root "I:\wow12\World of Warcraft" `
  --map Azeroth `
  --target lk-adt-v18 `
  --output-dir "I:\parp\parp-tools\wow-viewer\output\convert\azeroth-lk"
```

## Convert the same map to Alpha 0.5.3

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest `
  -- convert-map `
  --client-root "I:\wow12\World of Warcraft" `
  --map Azeroth `
  --target alpha-wdt-0.5.3 `
  --output-dir "I:\parp\parp-tools\wow-viewer\output\convert\azeroth-alpha"
```

## Batch several maps with asset inclusion

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest `
  -- convert-map `
  --client-root "I:\wow12\World of Warcraft" `
  --map Azeroth --map Kalimdor `
  --target lk-adt-v18 `
  --include-assets `
  --output-dir "I:\parp\parp-tools\wow-viewer\output\convert\batch-lk"
```

## Verify

1. **Load in the viewer**: open the produced `.wdt` (LK v18 or Alpha 0.5.3) and confirm terrain,
   textures, liquids and placements render (SC-002, operator visual witness).
2. **Read the report**: the run writes a per-tile merge report (kept/merged/dropped layers,
   unresolved textures) and, with `--include-assets`, an asset manifest.
3. **Determinism**: rerun the same command; outputs must be byte-identical (SC-004).

## In the Editor

Open the **Map Converter** dialog, choose the modern source direction and the target, pick the input,
and run — no per-tile or per-format input is required (FR-005).

## Proof boundary

Compilation and unit tests are not runtime proof. A real modern map converted to each target and
loaded in the viewer is the operator-owned acceptance witness (Constitution III, AGENTS.md §9.2).
