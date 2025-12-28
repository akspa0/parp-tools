# Alpha 0.5.3 WDT → LK → Alpha 0.5.3 WDT (Roundtrip Runbook)

This document describes how to take a **WoW Alpha 0.5.3 monolithic WDT** (`<Map>.wdt`), convert it to **LK 3.3.5-compatible map files**, then convert those LK outputs back into a **monolithic Alpha WDT**.

It uses only these components:
- `AlphaWDTAnalysisTool` (CLI) — Alpha → LK export
- `WoWRollback.Cli` (CLI) — LK → Alpha monolithic packer
- `WoWRollback.LkToAlphaModule` (library) — used internally by `WoWRollback.Cli` for LK→Alpha structures/writers

## Summary (Pipeline)

- **Step A (Alpha → LK)**: `AlphaWDTAnalysisTool` exports:
  - `World/Maps/<Map>/<Map>.wdt` (LK WDT)
  - `World/Maps/<Map>/<Map>_<x>_<y>.adt` (LK ADTs)
- **Step B (LK → Alpha)**: `WoWRollback.Cli pack-monolithic-alpha-wdt` consumes the LK WDT + map folder and writes:
  - `<out>/<Map>.wdt` (Alpha monolithic WDT)
  - `<out>/<Map>.wdl` (optional; emitted as a sibling file)

## Inputs

- **Alpha WDT**:
  - `.../World/Maps/<Map>/<Map>.wdt`
- **Community listfile** (CSV, required by AlphaWDTAnalysisTool):
  - `test_data/community-listfile-withcapitals.csv`
- **Optional**:
  - LK 3.x listfile (text) for better asset fixups
  - DBCTool.V2 `compare/v2/` mapping outputs (for strict AreaID patching)

## Output Locations (Suggested)

Use a single session root so the GUI can orchestrate later:

```
<sessionRoot>/
  stepA_alpha_to_lk/
  stepB_lk_to_alpha/
```

## Step A: Alpha → LK (AlphaWDTAnalysisTool)

Run the AlphaWDTAnalysisTool CLI to export LK map files.

### Command (single map)

```powershell
dotnet run --project .\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli -- `
  --input "<alpha_root>\World\Maps\<Map>\<Map>.wdt" `
  --listfile ".\test_data\community-listfile-withcapitals.csv" `
  --out "<sessionRoot>\stepA_alpha_to_lk" `
  --export-adt --export-dir "<sessionRoot>\stepA_alpha_to_lk" `
  --dbctool-patch-dir "<dbctool_session>\compare\v2"
```

### Notes

- The tool writes LK outputs under the export root in a **client-like layout**.
- AreaIDs:
  - If `--dbctool-patch-dir` is provided, patching is **strict CSV-only**. If a mapping row is missing, `0` is written.
- Asset fixups:
  - Use `--lk-listfile <3x.txt>` if available.
  - Use `--no-fixups` if you want to preserve original paths.

### Expected outputs

```
<sessionRoot>/stepA_alpha_to_lk/
  World/
    Maps/
      <Map>/
        <Map>.wdt
        <Map>_<x>_<y>.adt
  csv/
    maps/
      <Map>/
        areaid_verify_<x>_<y>.csv   (verbose runs)
```

## Step B: LK → Alpha monolithic WDT (WoWRollback.Cli)

Important distinction:
- `WoWRollback.Cli lk-to-alpha` is a **patcher for LK ADTs** (bury/holes/MCSH). It does **not** write a monolithic Alpha WDT.
- The monolithic Alpha writer is `WoWRollback.Cli pack-monolithic-alpha-wdt`.

### Command (build monolithic Alpha WDT from LK outputs)

```powershell
dotnet run --project .\WoWRollback\WoWRollback.Cli -- `
  pack-monolithic-alpha-wdt `
  --lk-wdt "<sessionRoot>\stepA_alpha_to_lk\World\Maps\<Map>\<Map>.wdt" `
  --lk-map-dir "<sessionRoot>\stepA_alpha_to_lk\World\Maps\<Map>" `
  --out "<sessionRoot>\stepB_lk_to_alpha\<Map>.wdt"
```

### Notes

- This command reads LK map inputs (WDT + tiles) and emits a **monolithic Alpha WDT**.
- Internally, `WoWRollback.Cli` uses `WoWRollback.LkToAlphaModule` types/builders for LK→Alpha structure conversion.
- The CLI also attempts to write a sibling WDL:
  - `<sessionRoot>\stepB_lk_to_alpha\<Map>.wdl`

## Optional: Patch LK ADTs before packing (WoWRollback.Cli lk-to-alpha)

If you want to apply the “bury placements / fix holes / disable mcsh” logic to the LK ADTs before packing back to Alpha:

```powershell
dotnet run --project .\WoWRollback\WoWRollback.Cli -- `
  lk-to-alpha `
  --lk-adts-dir "<sessionRoot>\stepA_alpha_to_lk\World\Maps\<Map>" `
  --map <Map> `
  --max-uniqueid 43000 `
  --fix-holes --disable-mcsh `
  --out "<sessionRoot>\stepA_alpha_to_lk_patched"
```

Then point Step B at the patched folder’s WDT/map-dir instead.

## GUI Orchestration Notes (Future: WoWRollback.Gui)

A GUI “Roundtrip” button can orchestrate these as pure file-pipeline steps.

### Required user inputs

- `alphaWdtPath` (file)
- `communityListfileCsvPath` (file)
- `sessionRoot` (dir)

### Optional user inputs

- `dbctoolPatchDir` (dir; points at DBCTool.V2 session `compare/v2/`)
- `lkListfilePath` (file)
- `maxUniqueId` (int; if using the optional LK patch step)
- Flags:
  - `noFixups` (bool)
  - `fixHoles` (bool)
  - `disableMcsh` (bool)

### Deterministic artifacts (for GUI to link)

- Step A outputs:
  - `World/Maps/<Map>/<Map>.wdt`
  - `World/Maps/<Map>/<Map>_<x>_<y>.adt`
  - `csv/maps/<Map>/*.csv`
- Step B outputs:
  - `<Map>.wdt`
  - `<Map>.wdl` (best-effort)

### Minimal health checks

- Step A:
  - Ensure `World/Maps/<Map>/<Map>.wdt` exists
  - Ensure at least one `World/Maps/<Map>/<Map>_*_*.adt` exists
- Step B:
  - Ensure output `<Map>.wdt` exists and is non-empty

