# Restoring the `development` map with PM4 tooling

This is the operator runbook for the v0.6.0 PM4-backed development-map restoration. It covers the
input folders, safe output layout, single-tile smoke run, full batch, verification, and loading the
result in WoWViewer. The implementation is
[`Pm4RestoreUnauthoredSupport.cs`](../../tools/inspect/WowViewer.Tool.Inspect/Pm4RestoreUnauthoredSupport.cs:20).

## What this tool restores

`restore-unauthored` writes monolithic LK/Wrath 3.3.5 `.adt` files named
`development_<x>_<y>.adt`. For each selected tile, it combines available base terrain and existing
placements with placements resolved from PM4 collision data. Existing object placements are kept as
the starting catalog; PM4 matches can refine their transform, and only unmatched in-tile PM4 objects
are appended. If a tile has no usable base ADT, the tool can make a blank terrain tile. For tiles
without texture layers, it can bake the corresponding minimap PNG into MCCV vertex colors.

This is **object-placement restoration plus terrain assembly**, not a full terrain reconstruction.
Results remain partial: the known gap is museum objects on tiles adjacent to PM4-covered tiles that
are not themselves recovered by the current resolver.

## Required inputs

Run from the repository root in PowerShell 7 with the .NET 10 SDK available. The included workspace
data uses these locations:

| Input | Expected contents |
|---|---|
| `wow-viewer/test_data/WoWMuseum/335-dev/World/Maps/development/` | Base LK ADTs and `development.wdt`; the viewer uses this WDT and client data. |
| `wow-viewer/test_data/original_development/World/Maps/development/` | `.pm4` files and any original `_obj0.adt` placement companions. |
| `wow-viewer/test_data/development/World/Textures/Minimap/` | `development_<x>_<y>.png` minimaps used for MCCV on untextured/missing terrain. |
| `wow-viewer/output/datasets/pm4_object_association_ledger.json` | The prepared association ledger with matched asset shapes and exact-Z records. |

The restore command **consumes but does not create** the ledger. Use the prepared association ledger for
the intended full-fidelity run. If it is missing or unreadable, the tool can still supplement its
shape library from authored base ADTs, but the exact-Z ledger lookup will be empty and results may be
less complete. The selected PM4 directory must exist; `--base`, `--pm4`, `--minimap`, and `--ledger`
are explicitly supplied below so the run does not depend on machine-specific defaults.

## Safety: use a fresh output directory

`--output` is the directory that receives the generated `.adt` files—not the run root. The command
overwrites matching output ADTs and, by default, recursively removes split `_obj0.adt`, `_obj1.adt`,
`_tex0.adt`, and `_lod.adt` companions already under that output directory. **Never set `--output`
to any input directory or to a valuable prior result.** Use a new output root for each run. The
source folders remain untouched when output is separate; generated ADTs are reserialized, so do not
assume their bytes are identical to source ADTs.

The output root must have the normal loose-map layout:

```text
<run-root>/
  restoration_summary.json
  World/Maps/development/development_<x>_<y>.adt
```

The tool writes ADTs and the optional JSON summary. It does **not** create a WDT. The viewer uses
`development.wdt` from the base `335-dev` tree and overlays the generated ADTs from the run root.

## 1. Set paths and check prerequisites

Run this from `I:/parp/parp-tools` (or the equivalent repository root):

```powershell
$Repo = (Get-Location).Path
$InspectProject = Join-Path $Repo "wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj"
$MuseumRoot = Join-Path $Repo "wow-viewer/test_data/WoWMuseum/335-dev"
$BaseMapDir = Join-Path $MuseumRoot "World/Maps/development"
$Pm4MapDir = Join-Path $Repo "wow-viewer/test_data/original_development/World/Maps/development"
$MinimapDir = Join-Path $Repo "wow-viewer/test_data/development/World/Textures/Minimap"
$LedgerPath = Join-Path $Repo "wow-viewer/output/datasets/pm4_object_association_ledger.json"

$RequiredPaths = @(
    $InspectProject,
    (Join-Path $BaseMapDir "development.wdt"),
    (Join-Path $BaseMapDir "development_13_50.adt"),
    (Join-Path $Pm4MapDir "development_13_50.pm4"),
    $MinimapDir,
    (Join-Path $MinimapDir "development_13_50.png"),
    $LedgerPath
)
$Missing = @($RequiredPaths | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($Missing.Count -gt 0) { throw "Missing required inputs:`n$($Missing -join "`n")" }

$RunRoot = Join-Path $Repo ("wow-viewer/output/development-restoration/runs/" + (Get-Date -Format "yyyyMMdd-HHmmss"))
if (Test-Path -LiteralPath $RunRoot) { throw "Run folder already exists; choose a new run name: $RunRoot" }
$SmokeRoot = Join-Path $RunRoot "smoke"
$SmokeMapDir = Join-Path $SmokeRoot "World/Maps/development"
$SmokeSummary = Join-Path $SmokeRoot "restoration_summary.json"
```

If the `13_50` minimap is absent in your local corpus, pick a tile that has a PM4 file and matching
PNG, then change that coordinate consistently in the preflight check and the next command.

## 2. Smoke-test one known tile

Start with `13_50`, a known unauthored Tauren settlement case. This writes
only that tile into a new smoke output:

```powershell
dotnet run --project $InspectProject -c Debug -- `
    restore-unauthored `
    --base $BaseMapDir `
    --pm4 $Pm4MapDir `
    --minimap $MinimapDir `
    --ledger $LedgerPath `
    --output $SmokeMapDir `
    --tiles "13_50" `
    --json $SmokeSummary
if ($LASTEXITCODE -ne 0) { throw "restore-unauthored failed with exit code $LASTEXITCODE" }

$SmokeResult = Get-Content -LiteralPath $SmokeSummary -Raw | ConvertFrom-Json
$SmokeResult | Format-List
if ($SmokeResult.RoundTripFailed -ne 0 -or $SmokeResult.RoundTripPassed -ne 1) {
    throw "Smoke tile did not pass its round-trip check; inspect the console output and summary."
}
if (-not (Test-Path -LiteralPath (Join-Path $SmokeMapDir "development_13_50.adt"))) {
    throw "Expected monolithic ADT was not written."
}
```

The console should also report the loaded shape-library and ledger-lookup counts, the tile's WMO/M2
counts, and `Round-Trip Check: PASS`. Review these before starting a full batch.

## 3. Run the full batch

Use a separate output root so the smoke result remains available for comparison. `--all-tiles`
expands the candidate coordinate set using PM4 files, base ADTs, and minimap PNGs found in the
selected input folders. The number of output tiles depends on the corpus present on your machine.

```powershell
$FullRoot = Join-Path $RunRoot "full"
$FullMapDir = Join-Path $FullRoot "World/Maps/development"
$FullSummary = Join-Path $FullRoot "restoration_summary.json"

dotnet run --project $InspectProject -c Debug -- `
    restore-unauthored `
    --base $BaseMapDir `
    --pm4 $Pm4MapDir `
    --minimap $MinimapDir `
    --ledger $LedgerPath `
    --output $FullMapDir `
    --all-tiles `
    --json $FullSummary
if ($LASTEXITCODE -ne 0) { throw "restore-unauthored failed with exit code $LASTEXITCODE" }

$FullResult = Get-Content -LiteralPath $FullSummary -Raw | ConvertFrom-Json
$FullResult | Format-List
if ($FullResult.RoundTripFailed -ne 0 -or $FullResult.RoundTripPassed -ne $FullResult.TilesProcessedCount) {
    throw "One or more tiles failed round-trip validation; inspect the console output and summary before using this output."
}
```

`--json` writes batch totals (processed tiles, MCCV-baked tiles, WMO/M2 counts, round-trip pass/fail)
to the named file; per-tile details are printed to the console. **Check the counts, not just the
process exit code:** per-tile errors are reported during the run, and the summary is the evidence
that every processed tile passed the reader round-trip.

## 4. Open the generated map in WoWViewer

The generated run root contains `World/Maps/development`, so pass the **run root** (not the map
directory) as the loose overlay. Load the base WDT from `335-dev`:

```powershell
$ViewerProject = Join-Path $Repo "wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj"
dotnet run --project $ViewerProject -c Debug -- `
    --game-path $MuseumRoot `
    --build "3.3.5.12340" `
    --loose-map-overlay $FullRoot `
    --world "World\Maps\development\development.wdt"
```

For a quick visual check of the smoke output, replace `$FullRoot` with `$SmokeRoot`. The base client
provides the WDT, textures, and referenced assets; the loose overlay supplies the generated ADTs.
The WDT is not regenerated by `restore-unauthored`.

## Options and behavior

| Option | Meaning |
|---|---|
| `--base <dir>` (`--adt`) | Base `335-dev` map directory. |
| `--pm4 <dir>` (`--map`, `--input`, `-i`) | `original_development` map directory containing PM4 and optional original `_obj0` ADTs. |
| `--minimap <dir>` (`-m`) | Development minimap PNG directory. |
| `--ledger <file>` | Prepared PM4 association ledger. |
| `--output <dir>` (`-o`) | Destination map directory for monolithic `.adt` files. |
| `--tiles <x_y[,x_y...]>` (`-t`) | Process selected coordinates from the PM4 candidate set; useful for smoke tests. |
| `--all-tiles` | Add coordinates discovered in base ADTs and minimap PNGs to the PM4 candidate set. |
| `--json <file>` | Write batch summary JSON. |
| `--no-validate` | Disable the ADT read-back check; do not use for normal restoration. |
| `--no-converge` | Disable multi-source placement convergence; not recommended. |
| `--no-bake-mccv` | Disable minimap-to-MCCV bake on untextured/missing tiles. |
| `--split-obj` | Also emit split object ADTs; the normal deliverable is monolithic, so omit this for WoWViewer. |
| `--no-clean-companions` | Do not purge stale split companions from the output directory. Use only when you understand the existing output contents. |

## Troubleshooting

- **“PM4 directory … does not exist”** — `--pm4` should point to the `World/Maps/development`
  directory itself, not to the repository's `test_data` parent.
- **Zero tiles processed** — without `--all-tiles`, only coordinates found in PM4 filenames are
  candidates. Check the coordinate spelling, or use `--all-tiles` for the corpus run.
- **Ledger lookup is zero** — check that `--ledger` points to the prepared JSON ledger. The command
  can fall back to shape profiles from base ADTs, but that is not the same matching signal.
- **Missing map / no terrain in the viewer** — load `development.wdt` from the base `335-dev`
  directory and attach the output run root with `--loose-map-overlay`; the tool does not write a WDT.
- **Untextured tile is black or missing its minimap colors** — verify `--minimap` contains the
  correctly named PNG. MCCV is baked only when the assembled tile has no texture layers; textured
  tiles intentionally do not receive this bake.
- **Round-trip failures or fewer passes than tiles** — do not treat the batch as complete even if
  the command returned exit code 0. Keep the run folder and inspect the per-tile console errors and
  summary before loading or distributing the results.
- **Expected structures are still absent** — restoration is partial and PM4-driven; see the known
  adjacent-tile gap above.
