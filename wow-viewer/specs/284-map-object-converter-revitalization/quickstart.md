# Implementation Handoff and Validation Commands

**Planning only — new projects/commands below do not exist yet.** Implement [tasks.md](tasks.md)
one validated phase at a time. These are the intended CLI contracts, not commands run this session.
Future implementation must replace assumptions with the measured source/target profiles and evidence.

**Operator correction — 2026-10-21**: Complete [Spec 285 v65 prerequisites](../285-v65-terrain-interchange/quickstart.md)
before map export. Current MPQ/loose/**CASC** harvesting supplies native terrain plus typed placement
metadata. Object geometry/masks are excluded. Old Converter executables/wrappers/parity are not a
dependency; port useful transformation logic behind the modern dataset export adapter.

## Start the next implementation chat

Read `specs/STATUS.md`, `memory-bank/activeContext.md`, then this spec/plan/tasks and Git state.
Preserve the current Spec 283/UI work. Start with T001–T009, not whole-map conversion or new readers.
Do not change `.specify/feature.json`'s persisted selection or switch the user's branch accidentally.
For Spec Kit scripts, use explicit process-local feature overrides:

```powershell
Set-Location I:/parp/parp-tools/wow-viewer
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/284-map-object-converter-revitalization'
$env:SPECIFY_FEATURE = '284-map-object-converter-revitalization'
& ./.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks
```

The feature override identifies the planning directory; it does not create/switch a Git branch.

## Agent checks after bounded code changes

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj `
  -c Debug --filter 'FullyQualifiedName~Conversion'
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Editor.Tests/WowViewer.Core.Editor.Tests.csproj `
  -c Debug --filter 'FullyQualifiedName~Conversion'
```

When UI integration changes and for final validation:

```powershell
pwsh -NoProfile -File I:/parp/parp-tools/wow-viewer/eng/ui/Verify-UiReference.ps1
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Use conversion test namespaces that make these filters select the actual tests; verify nonzero
discovery counts. Record exact commands, exits and baseline failures. Do not rerun broad checks
without changed code or an unresolved concern. None of these establish visual/client proof.

## Operator: bounded real 1.60.1 → 0.5.3 pilot (future commands)

Use configured client roots. Do not paste machine-local client paths into source or portable tests.
Select an asymmetric adjacent-tile pair containing real terrain, one static prop and one WMO.
Record exact product/build and target fingerprint; `1.60.1` and `0.5.3` alone are incomplete identities.

```powershell
Set-Location I:/parp/parp-tools/wow-viewer
$sourceClient = Read-Host 'Source 1.60.1 client root'
$targetClient = Read-Host 'Target 0.5.3 client root'
$mapName = Read-Host 'Real source map name'
$pilotTiles = Read-Host 'Canonical pilot tiles, x,y;x,y'
$v65Store = Read-Host 'Finalized v65 store from current harvester (including CASC source when selected)'
$mapTool = './tools/map-convert/WowViewer.Tool.MapConvert/WowViewer.Tool.MapConvert.csproj'
$outputRoot = Join-Path (Get-Location) 'output/conversion-284/pilot'
$requestPath = Join-Path $outputRoot 'pilot.request.json'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

dotnet run --project $mapTool -c Debug -- inspect `
  --source-client $sourceClient --target-client $targetClient `
  --dataset $v65Store --input "map:$mapName" --tiles $pilotTiles

dotnet run --project $mapTool -c Debug -- plan `
  --source-client $sourceClient --target-client $targetClient `
  --dataset $v65Store --input "map:$mapName" --tiles $pilotTiles `
  --preset target-compatible --include-objects --reitemize-ids `
  --output $outputRoot --request-out $requestPath
```

Review the actual detected profiles, hashes, masks/feature losses, dependencies, ID scope and output
paths. If profile evidence conflicts, choose the specific supported profile with the documented
override; never force an unavailable writer. For a partial-map patch, supply complete retained-ID
reservations; for a new standalone pilot namespace, record that no existing placements are retained.
The dataset identity must match the selected source revision. CASC ingestion records product/build/
configuration and FileDataID companion resolution; unavailable native fields block export rather
than re-opening source terrain silently. Python reads the store and streams native records to C#.

Run only after reviewing the printed loss codes:

```powershell
$lossCodes = Read-Host 'Accepted loss codes from preview, comma-separated; blank if none'
if ([string]::IsNullOrWhiteSpace($lossCodes)) {
    dotnet run --project $mapTool -c Debug -- run --request $requestPath
} else {
    dotnet run --project $mapTool -c Debug -- run `
      --request $requestPath --accept-loss $lossCodes
}

$runDirectory = Read-Host 'Run directory printed by the tool'
dotnet run --project $mapTool -c Debug -- validate `
  --run $runDirectory --report-out (Join-Path $runDirectory 'readback-validation.json')
```

CLI cannot infer a live loaded client. It accepts explicit roots or `known-good:<stable-key>` with
`--viewer-settings <existing-settings-file>`. Editor supplies the actual mounted source snapshot.
Output is loose files in the project output root; it is never installed or packed into MPQ/CASC.

## Operator: prove object tools are independent (future commands)

These commands inspect independently without a map conversion. Use the same target client.

```powershell
$modelTool = './tools/model-convert/WowViewer.Tool.ModelConvert/WowViewer.Tool.ModelConvert.csproj'
$wmoTool = './tools/wmo-convert/WowViewer.Tool.WmoConvert/WowViewer.Tool.WmoConvert.csproj'
$modelLocator = Read-Host 'Real model relative path or filedataid:number'
$wmoLocator = Read-Host 'Real WMO root relative path or filedataid:number'

dotnet run --project $modelTool -c Debug -- inspect `
  --source-client $sourceClient --target-client $targetClient --input $modelLocator
dotnet run --project $wmoTool -c Debug -- inspect `
  --source-client $sourceClient --target-client $targetClient --input $wmoLocator
```

Repeat plan/run/validate using separate `output/conversion-284/model` and `.../wmo` roots. Record
standalone and map-composed results. Animated/static fallback and source-mesh generation must be
explicitly selected after inspecting real limits; no universal numerical error budget is assumed.
Inspect each generated candidate before accepting substitution.

## Operator: Editor and target witness

1. In Editor → Converters, open Map, Model and WMO separately. Confirm the actual loaded source,
   select the known-good target, then use loaded-as-target/swap and inspect direction diagnostics.
2. Recreate a CLI request, compare its canonical options/IDs/content inventory, run and cancel a
   bounded operation, then open output/report. Capture concise labels and text-only glyph fallback.
3. Inspect the Alpha pilot in the viewer: asymmetric coordinates, terrain layers/holes/liquids,
   placement counts/transforms/textures, WMO exterior/interior and collision dispositions.
4. Use an already-supported operator loose-content target-loading workflow for actual 0.5.3
   acceptance. Record exact target build, generated hashes and observations. If such a workflow
   is unavailable, leave that witness open; do not add packaging or installation to this feature.
5. Only after the bounded target witness passes, prepare whole-map and multi-map runs using the
   same contracts. Those runs, generation heavy work and visual proof remain operator-owned.

## Required evidence matrix

Writer refresh follows [the actual viewer creation path](contracts/map-writer-reuse.md): canonical
LK ADT/WDT/WDL libraries plus the existing Alpha public boundary. Read back written files and join
Spec 285 source/target layer reports; note metadata defaults and native-versus-derived WDL. An
existing-template height patch is not full new-map creation. Keep source report readiness separate
from later output acceptance; no obsolete Converter executable is needed.

| Criterion | Required output | Owner |
|---|---|---|
| SC-001 | Client evidence snapshots; both orientations; unavailable-route diagnostics | Agent prepares; selected real-client data required |
| SC-002 | ID map, real placement/reference readback, zero collision/dangling counts | Bounded automated check |
| SC-003 | Real terrain + prop + WMO and closed dependencies; viewer and actual target captures | Agent prepares; operator visual/target witness |
| SC-004 | Three independent CLI results and matching Editor requests/content | Agent contract checks; operator Editor witness |
| SC-005 | Source/generated meshes, errors, topology/lineage maps, explicit candidate decisions | Agent checks; operator generation/review |
| SC-006 | Recorded Select → Options → Review → Run → Result walkthrough and glyph fallback | Operator |
| SC-007 | Original algorithm semantics checked against real native data; protected-file diff audit | Agent plus operator regression observations |
| SC-008 | Actual CASC and MPQ/loose → v65 evidence, source-detached terrain export, no old Converter dependency | Spec 285 checks plus operator data witness |
| SC-009 | Named canonical writer/control path, field capability table and independent mapped layer-opacity/metadata readback | Agent bounded checks; operator appearance/target proof |

Each receipt includes files changed, exact commands/exits, configured roots/builds/fingerprints,
output hashes and criterion → evidence mapping. Record losses, partial failures and absent proof.
Real-file structural readback and unit tests cannot substitute for target appearance/behavior.
