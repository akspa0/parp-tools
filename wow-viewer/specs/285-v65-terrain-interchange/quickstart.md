# v65 Implementation Handoff

**Planning only. All v65 modules/profile/commands below are proposed, not implemented or run.**
Begin with [tasks.md](tasks.md). Spec 284 waits for native terrain, typed placement and reverse-stream
gates. Keep placement metadata in v65; exclude WMO/M2 geometry, object masks and mesh libraries.

## Spec Kit routing

```powershell
Set-Location I:/parp/parp-tools/wow-viewer
$env:SPECIFY_FEATURE_DIRECTORY = 'I:/parp/parp-tools/wow-viewer/specs/285-v65-terrain-interchange'
$env:SPECIFY_FEATURE = '285-v65-terrain-interchange'
& ./.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks
```

Process-local routing only; preserve the current Git branch and persisted `.specify/feature.json`.

## Bounded implementation checks

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj `
  -c Debug --filter 'FullyQualifiedName~Harvest.V65'
Set-Location I:/parp/parp-tools/wow-viewer/data-harvester
uv run pytest tests/v65 -q
```

Require nonzero focused test discovery, exact command/exits and actual source-derived native records.
No test run establishes real CASC compatibility, visual output or target-client acceptance.

## Operator: future real CASC pilot

Use a bounded actual 1.60.1 CASC selection plus MPQ/loose controls. Record exact install/product/
build/config identity and native file/companion hashes. Unknown product or missing local files
must be reported. CDN fill remains disabled by default; no CASC archive output.

```powershell
Set-Location I:/parp/parp-tools/wow-viewer/data-harvester
$sourceClient = Read-Host 'Configured local CASC source root'
$sourceBuild = Read-Host 'Exact selected source build identity'
$sourceProduct = Read-Host 'Selected CASC product'
$mapName = Read-Host 'Source map name'
$pilotTiles = Read-Host 'Bounded tile selection x,y;x,y'
$pilotRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path '../output/datasets/v65/pilot'))
New-Item -ItemType Directory -Path $pilotRoot -Force | Out-Null
$v65Store = Join-Path $pilotRoot 'source.zarr'

uv run python scripts/v65_dataset.py ingest `
  --client-root $sourceClient --build $sourceBuild `
  --source-backend casc --casc-product $sourceProduct `
  --map $mapName --tiles $pilotTiles --output $v65Store
uv run python scripts/v65_dataset.py inspect --store $v65Store
uv run python scripts/v65_dataset.py validate `
  --store $v65Store --report (Join-Path $pilotRoot 'validation.json')
```

Do not use the desktop, drive root or OS temporary storage. Repeat with MPQ/loose source and a
different output store, without overwriting source datasets. No full-client harvest is implied.

Inspect native145 samples/zeros, all layers/full-resolution alpha, exact hole topology, liquid
layers, metadata, row validity, typed placement IDs/flags and reference joins. Check IDs above
`2^24` and the integer upper range. Inspect the per-stage chunk ledger and confirm zero geometry/
mask payloads. Opaque chunks must remain labelled opaque.

## Source-detached terrain export and object path

Python's `stream` command supplies the C# adapter under supervision; never pipe binary frames
through PowerShell text pipelines. Use Spec 284's Map CLI `--dataset` request so the bridge preserves
raw bytes/progress/cancellation and validates finalization/digests. Detach source terrain access
for the terrain export check; target terrain material dependencies must remain explicitly resolved.

Full map/object conversion reads placements from v65 and separately resolves actual assets from
their declared read-only MPQ/loose/CASC sources. Geometry is never recovered from object masks or
incomplete V22 model payloads. Spec 284 owns target IDs/remeshing/grouping and visual acceptance.

## Retained appearance and future-model checks (future commands)

All native/source/unknown terrain records and referenced textures stay in v65 after export.
Before claiming complete appearance, verify every referenced diffuse/auxiliary/height/liquid
texture has source bytes/format/mips and native RGBA PNGs, plus build/path/FDID identity and
cell/layer usage. No opaque-alpha, resized RGB, proxy or dominant-label substitution.

```powershell
Set-Location I:/parp/parp-tools/wow-viewer/data-harvester
$pngOutput = Join-Path $pilotRoot 'tileset-pngs'
uv run python scripts/v65_dataset.py export-textures --store $v65Store --output $pngOutput
$textureKey = Read-Host 'Retained texture stable ID from the manifest'
uv run python scripts/v65_dataset.py query-texture --store $v65Store --texture $textureKey
uv run python scripts/v65_dataset.py query-chunks --store $v65Store
```

Use `query-cell` with recorded map/tile/chunk/cell coordinates to inspect every contributing layer,
native mask footprint and UV/material/effect parameters. Compare inverse texture usage and
native RGBA/dimensions/mips with source records. Add an interpretation/model view and verify
committed hashes remain unchanged, including unknown chunks. Future consumers need no source
client for retained resources; missing dependencies remain explicit rather than guessed.

## Layer opacity reporting (future command)

```powershell
Set-Location I:/parp/parp-tools/wow-viewer/data-harvester
$layerSelection = Read-Host 'Selection manifest for source tile/chunks/layers'
$opacityOutput = Join-Path $pilotRoot 'layer-opacity'
uv run python scripts/v65_dataset.py report-opacity `
  --store $v65Store --selection $layerSelection --output $opacityOutput
```

Inspect one record per declared layer, including base/zero/fully overpainted/repeated-texture and
unavailable layers. Compare MCLY flags/effects/order, raw MCAL ranges, decode/fixup validity,
histograms/coverage and cell summaries. Decoded alpha and effective contribution are separate;
unknown blending remains unknown. Source reports do not wait for a writer. Later, compare an
independently decoded output report through the explicit source→target layer map, with quantization
and metadata losses recorded. No raw-channel normalization or silently dropped invisible layers.

## Full acceptance receipt matrix

| Criterion | Evidence required |
|---|---|
| SC-001 | Native decode→stream→store→reverse-stream equality, exact identity and row validity. |
| SC-002 | Every real observed chunk occurrence accounted for at each stage; native vs opaque distinction. |
| SC-003 | Typed placement/reference equality and zero geometry/mask payload audit. |
| SC-004 | Source-detached dataset export through current C# adapters, no old Converter invocation. |
| SC-005 | Read-only audited migration; explicit missing/re-harvest fields, original stores unchanged. |
| SC-006 | Actual CASC install/product/build/FDID companion ingestion/readback plus MPQ/loose controls. |
| SC-007 | Source-detached indexed raw/native/unknown terrain occurrence retrieval and exact hashes. |
| SC-008 | Complete referenced texture inventory, retained source bytes and native RGBA PNG/mip coverage. |
| SC-009 | Real cell/layer/native-mask/material→PNG query and inverse texture usages agree. |
| SC-010 | Committed record/resource hashes unchanged after export/new interpretation/model view. |
| SC-011 | Every-layer metadata/alpha/contribution/validity report; later mapped writer readback and numeric/quantization checks. |

Record files changed, exact commands/exits, roots/builds/config identities, source/output hashes,
coverage and limitations. No broad harvest/training/GPU run or client visual proof is agent-owned.
