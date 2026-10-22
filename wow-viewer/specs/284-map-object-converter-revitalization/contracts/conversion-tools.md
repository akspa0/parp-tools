# Contracts: Separate CLI and Editor Converter Tools

**Status**: Proposed. No HTTP service is needed: these are local library/CLI/Editor contracts.
All three tools use the same versioned request/report model in [data-model.md](../data-model.md).

## Independent hosts

| Tool | Proposed project | Scope |
|---|---|---|
| Map | `tools/map-convert/WowViewer.Tool.MapConvert/WowViewer.Tool.MapConvert.csproj` | Map/tile conversion, re-itemization and composition of required object converters. |
| Model | `tools/model-convert/WowViewer.Tool.ModelConvert/WowViewer.Tool.ModelConvert.csproj` | Independent M2/MDX conversion and explicit static/generated candidates. |
| WMO | `tools/wmo-convert/WowViewer.Tool.WmoConvert/WowViewer.Tool.WmoConvert.csproj` | Independent WMO conversion, regrouping and generated mesh candidates. |

Each host is usable without a viewer process. The old `WowViewer.Tool.Converter` is historical
algorithm reference, not a functional dependency or compatibility gate. No mandatory wrapper or
legacy-command parity work. New requests default to deterministic target ID re-itemization.

Map requests consume finalized **v65 native terrain + typed placement metadata** from Spec 285.
Python reads Zarr and supplies the versioned reverse stream; C# performs target serialization.
Source roots (MPQ/loose/CASC) resolve existing dataset identity or prepare ingestion through current
data-harvester tooling. A missing native field is a coverage blocker, never a direct-client fallback.

## Shared CLI vocabulary

```text
inspect --source-client <root-or-selection> --target-client <root-or-selection>
        --input <map:name|relative-asset-path|filedataid:number> [domain selection]
plan    <same selectors> --preset <target-compatible|preserve-features|custom>
        --output <dir> --request-out <file> [domain options]
run     --request <file> [--accept-loss <code,...>]
validate --run <run-dir> --report-out <file>
```

- Common selectors: explicit client root or `known-good:<stable-key>` with
  `--viewer-settings <path>` pointing at existing local settings. A missing key is an error, never a
  different-client fallback. `loaded` is Editor-only; CLI requires roots or a captured request.
- Optional `--source-profile` / `--target-profile` overrides retain conflicting evidence and cannot
  bypass unsupported readers/writers. Identity and route diagnostics are shared with Editor.
- Map: `--tiles <x,y;x,y>` or `--all-tiles`, `--include-objects` (default), `--terrain-only`,
  `--reitemize-ids` (default), `--id-start <n>`, `--reserved-ids <manifest>`.
  Add `--dataset <finalized-v65.zarr>` with source/build/digest validation; object metadata is read
  from v65, while geometry comes from separate asset sources. CASC source selectors retain product/
  build/FDID identity; CASC is read-only input and CDN fill defaults off.
  Whole-map/batch execution is operator-owned. Terrain-only output is labelled incomplete for SC-003.
- Model: material policy and explicit `--static-replacement --pose <source-pose>` when required.
- WMO: regrouping policy and measured target budgets, with portal/interior/collision dispositions.
- Model/WMO candidate plan: `--generate source-mesh` plus validated geometry/error options.
  Generation remains explicit; candidate rejection prevents substitution.
- `inspect` emits capability evidence; `plan` may write request/review metadata but no target
  content. `run` rechecks identity/input hashes, enforces reviewed losses, stages and validates.
- `validate` is structural/real-file readback only. It cannot mark viewer/target proof as passed.
- Cancellation is cooperative (including Ctrl+C) with partial-state reporting and no completed
  manifest. Existing output requires a new run directory; no client-root output is accepted.

New CLI exit codes: `0` = requested operation complete; `2` = invalid/ambiguous/blocked route or
unaccepted loss; `3` = conversion/readback failure or partial batch; `130` = cancelled.
Reports explain distinctions. Inspecting an unavailable requested route returns `2` with evidence.

## Shared service boundary (conceptual signatures)

The [canonical writer contract](map-writer-reuse.md) requires the actual viewer New Map writer
path behind `MapTargetMaterializer`. Source/decode/effective layer reports from Spec 285 accompany
writer planning and independent output readback. Existing height patching is a distinct mode,
not fresh map serialization; silent defaulted metadata is not preservation.

```text
ClientCapabilityProbe.Inspect(clientSelection, selectedContent, cancellation)
ConversionRouteResolver.Resolve(sourceEvidence, targetEvidence, converterKind)
MapConversionService.Plan(request, sources, cancellation)
ModelConversionService.Plan(request, sources, cancellation)
WmoConversionService.Plan(request, sources, cancellation)
<domain service>.Execute(plannedRequest, progress, cancellation)
<domain service>.Validate(run, cancellation)
PlacementIdentityGraphBuilder.Build(records, identityDecisions)
PlacementIdAllocator.Allocate(graph, reservations, targetLimits)
PlacementReferenceRewriter.Apply(conversionDocument, allocation)
```

Sources are passed as read-only interfaces, not ImGui, app internals or hardcoded roots. Planning
returns a resolved request, review summary, feature dispositions and blockers. Execution produces
reports and manifests, not UI text. Map composition passes explicit dependency subrequests to
Model/Wmo services. Core.Editor coordinates jobs and progress; it does not become another converter.

## Minimum request/report envelope

Map requests bind the retained native/unknown appearance metadata, cell/layer mappings and source/
PNG texture resource digests from Spec 285. Python supplies typed resource/metadata frames as well
as terrain/placements; C# uses no direct Zarr reader. Full appearance acceptance requires complete
referenced texture/metadata coverage; unsupported target semantics are reported reductions.

Request fields: `schemaVersion`, `converterKind`, `source`, `target`, `selection`, `targetProfile`,
`preset`, `options`, `idPolicy`, `dependencyPlan`, `outputRoot`, `requiredLossCodes`, `requestDigest`.
Report fields: `schemaVersion`, `requestDigest`, `state`, `publicationState`, `sourceEvidence`,
`targetEvidence`, `items`, `losses`, `blockers`, `outputs`, `placementIdMap`, `pathMap`, `topologyMap`,
`generationCandidates`, `verificationLevels`, `receiptPaths`.

Schema version, profile/tool versions, hashes, unknown values and loss codes are explicit. CLI/Editor
parity compares deterministic content/decisions; run timestamps, duration and run IDs may differ.

## Editor contract

- Sole execution home: **Editor → Converters → Map / Model / WMO**. Other presets may navigate to
  this home, without executing a conversion there.
- Shared source/target picker defaults to the actual mounted source; either slot accepts saved
  known-good clients. Explicit loaded-as-target and swap retain role intent.
- Tool flow: Select → Options → Review → Run → Result. Same resolved request as CLI; clear output,
  cancellation, partial/completed status, open-folder/report actions.
- Domain options are separate and progressively disclosed. Optional emoji supplement readable
  labels; absent glyphs have plain-text fallback. No color-only or icon-only loss/error state.
- Use `SharedUiWidgets`, existing UI manager/page injection and authoritative control registry.
  Do not add another global menu or app/scene state. Hidden tools do no per-frame conversion work.
