# Implementation Plan: Map and Object Converter Revitalization

**Feature**: `284-map-object-converter-revitalization` | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)  
**Actual branch**: `280-ui-rework`, preserved with unrelated UI WIP  
**Delivery**: Planning only. All implementation and operator acceptance tasks remain unchecked.

**Operator correction, 2026-10-21**: Map interchange is current **data-harvester → v65 dataset →
modern target serialization**, including **CASC reads**. [Spec 285](../285-v65-terrain-interchange/plan.md)
owns v65 native terrain, typed placement metadata and missing chunk handling. Object geometry/masks
are excluded. Old Converter CLI preservation/parity is superseded; original algorithms are reference
logic to rebase, with no confirmed current runtime baseline.

## Summary

Revive map, M2/MDX and WMO conversion as **three independently runnable CLI tools**, with three
separate tools under **Editor → Converters**. Shared libraries own conversion, client evidence,
placement re-itemization and reports. Preserve proven loading paths; old command aliases are not a deliverable.

Deliver a v65-native terrain/placement export first, then a bounded **1.60.1 → 0.5.3 terrain + objects** witness,
then broader map batches. Reuse WMO split/merge support before adding geometry reduction. Attempt
source-derived replacement objects explicitly; retain failures as failures, with inspectable output.

## Technical Context

**Language/Version**: C# / .NET 10; PowerShell 7 operator commands.  
**Dependencies**: Existing Core/Core.IO/Core.Editor, `MapConversionFormats`, ADT/WDT/M2/MDX/WMO
readers and writers, mounted data sources, `SharedUiWidgets`, and current Editor page injection.  
**Interchange dependency**: Spec 285 current MPQ/loose/CASC harvesting, Python-owned v65 Zarr and
typed reverse stream. No direct C# Zarr reader or obsolete Converter executable dependency.  
**Storage**: Loose client-content files plus small JSON requests/reports/ID mappings. No MPQ/CASC
output; any persistent project content cache follows existing Zarr v3 storage contracts.  
**Validation**: Focused behavior checks, real-file readback, then operator viewer/target-client
evidence. Build/test success is never target compatibility or visual acceptance.  
**Platform/type**: Desktop viewer plus three headless CLI hosts over library services.  
**Performance**: Cancellable bounded work, deterministic allocation independent of scheduling,
dependency conversion once per source-content/options/profile key. No invented speed/FPS target.  
**Scale**: Selected objects/tiles → one map → map batch after bounded route acceptance.  
**Constraints**: Reader/writer freeze; no new `ViewerApp`/`WorldScene` members; files below approximately
2,000 lines; configured client roots; operator owns heavy and visual runs.

## Operator Intent Check

| Operator request | Design response |
|---|---|
| Revitalize map and object converters | Rebase original logic on current harvester/library tooling with independent hosts. |
| Even 1.60.1 → 0.5.3 | Explicit target profiles and terrain/prop/WMO pilot; no success claim from an enum route. |
| Re-itemize each object's uniqueID | Deterministic placement graph allocation, reservations, reference rewriting and provenance. |
| Generate objects / remesh / regroup | Source-derived candidates and bounded mesh processing; reuse WMO regrouping first. |
| Separate CLI and Editor tools | Map, Model and WMO each have a CLI project, service and Editor tool. |
| Logical options / maybe emoji | Goal presets, progressive options, concise review, optional glyphs with text fallback. |
| Loaded and known-good clients in either slot | Mounted-source snapshot; saved-entry adapter; loaded-as-target and swap controls. |
| Recognize direction from versions/information | Source/target capability graph; version is supporting evidence. |
| Plan and tasks only | Documentation package and registry/continuity references; no executable code or client operations. |
| Dataset convergence; v65; keep placements, exclude geometry/masks | Spec 285 owns native terrain and exact placement metadata; separate asset tools own geometry. |
| CASC input | Current CASC catalog/FDID/companion reads enter v65 with product/build evidence. |

## Constitution Check

| Gate | Before research / after design |
|---|---|
| Repo independence / library first | PASS: proposed code stays inside `wow-viewer/`; thin CLI and Editor adapters. |
| Real-data evidence | PASS by design: checks, readback, viewer and target witness remain distinct; actual acceptance pending. |
| Client configuration | PASS: no portable hardcoded roots; configured roots/builds/hashes in validation receipts. |
| Container policy | PASS: loose client-content output only; no packaging or installation. |
| Proven base / frozen readers and Alpha writer | PASS: adapters above existing base; a failed generated file cannot authorize reader/renderer edits. |
| UI ownership / god-class freeze | PASS: existing Editor injection, owned services, UI inventory, no app/scene members. |
| Scope / receipts | PASS: operator request is the contract; archived backlog supplies evidence only. |
| Operator-owned work | PASS: heavy conversion, GPU/training and real-client proof prepared for operator execution. |

Dataset streaming and Python storage ownership apply through Spec 285; training is not involved.
No exceptions requested. Recheck gates
after each future phase; do not turn a failed gate into an implicit exception.

## Architecture Decisions

### AD-1 — Separate domain services and hosts; shared primitives

Core owns immutable client evidence, profiles, route decisions, requests, reports and allocation
models. Core.IO owns probes, Map/Model/Wmo services, geometry processing, dependencies and file
emission. Core.Editor owns request/job coordination without ImGui. Viewer adapters read the
current session and saved settings; neither frontend duplicates conversion logic.

New thin hosts: `WowViewer.Tool.MapConvert`, `WowViewer.Tool.ModelConvert`,
`WowViewer.Tool.WmoConvert`. The old Converter project is source reference only, not a dependency,
mandatory adapter target or parity baseline. Current data-harvester tooling owns source ingestion,
Python v65 read/write and typed interchange; target domain services own serialization. Factor
private current harvesting orchestration into the reusable service described by Spec 285.
Map may compose Model/Wmo APIs for dependency
closure; it does not invoke their executables. Standalone object conversion requires no map session.

Renderer-used duplicate converters remain untouched. Convergence here applies to output conversion,
not base loading/rendering. Splitting format-reader ownership or sweeping old code is out of scope.

### AD-2 — Client roles are chosen; route direction is inferred

Default source = actual mounted client, not `GetActiveGamePath()`'s remembered-folder fallback.
Offer **Use loaded as target** and **Swap**. Either slot can select an existing saved known-good
client or an explicit root. Never silently swap slots based on version magnitude.

Adapt saved `{Name, Path, BuildVersion}` records without calling them tested conversions. Probe
selected clients on demand: availability, product/build metadata, mounted base/overlay identity,
and representative file signatures. Capture metadata hashes to detect changed clients; complete
dependency hashes when selected content is planned. No broad library scan is needed.

Resolve source signatures + target profile + converter capabilities. `1.60.1` does not imply older
formats than `3.3.5`; numeric order is not format chronology. Short versions/folder names are hints.
Conflicts expose **Choose profile** with evidence. Overrides cannot enable nonexistent writers.
Preserve the global loader's explicit version-selection behavior.

### AD-3 — Preview freezes a concrete request

`Inspect/Plan → Review → Execute → Validate → Result`. Planning resolves dependencies, transforms,
losses, paths and ID namespace before publishing client-content files. CLI and Editor execute the
same request and recheck fingerprints before writing. Stage under a run-scoped output directory;
publish a completed manifest only after validation. Cancellation/failure leaves an explicitly
partial report; never modify source/target installations or overwrite prior accepted output.

Offer **Target-compatible** (recommended), **Preserve features** (blocks unrepresentable losses),
and **Custom** presets. Retain source geometry as far as target limits permit. Generation is off
until explicitly selected. Loss review is concise, with expandable detail and explicit acceptance.

### AD-4 — Re-itemization consumes typed v65 placement metadata

Read exact original placement/reference records from v65, without changing stored source IDs.
Do not route them through old float32 placement projections or old-ID deduplication. Identity uses source
fingerprint, map, kind, original ID, canonical asset, exact transform and source-record provenance.
Identical cross-tile references share an object only with matching identity evidence. Different
transforms/assets sharing an old ID remain distinct. Do not deduplicate by proximity.

Allocate sequential IDs in stable canonical order across M2/WMO instances in the declared output
map namespace. Retain raw original IDs in provenance. Respect target-measured field limits and
current signed writer DTO ranges; overflow blocks the run. Reserve retained output IDs. Partial-map
patches inventory retained placements or require a complete reservation manifest; a tile-local
high-water mark is insufficient.

Emit source-instance → target-instance records, never a dictionary keyed only by old ID. Repeated
references share one new ID. Rebuild MDDF/MODF, target-appropriate MCRF and preserved Alpha uniqueID
arrays before frozen writers run. MCRF can contain placement indices: never substitute IDs into
index fields. WMO group and MODD-local indices remain separate namespaces. Root-group splitting
alone allocates no extra placement IDs; real one-to-many replacement placements get child IDs.

### AD-5 — Modern maps normalize into existing writer inputs

Resolve the selected source (MPQ/loose/CASC) to a matching finalized v65 dataset, or prepare bounded
ingestion using the current data-harvester tooling. Validate source revision/product/build, store
schema/layout/digest and native field/placement coverage. Python reads the store and emits typed
records; C# consumes them through the Spec 285 reverse stream. Map export does not re-open source
terrain as a fallback. CASC file/companion reads and FileDataID resolution happen during ingestion.

Materialize canonical writer inputs from native v65 chunk samples, full source layers, exact holes,
liquid records, metadata and placements. Port useful original coordinate/layer/reference algorithms
onto this adapter, using current Core.IO writers including frozen `AlphaWdtWriter`. An intermediate
LK document is allowed where it preserves semantics, but neither it nor old CLI compilation proves
Alpha compatibility. New target transforms are checked against actual source/native records.

Record preserve/transform/reduce/block per feature: excess layers, unknown alpha codec, height
blending, high-res holes, liquids, MCCV/MCSH, area/flag differences, placement flags and FileDataIDs.
Unknown codecs block conversion; missing source semantics belong in Spec 285's dataset handlers,
not another source decoder in the map tool. Opaque bytes alone are not decoded interchange.

For decoded over-budget layers under a supported composition model, use the opacity report:
keep base, select remaining layers by integrated effective contribution with stable ties, assign
discarded effective contribution to base, then re-encode target sequential alpha. Never normalize
raw MCAL channels. Report layer maps, coverage/material/quantization changes. Preserve-features blocks
reduction. Verify masks/boundaries on real data; no shared renderer/alpha-reader change is included.

Read terrain material identities and typed placement asset references from v65. Resolve actual
object geometry and its companions through separately configured MPQ/loose/CASC asset sources;
v65 contains no WMO/M2 mesh payload or object masks. Generate deterministic
target paths for unnamed assets, detect normalized-path collisions, and retain translation records.
Convert WMO doodad/model dependencies recursively, handling cycles. Target asset reuse requires
identity/compatibility evidence, not basename similarity. State whether dependencies are bundled
or supplied by the target; unresolved required assets block full conversion acceptance.

### AD-6 — Model conversion reports actual fidelity

Reuse Core.IO's M2/MDX path with explicit profiles. Address output limitations through conversion
documents/writers or bounded converter changes, keeping readers frozen. Account for per-material
geometry, normals, UVs, winding, pivot, bounds, textures, replaceables and companions. Audit skinning
and tracks separately: current `VERS=1300` output with empty skinning tables does not prove a valid
animated Alpha model.

Static props are the first accepted slice. Unsupported animation/skinning is blocked or uses an
explicit **Static replacement** with pose/losses recorded. Existing supported reverse routes remain;
modern M2 emission is not promised by modern M2 reading. No full retargeting or new FX project.

### AD-7 — WMO topology changes retain semantic accountability

Start with `Core.IO/Wmo/WmoV17ToV14Converter`: splitting, compaction, spatial merge and portal
remapping already exist. Its 384-group/`0xBFFF` vertex limits are current code constraints, not
universal build facts; confirm against the selected target.

Expose budgets and source-group → output-group mappings. Prefer compaction/lossless partitioning,
then compatible spatial/material regrouping, then explicit geometry reduction if needed. Validate
root/group counts, MOBA batches, portal ranges/graph, BSP/collision, bounds, liquids, lighting,
flags, doodad sets/references. Existing dropped/approximated semantics are reported or blocked.
Converting an interior to an exterior static shell requires an explicit lossy choice.

### AD-8 — Generation is an explicit experimental lane

Extract available source visual geometry into an inspectable mesh document. Preserve connected
components, UV/material boundaries, silhouette and collision requirements through bounded cleanup,
partitioning and, where required, deterministic error-bounded simplification. Record before/after
counts, bounds, geometric error, seam changes and semantic dispositions.

No general visual remesher was found; algorithm/dependency selection is an implementation feasibility
task. Prefer a contained CPU approach; any dependency must fit licensing/project constraints.
Operator owns expensive runs. Initial candidates are static MDX props and WMO root/groups derived
from source geometry. Legacy geometry-to-WMO code is read-only reference, not production support.

Proxy cubes, collision surfaces or silently blank materials cannot pass visual-object acceptance.
Missing visual geometry is a reported inability to generate. No unseen geometry, AI training or
PM4 semantic inference is invented. Candidates require lineage and explicit review before substitution.

### AD-9 — One Editor home, three clear workflows

Keep `EditorConvertersPage` and its injected content seam. Introduce small Map/Model/Wmo tool
services sharing picker/review/progress widgets. Do not add state to `ConverterDialogsService` or
the app/scene god-classes. Modify existing composition/delegation only. Honor Spec 283's Editor
preset ownership when it lands; this spec does not implement that shell.

```text
Editor > Converters    [Map] [Model] [WMO]
Source  [Loaded: 1.60.1 …]     [Swap]
Target  [Known good: 0.5.3 …]  [Use loaded as target]
Route   Modern assets → Alpha 0.5.3   ·   Needs conversion
Content [Loaded map / selected object]    Output [Project folder…]
Preset  [Target-compatible]              [Advanced ▸]
[Review] → 3 losses · 0 blocked → [Convert] → [Open output] [Details]
```

Advanced options: map scope/dependencies/ID range/reservations; model static/material handling;
WMO regrouping/budgets; explicit generation. Hide irrelevant controls. Optional `🗺 Map`, `🧩 Model`,
`🏠 WMO`, `⚠ Review`, `✓ Ready` accompany text and verified glyph fallbacks. Do not add a font
pipeline solely for emoji. Register controls and regenerate UI reference documentation.

### AD-10 — Refresh output around the viewer's canonical map writers

The [writer reuse contract](contracts/map-writer-reuse.md) records the actual New Map path:
`TemplatedTerrainGenerator → LkAdtWriter + LkWdtWriter + WdlWriter`. Use a small Core.IO
`MapTargetMaterializer` to select/invoke these canonical writers from validated target documents.
`V65TerrainExportAdapter` supplies native conversion data; `MapConversionService` coordinates the
request/dependencies/IDs/publication. Do not pass converted source terrain through a procedural
template or add a parallel serializer. Existing UI new-map orchestration can delegate to the same
facade behavior-preservingly, with the same generation/defaults/load callback.

`AdtTerrainWriter` is an existing-template MCVT/MCNR patcher, not a fresh-map writer. Current LK
output defaults some MTXF/MCSE/placement metadata; WDT global metadata and WDL holes/native fidelity
also have limits. Audit every source field against target DTO and actual writer output. Add bounded
opt-in field/adapter support where required, or block/report reductions; retain original v65 data.
Existing new-map defaults/readers and frozen Alpha writer stay intact.

Use Spec 285's per-layer opacity reports for source and independently read target output.

Source-report readiness (Spec 285 T050–T055/T057) is a prerequisite for comparison and contribution-
based reduction, not for starting the target adapter. Source reports do not require target output;
actual output/readback is the later joint comparison gate.

Preserve one disposition per declared source layer, including repeated textures/zero/occluded layers; never
normalize raw alpha channels. Known target layer reduction operates on effective contributions
under a supported composition model, re-encodes the target alpha contract and reports quantization/
spatial/metadata differences. Unknown blend behavior remains a blocker or explicit reference view.

## Project Structure

All source paths are **future work**, relative to `wow-viewer/`. Existing files are touched only
at named seams. See [contracts/conversion-tools.md](contracts/conversion-tools.md).

```text
specs/284-map-object-converter-revitalization/
  spec.md  plan.md  tasks.md  research.md  data-model.md  quickstart.md
  contracts/conversion-tools.md  evidence/planning-receipt.md
src/core/WowViewer.Core/Conversion/
  ClientEvidence.cs  ConversionProfiles.cs  ConversionRequest.cs  ConversionReport.cs
  PlacementIdentity.cs  PlacementIdMap.cs  MeshCandidate.cs
src/core/WowViewer.Core.IO/Conversion/
  ClientCapabilityProbe.cs  ConversionRouteResolver.cs  ConversionDependencyResolver.cs
  PlacementIdentityGraphBuilder.cs  PlacementIdAllocator.cs  PlacementReferenceRewriter.cs
  ConversionOutputPublisher.cs
  Maps/MapConversionService.cs  Maps/V65TerrainExportAdapter.cs  Maps/MapTargetMaterializer.cs
  Maps/LegacyLayerReducer.cs
  Models/ModelConversionService.cs  Models/TargetModelWriter.cs
  Wmo/WmoConversionService.cs  Wmo/WmoTopologyValidator.cs
  Geometry/ConversionMesh.cs  Geometry/TargetMeshProcessor.cs
  Geometry/ObjectCandidateGenerator.cs  Geometry/TargetWmoMeshWriter.cs
src/core/WowViewer.Core.Editor/Conversion/
  ConversionJobCoordinator.cs  ConversionClientSelection.cs
tools/map-convert/WowViewer.Tool.MapConvert/       # separate CLI
tools/model-convert/WowViewer.Tool.ModelConvert/   # separate CLI
tools/wmo-convert/WowViewer.Tool.WmoConvert/       # separate CLI
src/core/WowViewer.Core.Editor/Conversion/V65DatasetBridge.cs  # Spec 285 Python/store bridge
src/viewer/WoWViewer/Workbench/Services/Converters/
  ConversionClientSelectionAdapter.cs  ConverterToolWidgets.cs
  MapConverterToolService.cs  ModelConverterToolService.cs  WmoConverterToolService.cs
src/viewer/WoWViewer/Workbench/Pages/EditorConvertersPage.cs       # existing home
src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.Content.cs
src/viewer/WoWViewer/ViewerApp_Editor.cs                         # existing injection only
eng/ui/ui-controls.json  docs/WoWViewer/UI-REFERENCE.md          # registry/generated docs
tests/WowViewer.Core.Tests/Conversion/
tests/WowViewer.Core.Editor.Tests/Conversion/
```

Files may be subdivided within the same ownership boundaries to honor the file budget. No new
shared renderer behavior, `ViewerApp` fields/partial files or scene members are authorized.

## Phase Roadmap and Gates

| Phase | Outcome | Gate |
|---|---|---|
| 1–2 | Algorithm lineage, v65 prerequisites, contracts and hosts | Old Converter assumptions removed; Spec 285 contracts/gates linked. |
| 3 / US1 | Client evidence + route preview | Real client evidence, both slot orders, conflict/unavailable cases. |
| 4 / US2 | Deterministic re-itemization | Typed v65 placement mapping/readback; collision, precision, reservation and partial-map cases. |
| 5 / US3-A | v65 target adapter | Source-detached native dataset export; original algorithm semantics checked on real data. |
| 6 / US3-B | Modern CASC→v65→Alpha staging | Bounded real CASC dataset readback; missing chunk/object blockers explicit. |
| 7 / US4 | Standalone model output | Real static prop; explicit animated limitations; independent CLI. |
| 8 / US5 | Standalone WMO output | Real multi-group topology checks; dependencies can close map pilot. |
| 9 / US6 | Editor integration | CLI/Editor parity and operator workflow; structurally complete pilot, SC-003 target witness pending. |
| 10 / US7 | Generation attempt | Geometry/lineage metrics and explicit candidate review. |
| 11 | Target witness and scale | Operator target-client acceptance, then whole-map/batch run. |

Implement one validated phase at a time. A phase stays open while its gate is open; independent
preparation retains a partial status. MVP = finalized native v65 through the separate Map tool,
client preview and deterministic IDs, not old CLI parity or a claimed full modern route.

## Verification and Handoff

Spec 285 additionally owns durable unknown/native metadata, terrain-cell layered appearance
indexes and all referenced source/PNG texture resources for future viewers/models. Full appearance
export acceptance requires those coverage gates; adapter development may start at native/placement/
transport readiness. Python supplies selected appearance/table/resource records through typed
streams; C# never reopens a client for terrain metadata omitted by an incomplete dataset. Target
texture/material conversion reports losses without replacing retained source resources.

[quickstart.md](quickstart.md) contains future commands and the witness matrix. Every phase receipt
records changed paths, exact command/exit status, source/target build/root/fingerprints, output
hashes and criterion → real evidence. Track planned, implemented, readback-validated,
viewer-inspected and target-client-accepted independently. Unknown routes remain unknown.

Final pilot: real CASC-backed terrain through v65, a converted static prop, a converted WMO, dependency closure and ID/
reference validation. Viewer appearance is not original 0.5.3 client acceptance. Prepare an operator
witness using an already-supported loose-content workflow; do not invent packaging/install support
or write MPQ. If unavailable, record the gap and leave SC-003 open. Never repair base rendering or
the frozen writer merely to pass generated data.

## Complexity Tracking

Three small CLI hosts satisfy the explicit separate-tools requirement; shared services avoid three
implementations. No other complexity exception or unrelated backlog adoption.
