# Research: Converter Revitalization

**Date**: 2026-10-21. Source inspection only; no conversion, client scan, build or runtime proof.
Paths are relative to `wow-viewer/` unless stated otherwise.

## R-1 — Rebase on current harvesting; historical commands are not a functional baseline

**Decision**: Separate services/CLI hosts; map export consumes Spec 285 v65 through current harvesting tooling.
**Evidence**: `tools/converter/WowViewer.Tool.Converter/Program.cs` dispatches all three domains in
one executable. Adjacent command classes provide historical algorithm references, not confirmed
working entry points. The operator has not tested that project since approximately April.
`Core/Maps/MapConversionFormat.cs`
exposes Alpha/LK writers and disables native MoP split writing. `ModernToLegacyMapConversionService`
is absent; Epic 248 F-03 adopts archived 243's design and records implementation as absent.
**Rationale**: File presence/compilation does not establish useful current behavior. Reuse algorithms
behind the current data-harvester and shared libraries, with fresh real-data gates.
**Alternatives**: Mandatory old CLI adapters/parity and direct-client map conversion bypassing v65
are superseded by the operator correction; current harvest tooling is the interchange owner.

## R-2 — Regrouping exists; general remeshing remains work

**Decision**: Reuse `src/core/WowViewer.Core.IO/Wmo/WmoV17ToV14Converter.cs` first.
**Evidence**: `SplitLegacyGroupForLegacyBatchIndexLimit`, `CreateSplitLegacyGroup`,
`ExpandPortalLayout`, `MergeOverflowGroups`, `MergeLegacyGroups` implement splitting, compaction,
portal remapping and spatial grouping. Code uses 384 groups and `0xBFFF` vertices/group.
`tests/WowViewer.Core.Tests/WmoV17ToV14ConverterTests.cs` contains synthetic checks, not fresh
real-client proof. Another converter in `Core.IO/Converters/` serves loading and stays untouched.
**Rationale**: Real regrouping support exists; no general visual-object remesher was found in the
searched core/tools. PM4 planar simplification is not equivalent visual-mesh support.
**Alternatives**: Global converter unification risks base behavior; blind decimation hides semantics.

## R-3 — Model output has fidelity limitations

**Decision**: Static props first; inspect skinning/animation independently.
**Evidence**: `src/core/WowViewer.Core.IO/M2/M2ToMdxConverter.cs` emits `MDLX`, `VERS=1300`.
`WriteGeos` uses one geoset, first UV set, zero GNDX and empty MTGC/MATS/BIDX/BWGT. A second
implementation at `src/viewer/WoWViewer/Terrain/Transfer/M2ToMdxConverter.cs` is outside output ownership.
**Rationale**: Signature/bone chunks do not establish correct animated Alpha output.
**Alternatives**: Silent freezing hides loss; full retargeting exceeds this bounded revival.

## R-4 — IDs require a request-wide identity graph

**Decision**: Normalize instances, reserve IDs, allocate once, rebuild references.
**Evidence**: `Core.IO/Maps/LkToAlphaConverter.cs` preserves IDs and maps chunk references using
`MapDoodadRefsToUniqueIds` / `MapWorldModelRefsToUniqueIds`. `AlphaWdtWriter.cs` reconstructs index
references from preserved uniqueID arrays. Existing CLI commands deduplicate by old ID;
`AdtPlacementEditor.cs` uses a tile-local high-water mark. None is a deterministic map allocator.
**Rationale**: MDDF/MODF-only rewriting leaves reference hazards; old-ID dedup can discard instances.
**Alternatives**: Random or per-tile IDs fail reproducibility and namespace safety.

## R-5 — Adapt existing client knowledge

**Decision**: Use actual mounts and existing saved entries, probing selected clients only.
**Evidence**: `ViewerApp.cs` defines `KnownGoodClientPath {Name, Path, BuildVersion}`.
`Workbench/Services/ClientDialogs/ClientDialogsService.ClientDialogs.cs` saves loaded MPQ bases
through `SaveCurrentGameFolderAsKnownGoodBase`; `Settings/ViewerSettingsService.cs` persists them.
They lack content fingerprints/conversion receipts. `DataSourceSession/DataSourceSessionService.cs`
can return the last folder from `GetActiveGamePath`. `Core/Maps/ClientBuildCatalog.cs` and
`Terrain/BuildVersionCatalog.cs` provide hints, not content-wide compatibility evidence.
**Rationale**: Known-good selections exist, but saved versions do not establish writable routes.
**Alternatives**: A broad discovery scan is unnecessary; recent folders are not known-good clients.

## R-6 — Reuse the Editor seam

**Decision**: Inject three owned tools through Editor → Converters.
**Evidence**: `Workbench/Pages/EditorConvertersPage.cs` accepts a draw delegate;
`Services/MainMenu/MainMenuBarService.Content.cs:DrawConvertersContent` provides Map/WMO launchers.
`Services/Converters/ConverterDialogsService.cs` owns current dialogs and invokes the old WMO CLI.
`ViewerApp_Editor.cs` contains composition. Spec 283 plans Editor-only conversion ownership.
**Rationale**: Shared APIs support parity without new app/scene state. Emoji coverage is unverified;
plain-text fallback is required. Preserve current UI WIP.
**Alternatives**: New global menus or disconnected frontends conflict with one-home governance.

## R-7 — Generation produces candidates, not debug stand-ins

**Decision**: Use source visual geometry, retain lineage, require review.
**Evidence**: Outside `wow-viewer/`, read-only references
`gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Analysis/Pm4WmoWriter.cs` and
`Generation/Pm4DebugWmoWriter.cs` emit geometry-based WMOs. The debug path may substitute a cube.
Rosetta/PM4 placement generators do not establish visual-mesh generation support.
**Rationale**: These are feasibility references, not accepted production generation. No runtime
dependency may point outside `wow-viewer/`; no legacy reference edits or copied format readers.
**Alternatives**: AI generation, proxy cubes and collision-only surfaces are not assumed.

## Remaining engineering proof and resolution path

No user clarification is needed to author this plan. Implementation must measure exact source/
target profiles, writer limits, dependencies, layer encoding, model fidelity and mesh error on an
operator-selected bounded corpus. Setup tasks establish that corpus; US3–US7 own the checks.
Missing support remains blocked or explicitly lossy. This is an implementation feasibility gate,
not a claim that every requested conversion is already possible.

## R-8 — v65 decoded convergence and CASC input (operator correction)

**Decision**: [Spec 285 research](../285-v65-terrain-interchange/research.md) and
[coverage contract](../285-v65-terrain-interchange/contracts/terrain-interchange.md) own the modern
harvester/store gaps. v65 retains typed placement metadata but excludes mesh payloads/object masks.
**Evidence**: Current harvest CLI opens MPQ/CASC catalogs and emits V22 into the Python v60 builder;
native fields are omitted/reduced, metadata can be lost and placement IDs pass through float32.
**Rationale**: Extend extraction/transport/storage once, then export either target from decoded data.
Python retains Zarr ownership; C# consumes a versioned reverse stream. CASC local reads and FDID/
companion identity require real source evidence. No CASC output or implicit CDN retrieval.

## R-9 — Viewer New Map writers are the canonical output starting point

**Decision**: Shared target-document/materialization services call the same current library writers
used by viewer New Map creation; source and independently read output opacity reports expose losses.
**Evidence**: `Workbench/Services/NewMapCreatorService.cs` calls `TemplatedTerrainGenerator.GenerateMap`
then `LkAdtWriter.Write`, `LkWdtWriter.Write` and `WdlWriter.Write`. Harvest relief-map, Inspect map
generation and `DatToLkAdtFolderExporter` also use these libraries. `AdtTerrainWriter.ApplyHeightmap`
only patches MCVT/MCNR in existing bytes; it does not create a new map or serialize all metadata.

`LkAdtWriter` currently emits default MTXF/MCSE/MDDF flags; WDT global metadata and WDL holes/native
heights also have limited contracts. These are field-capability gaps, not proof of preserved source
data. Native absolute Z must map deliberately to relative heights/base in LK documents.
**Rationale**: Reuse canonical serialization, port v65 data explicitly, and audit DTO/writer/readback.
**Alternatives rejected**: Parallel serializers, fresh procedural templates replacing source terrain,
height patching presented as full conversion, or compilation presented as actual map acceptance.
