# Feature Specification: Map and Object Converter Revitalization

**Feature ID**: `284-map-object-converter-revitalization`  
**Created**: 2026-10-21  
**Status**: Draft planning package; implementation belongs to a later chat  
**Working branch**: `280-ui-rework` retained; no branch switch requested  
**Owner**: Epic 248 (conversion), with Editor integration under Epics 250/251

## Operator intent and scope authority

The operator requested:

> "revitalize the map and object converters, supporting even 1.60.1 -> 0.5.3 conversion"
>
> "re-itemization of the uniqueID value for each object"
>
> "an attempt at generating objects" involving "re-meshing, WMO re-grouping"
>
> "converters to all be separate - continue to be cli tools, but also as part of the viewer's Editor tools"
>
> "options that make logical sense, maybe even use emoji's to help guide them"
>
> "Use currently loaded client by default as the source, or target" and "previously known good clients"
>
> "automatically being able to recognize which direction we are going" from versions or client information
>
> "only write a speckit plan, with tasks. I will handle implementation in another chat."
> "with regard to speckit plan 284 - let's make sure we also handle the original WDT and WDL data, if it exists, and put that into some metadata too. This will help us later on with regard to reconstructing the output wdt or wdl, and ensuing they are all in the right formats with the right flags set. We didn't have WDT flags being supported before, but now we do. You can also be sure that our reading/parsing is right by comparing against the 4.3.4 decomp project, which is pretty accurate, it's not 100% perfect, but a better-than-pseudocode source.
Once we figure out the ui, this is the next big important thing to deal with."

This specification records that request so the plan and tasks have an explicit contract. Architectural
choices in the plan implement these requirements; unrelated epic backlog is not adopted. Existing
tools establish a starting point, not proof of a complete 1.60.1 → 0.5.3 conversion.

### Operator correction — 2026-10-21

Use the existing **data-harvester → v65 decoded dataset → modern target exporter** as the map
interchange route, including **CASC input reads**. [Spec 285](../285-v65-terrain-interchange/spec.md)
owns missing chunk coverage and the v65 schema. The operator explicitly retains placement metadata
in v65 and excludes object geometry/masks. Separate object tools resolve actual assets outside
the dataset. Re-itemization consumes typed stored placements, not raster labels or float-cast IDs.

The old `WowViewer.Tool.Converter` has no confirmed functional baseline in the current codebase;
the operator last tested it around April. Retain useful original conversion algorithms as source
references and rebase them onto current libraries. Legacy CLI parity/wrappers are **not required**
and the old executable must not be a runtime dependency. This supersedes earlier planning assumptions.

### Operator addition — writer refresh and per-layer opacity, 2026-10-21

Reuse the current library writers used by the viewer's New Map creation tooling for refreshed
conversion output. Inventory their actual data contracts and metadata limits; no second serializer
or obsolete CLI dependency. Spec 285 adds retained MCAL/MCLY metadata/opacity for every layer;
target output must be independently read back and compared using explicit layer mappings.

## User Scenarios & Testing

### User Story 1 — Choose clients and understand the route (Priority: P1)

An operator opens a converter with the actual loaded client preselected as source. They can instead
use it as target, select a saved known-good client in either slot, or explicitly choose a client.
The tool determines the supported conversion route and explains any conflict or missing capability.

**Independent Test**: Choose modern 1.60.1 and Alpha 0.5.3 clients in both slot orders; compare the
detected route with their real content formats. An unavailable writer must remain unavailable.

**Acceptance Scenarios**:
1. **Given** a mounted client, **when** a converter opens, **then** the source shows that client and
   its identity evidence; a remembered but unmounted folder is not labelled loaded.
2. **Given** saved known-good clients, **when** one is selected, **then** current availability and
   format evidence are checked; the saved label alone does not prove conversion compatibility.
3. **Given** selected source and target, **when** their evidence is sufficient, **then** direction
   is inferred without manually choosing an unrelated converter mode; ambiguous evidence requests
   a specific profile choice and preserves the reason.

### User Story 2 — Re-itemize object uniqueIDs reliably (Priority: P1)

The operator receives converted placements with a collision-free uniqueID assignment and an
auditable mapping back to every source instance, including converted or generated replacements.

**Independent Test**: Convert a real selection containing M2 and WMO placements, repeated tile
references, and conflicting original IDs; inspect output IDs and references against the mapping.

**Acceptance Scenarios**:
1. **Given** distinct output instances, **when** re-itemization runs, **then** each has a unique ID
   in the declared output namespace and all references to one instance agree.
2. **Given** identical inputs, reserved IDs and options, **when** conversion repeats, **then** the
   same IDs and mapping result regardless of enumeration order.
3. **Given** existing output placements or a range that is too small, **when** IDs are allocated,
   **then** reservations are respected or conversion stops before publishing incomplete output.

### User Story 3 — Convert maps through a separate CLI tool (Priority: P1)

The operator can run map conversion independently from finalized v65, preserving useful conversion logic and progressing
to a complete real-data 1.60.1 → 0.5.3 map slice with converted object dependencies.

**Independent Test**: Convert a bounded real map selection with terrain, M2 and WMO placements;
read back all produced files and have the operator inspect them with the target environment.

**Acceptance Scenarios**:
1. **Given** a finalized v65 selection, **when** the separate map CLI exports it, **then** it uses
   the current harvesting/interchange libraries and verified original transformation logic,
   without re-decoding source terrain or running the obsolete converter executable.
2. **Given** a 1.60.1 selection, **when** Alpha 0.5.3 is the target, **then** the output uses that
   target's map and object formats; copying modern assets into renamed files is insufficient.
3. **Given** unsupported terrain or asset features, **when** conversion is previewed, **then**
   preserved, transformed, reduced and blocked features are reported before execution.

### User Story 4 — Convert models through a separate CLI tool (Priority: P1)

The operator can convert M2/MDX objects independently with target-specific feature handling and
complete dependency reporting, without converting a map or WMO.

**Independent Test**: Convert one real static prop and one real animated model separately; compare
geometry, materials, textures, and supported animation behavior with the source and target.

**Acceptance Scenarios**:
1. **Given** a static prop, **when** a supported conversion runs, **then** its geometry, transform
   basis, materials and required textures remain accounted for.
2. **Given** an animated or skinned model, **when** the route cannot preserve its behavior, **then**
   it is reported as unsupported or an explicitly selected static replacement, never full fidelity.

### User Story 5 — Convert and regroup WMOs through a separate CLI tool (Priority: P1)

The operator can convert a WMO independently and inspect the consequences of group splitting,
merging, remeshing and target limits without running map conversion.

**Independent Test**: Convert real multi-group and oversized WMOs; inspect group membership,
material assignments, bounds, doodad sets, portals, collision and target load behavior.

**Acceptance Scenarios**:
1. **Given** an oversized WMO, **when** regrouping is required, **then** the plan shows target
   limits, the resulting group mapping, and any topology or semantic losses.
2. **Given** interior, portal, liquid or collision data, **when** topology changes, **then** it is
   preserved, rebuilt with evidence, explicitly reduced, or blocked; it is not silently discarded.

### User Story 6 — Use clear, separate Editor tools (Priority: P1)

Editor → Converters presents separate Map, Model and WMO tools sharing the same behavior as their
CLIs. The operator chooses clients, content, a clear preset, reviews consequences, runs, and opens output.

**Independent Test**: Preview and run the same bounded request in Editor and CLI; compare the
resolved route, options, ID mapping, output inventory and diagnostics.

**Acceptance Scenarios**:
1. **Given** the Editor preset, **when** Converters opens, **then** each converter has a separate
   entry and only relevant options; conversion actions have no second execution home elsewhere.
2. **Given** a conversion preview, **when** the operator reviews it, **then** source → target,
   losses, blocked items and output location are clear in plain text; optional emoji have fallbacks.
3. **Given** a running conversion, **when** the operator cancels, **then** the UI remains usable
   and partial output is clearly separated from a validated result.

### User Story 7 — Attempt target-compatible object generation (Priority: P2)

For objects that cannot be directly converted, the operator can explicitly attempt a replacement
derived from available source geometry, using remeshing and regrouping as needed, and review it.

**Independent Test**: Generate one real source-derived static replacement and one WMO candidate;
retain before/after geometry evidence and accept or reject each candidate independently.

**Acceptance Scenarios**:
1. **Given** a conversion failure with usable source geometry, **when** generation is selected,
   **then** the candidate records its source, geometry changes, dependencies and measured limits.
2. **Given** a generated candidate, **when** it fails target checks or operator review, **then**
   it is not silently substituted or counted as a successful conversion.

### Edge Cases

- Version labels conflict with actual content; a short version has multiple build candidates.
- A saved client is missing, changed, or only suitable for a different converter.
- No client is mounted, or the same client is selected in both slots.
- Different instances reuse an original ID; one large WMO is referenced on multiple tiles.
- A tile subset references placements outside the subset or retains unconverted neighboring data.
- FileDataIDs have no known path; dependency paths collide after target normalization.
- WMO splitting exceeds another target limit; regrouping invalidates portals or collision.
- An animated model cannot be represented; unsupported geometry has no usable source mesh.
- A required modern terrain feature cannot be decoded or represented in Alpha.
- Output already exists, cancellation occurs, or source evidence changes after preview.

## Requirements

### Functional Requirements

- **FR-001**: Provide independent map, model and WMO CLI tools and separate Editor tools backed by
  the same conversion behavior; rebase original conversion logic onto current tooling without an
  old Converter project/runtime dependency or mandatory legacy command wrappers.
- **FR-002**: Support actual loaded-client and saved known-good selections in either slot, with
  loaded-as-source default and an explicit loaded-as-target choice.
- **FR-003**: Infer direction using client identity and actual format capabilities; expose conflicts,
  unavailable writers and explicit overrides instead of guessing from version order alone.
- **FR-004**: Re-itemize each distinct output object placement with deterministic, collision-free
  IDs; preserve original identity and rebuild all affected references and allocation provenance.
- **FR-005**: Plan and implement a bounded 1.60.1 → 0.5.3 map-and-object route, including dependency
  resolution and target format validation, then scale the proven route to map batches.
- **FR-006**: Handle model conversion and WMO regrouping/remeshing through target-specific
  capabilities, with explicit feature disposition and no silent claims of unsupported fidelity.
- **FR-007**: Attempt source-derived object generation as an explicit, reviewable fallback, retaining
  source geometry lineage, target checks and accept/reject results.
- **FR-008**: Offer sensible presets and relevant options, progressive detail, plain-language status,
  and optional emoji with readable fallbacks; keep conversion execution inside Editor.
- **FR-009**: Preview consequences, report losses and failures, support cancellation, and write
  validated loose-file results to project/operator output rather than changing a client installation.
- **FR-010**: Record real-data evidence for each supported route separately from build/test evidence;
  reserve heavy runs and real-client visual proof for the operator.
- **FR-011**: Use Spec 285's v65 dataset as the decoded terrain/placement interchange point for
  map export. Use existing harvester tooling for ingestion, including CASC source reads; add
  missing required chunk handling there. No silent direct-client fallback for incomplete datasets.
- **FR-012**: Read exact original IDs/transforms/asset references and chunk relationships from
  v65 placement metadata. Object meshes/masks stay outside v65 and use separate asset conversion.
- **FR-013**: Consume Spec 285's retained appearance/resource reference, including native layer/
  cell/material metadata, original textures and PNG derivations; preserve unknown source evidence
  and report per-target reductions without modifying the source dataset.
- **FR-014**: Route refreshed map materialization through the actual canonical writers used by
  viewer New Map creation; expose their field capabilities, defaults/reductions and target modes.
  Compare source and output per-layer opacity/metadata through real-file readback, without
  treating a successful write or a height-only patcher as full map preservation.

### Key Entities

- **Client selection**: actual mounted or saved client, identity evidence and role in this request.
- **Route/capability**: converter kind, source and target profiles, supported transformations and gaps.
- **Placement identity map**: source instance identity, output instance identity, assigned uniqueID.
- **Conversion request/result**: inputs, options, output namespace, losses, dependencies and evidence.
- **Generated candidate**: source-derived replacement with geometry lineage and review state.

## Success Criteria

- **SC-001**: Both orientations of a real 1.60.1/0.5.3 pair are identified correctly; an unavailable
  reverse writer is shown as unavailable rather than producing mislabeled output. (US1)
- **SC-002**: Every output placement has an allocation record, zero collisions between distinct
  instances, and zero dangling affected references; reruns produce identical mappings. (US2)
- **SC-003**: A bounded real 1.60.1 map selection produces Alpha terrain plus at least one M2-derived
  prop and one WMO, with every dependency resolved or explicitly blocking completion. The operator
  confirms target loading and appearance; until then this criterion remains open. (US3–5)
- **SC-004**: Map, Model and WMO tools each run independently through their CLI and Editor entry;
  equivalent requests yield the same route, mapping, losses and content inventory. (US3–6)
- **SC-005**: Every changed/generated object has measured geometry/semantic disposition and retained
  lineage; at least one static replacement and one WMO candidate receive explicit review. A failed
  candidate is an evidenced attempt, not accepted generated-object support. (US5/7)
- **SC-006**: Editor operators can follow Select → Options → Review → Run → Result with separate
  converter entries and readable status when emoji are absent; proof is an operator walkthrough. (US6)
- **SC-007**: Rebased routes retain verified original transformation semantics on real inputs; no source/target client
  installation or protected reader/writer is modified to make a new route pass. (All stories)
- **SC-008**: Real MPQ/loose and CASC sources can converge through v65; finalized terrain/placement
  export works without the source terrain mounted or the old Converter executable, with all
  required source metadata and target losses accounted for. (US1–3; Spec 285)
- **SC-009**: New-map control and v65 export use the same canonical ADT/WDT/WDL library writers.
  Actual readback accounts for source-layer→target-layer metadata, alpha/contribution and known
  defaulted/unsupported fields, with explicit losses rather than a blanket preservation claim.

## Assumptions and boundaries

- “uniqueID” means placed object instance identity (MDDF/MODF and corresponding map references),
  not FileDataID, asset path ID, WMO group index, or a WMO's local doodad index.
- “Separate” means independently runnable CLI projects for Map, Model and WMO, separate Editor
  tools, and separate domain services; shared low-level libraries remain shared.
- “Generate” initially means derive target-compatible objects from available source geometry.
  Prompt/image generation, new training, or inventing unseen object geometry is not assumed.
- The loaded client is the initial source. Slot choice is explicit; route inference follows the
  chosen slots and never silently swaps them. Missing targets require selection.
- Exact 1.60.1 build/product and target 0.5.3 fingerprint are recorded during implementation proof;
  no local library scan or client conversion is required for this planning session.
- No global client-loader redesign, PM4 association research, editor-save overhaul, archive writer,
  renderer/camera change, full animation retargeting project, or blanket bidirectional modern export.
- Protected readers and `AlphaWdtWriter` stay frozen. A proven defect requiring those changes is
  handled by a separate evidence-backed scope decision, not absorbed into this plan.
