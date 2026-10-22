# Feature Specification: v65 Decoded Terrain and Placement Interchange

**Created**: 2026-10-21 | **Status**: Planning only | **Owner**: Epic 254, consumed by Spec 284  
**Working branch**: Existing `280-ui-rework`; no implementation or harvest authorized this session.

## Operator authority and correction

The operator directs the converter work to use the existing data-harvester and dataset as the
convergence point where data is decoded and interchangeable, extend the dataset specification for
chunks not yet parsed/handled, remove WMO/object geometry and masks, and name the revised dataset **v65**.
The operator explicitly clarified: **"Keep placement metadata in v65; exclude object geometry and masks."**
They also require **CASC support for reading source data**. CASC is an input backend, not an output format.
They direct reuse of the original conversion logic rebased on current modern tooling, and report
that the old Converter executable has not been tested since approximately April. Its presence or
successful compilation is not a functional baseline.

This is an operator-requested successor contract, not an edit to archived v60 experiments. Existing
v50/v60 data, trainers and readers remain intact. Spec 284 retains separate CLI/Editor converter
tools; this spec owns their common terrain interchange. Planning only remains the active scope.

### Operator addition — retained terrain reference, 2026-10-21

Retain ingested data for later viewing and future models, including metadata/chunks without a
semantic home yet. Preserve all data used by terrain appearance, texture identity down to terrain
cells, and **all referenced terrain tilesets decoded to PNG**. v65 is the durable reference for
future target-version exporters; individual exporter availability remains evidence-gated.
Placement metadata stays included, geometry/masks excluded. Unknown terrain bytes are retained
and indexed, superseding the initial proposal to omit unclassified terrain payloads.

## User Scenarios & Testing

### US1 — Decode selected client terrain once into v65 (P1)

An operator uses the existing harvesting pipeline to obtain version-independent terrain with native
fidelity, source identity and explicit availability. No map converter independently re-decodes the
client terrain or uses training rasters as its only source.

**Independent verification**: Harvest a bounded real selection from Alpha 0.5.3, a legacy control
and the selected 1.60.1 build; compare native decoded values and lineage after store readback.

1. Given selected client/map/tiles, v65 records exactly those rows with build/product/hash identity.
2. Given native heights, layers, holes or liquids, their structure is retained before any target
   reduction; derived 257/256 rasters are labelled views, not native replacements.
3. Given absent/failed fields, availability and reason remain explicit and row alignment is stable.
4. Given a configured CASC install/product, the same pipeline reads terrain, placement metadata,
   textures/table identities and companions by path/FileDataID with recorded build identity;
   missing local data is explicit and cannot silently fall back to an unrelated client.

### US2 — Account for every observed chunk and extend missing handling (P1)

The operator can see which input chunks are decoded and interchangeable, decoded but reduced,
opaque, excluded as object data, structurally regenerated, or unsupported. Missing terrain
semantics required for conversion get dedicated harvesting/dataset handlers.

**Independent verification**: A real-file inventory accounts for every observed chunk occurrence;
its decode → stream → store → export → target disposition is independently recorded.

1. Opaque bytes do not count as a decoded cross-version capability.
2. Known required but unhandled terrain fields prevent a full-fidelity export until implemented
   or explicitly reduced by a reviewed target policy.
3. Unknown terrain bytes are deliberately retained/indexed with provenance; known object geometry/
   masks are excluded. Generic blobs cannot bypass identity/classification and evidence accounting.

### US3 — Preserve placements while excluding object geometry and masks (P1)

The dataset has no WMO/M2 geometry, object masks, instance labels or mesh libraries. Typed placement
tables and reference relationships remain in v65 for the separate object converter tools.

**Independent verification**: Validate the v65 field allowlist and absence of object geometry/masks;
join stored placement records to the same source revision/map/tile and verify exact original IDs.

1. Object geometry, WMO grouping/material/portal data and embedded model payloads are not stored.
2. Typed placement tables inside v65 carry original integer IDs, transforms, flags,
   references and ownership; identity is never recovered by casting float32 placement columns.
3. Terrain-only export needs no object decoder. Full map/object conversion uses the
   stored placement metadata and separate object tools, with closure and re-itemization owned by Spec 284.

### US4 — Export decoded v65 through modern tooling (P1)

Separate CLI and Editor map tools consume one decoded v65 contract and serialize target formats
using the established conversion algorithms behind current library/harvester boundaries.

**Independent verification**: Read a finalized v65 selection through the Python-owned store layer,
stream typed terrain to the C# export adapter, and compare native/target output with source evidence.

1. Python remains the store I/O owner; C# consumes a versioned typed stream, not a new Zarr reader.
2. After ingestion, terrain export is possible with the original source terrain unavailable;
   terrain textures/tables required for export must be resolved from declared material dependencies.
3. The original Converter executable is not invoked, linked or treated as proof. Historical
   transformations are ports/reference algorithms with fresh real-data validation.

### US5 — Retain all terrain evidence for later use (P1)

An operator can revisit a source revision, inspect decoded metadata or unknown raw chunks, and
add newly understood decoders/views without losing or re-harvesting existing evidence.

**Independent verification**: Retrieve native and unknown records with the client unmounted;
verify bytes/hashes/indexes survive a derived-view or interpretation revision.

1. Every selected terrain file/chunk/header and observed metadata field has a retained home and
   queryable source scope/offset/tag/hash, even without a known semantic interpretation.
2. New interpretations link to retained source evidence; opaque values never count as decoded.
3. No automatic export cleanup/eviction removes committed source/native/resource records.

### US6 — Trace terrain appearance to cells and PNG tilesets (P1)

Operators and future model consumers can resolve a terrain cell to ordered contributing texture
layers, native masks, UV/blend/material metadata and all required retained PNG texture resources.

**Independent verification**: On real MPQ/CASC samples, verify both cell→layer→texture→PNG and
texture→affected-cell queries, comparing source masks/metadata and native texture pixels.

1. A cell's multiple contributing textures remain a layer/weight set; a dominant texture ID is a
   labelled derived view, not the authoritative material representation.
2. Each referenced terrain diffuse/auxiliary texture has retained identity/source bytes, native
   dimensions/mips/format/alpha metadata and PNGs, or an explicit missing/decode-failed state.
3. Geometry, UV scale/orientation/animation, height blending, color/shadow/lighting and relevant
   appearance tables remain traceable; PNG pixels alone do not represent all material behavior.

### US7 — Inspect every layer's opacity and metadata (P1)

Operators and future consumers can inspect one report row for each declared MCLY layer, its MCAL
source/decode evidence, decoded alpha statistics and supported effective contribution. Layer zero,
zero-alpha, overpainted, repeated-texture and unavailable layers remain visible and distinct.

**Independent verification**: For bounded real layers and known blend controls, compare retained
bytes/flags/offsets, native decoded maps, per-layer statistics and cell summaries; after export,
compare an independently read target report through the explicit source→target layer mapping.

1. Every declared layer has identity/texture/flag/effect/offset/encoding/decode/fixup metadata,
   valid/unknown sample counts and statistics or a specific unavailable reason.
2. Stored alpha, effective ordered contribution and texture-image alpha are separate concepts.
   Raw MCAL channels are not normalized. An implicit base and unsupported height blending are explicit.
3. Native maps/histograms/coverage and cell summaries are versioned, indexed and retained with
   source digests. Zero or invisible layers are not silently removed; missing values are not zeros.

## Requirements

- **FR-001**: Name and version the new contract `v65`; introduce an opt-in harvesting/serialization
  profile and Python store modules using existing tooling rather than relabelling v60.
- **FR-002**: Preserve native terrain samples, material/layer semantics, full-resolution masks,
  hole topology, liquids, map/chunk metadata, shading, WDL and terrain dependencies with typed
  identity, coordinate conventions, provenance and per-field coverage.
- **FR-003**: Inventory every observed chunk through the complete pipeline, and add missing terrain
  handling needed by the selected conversion routes; unknown semantics remain explicitly unknown.
- **FR-004**: Exclude WMO/object geometry payloads and object-derived masks/instance targets from v65;
  retain MDDF/MODF/MCRF/MCRD/MCRW placement metadata and asset identity tables without mesh data.
- **FR-005**: Preserve map placements inside v65 for conversion with exact integer uniqueIDs,
  source record identity and reference semantics; never derive source IDs from float32 tables.
- **FR-006**: Retain Python ownership of Zarr read/write and existing framed streaming concepts;
  support typed reverse streaming to current C# target serialization without ad-hoc target files.
- **FR-007**: Keep old datasets immutable; migration copies only eligible fields with validated
  lineage/fidelity, and requests fresh harvesting for data old stores never preserved.
- **FR-008**: Validate per-field native semantics and dependency/coverage integrity on bounded real
  inputs, followed by the separate converter's target output checks and operator witness.
- **FR-009**: Read selected source data from both MPQ/loose and CASC through current catalog tooling;
  CASC selection records install/product/build identity and FileDataID/path resolution, resolves
  required companions, and reports missing/unavailable files. No CASC writing or implicit CDN fill.
- **FR-010**: Retain committed native/source terrain data, observed metadata and unknown chunk
  bytes with provenance, indexes and versioned interpretations for later inspection/model use;
  export and view generation must not evict or overwrite the source revision.
- **FR-011**: Preserve source material/texture identities and map/tile/chunk/terrain-cell/layer
  relationships with native masks, sampling, UV and blend semantics, including unknown parameters.
- **FR-012**: Resolve every referenced terrain appearance texture and retain original bytes plus
  native-resolution RGBA PNG derivations, mip/format/color/alpha metadata, hashes and usage indexes.
  No proxy/resized RGB counts as a faithfully retained texture.
- **FR-013**: Preserve terrain geometry/holes/liquids, material/effect/flag data, source colors,
  shadows/baked lighting and relevant appearance table/context inputs, with explicit incomplete
  dependency states. Unknown terrain semantics retain bytes and indexes for future interpretation.
- **FR-014**: Provide versioned read/query/export contracts for later viewers/models, including
  cell/layer/texture lookup and deterministic PNG export; no model training is required now.
- **FR-015**: Retain/query/export MCAL/MCLY per-layer opacity reports with exact layer identity,
  payload/decode/fixup provenance, native alpha and supported contribution maps, statistics/
  histograms/coverage/validity and cell summaries. Record every declared layer and separate
  raw alpha, composition-derived contribution and unknown semantics; retain all common field metadata.

## Key entities

Client revision; map/tile/chunk identity; native terrain record; terrain material dependency;
field availability; chunk disposition; finalized v65 store; typed placement tables; export
stream; target transformation report.

## Success Criteria

- **SC-001**: Every selected row retains exact source identity and valid field alignment; available
  native terrain values agree after decode → stream → store → stream round-trip.
- **SC-002**: Every chunk occurrence in the bounded real corpus has a disposition at all pipeline
  stages; zero opaque-only chunks are reported as decoded cross-version support.
- **SC-003**: v65 validation finds zero embedded WMO/object geometry or object target arrays.
  Stored placement IDs/flags/reference relationships retain integer precision and complete lineage.
- **SC-004**: The separate map exporter produces structural Alpha/LK output from finalized v65
  without re-reading source terrain or running the old Converter; target losses are explicit.
- **SC-005**: Existing dataset versions remain unchanged and are never silently promoted to v65;
  migration reports preserved, missing, rejected and re-harvest-required fields.
- **SC-006**: A bounded real CASC-backed source selection reaches finalized v65 with correct
  source identity, FileDataID/companion coverage and native/placement readback; MPQ/loose controls
  still pass. Compilation or a mocked catalog alone cannot close this criterion.
- **SC-007**: Every observed terrain chunk/metadata occurrence is retained with indexed provenance
  or a known geometry/mask exclusion; unknown terrain bytes retrieve hash-identically with the
  source unmounted and survive later interpretation/view revisions.
- **SC-008**: Every referenced terrain texture in the accepted pilot has source/hash identity,
  native-resolution PNG coverage and usage indexes. Missing/decode-failed assets leave appearance
  completeness open, without placeholders counted as successful texture conversion.
- **SC-009**: A sampled real terrain cell resolves every contributing layer, native mask and
  UV/blend/material/effect metadata and PNG resource; inverse texture→cell lookup agrees.
- **SC-010**: Committed native/resource records remain unchanged after export/model-view generation;
  later consumers read them through the versioned contract without source-client access.
- **SC-011**: The report accounts for every declared layer, preserves raw/decode metadata and
  separates measured zero from missing/invalid data. Supported complete composition samples obey
  their declared numeric consistency checks; source/target reports expose writer losses and
  quantization per mapped layer without changing source evidence.

## Assumptions and boundaries

- Operator-confirmed boundary: placements remain inside v65; object geometry and masks do not.
- An authored minimap may contain object pixels. It may be retained only as a labelled source
  observation; it is neither decoded object encoding nor clean terrain ground truth. No inpainting
  or training is required. Terrain interchange never depends on it.
- Observed unknown terrain chunks need retained bytes, indexes and disposition; this does not promise universal decoding
  of every game version. Required pilot semantics are implemented before claiming that route.
- New chunk handlers extend the harvest interchange layer. Working readers/renderer behavior and
  frozen `AlphaWdtWriter` are not rewritten. A proven underlying bug is recorded separately.
- No object reconstruction, remeshing, grouping, target uniqueID allocation or generation inside v65.
  Those remain in Spec 284's separate asset/output path.
- “All tilesets” means every terrain appearance texture referenced by the selected maps/revisions,
  including auxiliary/height/liquid maps and relevant table references, rather than unrelated client art.
- “Any output version” is a target-independent reference architecture, not a claim every target
  writer exists or every modern appearance feature is representable in every older client.
- Broad harvesting, GPU/training and client visual proof remain operator-owned. No implementation
  or heavy run is performed during specification/planning.
