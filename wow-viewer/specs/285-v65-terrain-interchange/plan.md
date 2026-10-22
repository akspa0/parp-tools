# Implementation Plan: v65 Decoded Terrain and Placement Interchange

**Date**: 2026-10-21 | **Spec**: [spec.md](spec.md) | **Branch**: existing `280-ui-rework`  
**Status**: Planning only. Spec 284's map path depends on this contract.

## Summary

Use the current **data-harvester as the data interchange tooling**:

```text
MPQ / loose / CASC client or selected mounted source
  → current archive + terrain extraction libraries
  → opt-in v65 typed terrain and placement stream
  → Python-owned v65 Zarr dataset
  → Python typed export stream
  → current C# target materialization service
  → loose Alpha/LK terrain files

v65 placement metadata + separately resolved object assets → Spec 284 object tools
                                                       ↘ target placements + re-itemized IDs
```

The store is the terrain convergence point. Map target conversion does not independently read
source terrain. Existing original conversion algorithms supply coordinate/layer/reference logic
to port into this route; the old Converter executable supplies no functional guarantee or dependency.

## Technical Context

**Language**: C#/.NET 10 extraction and format output; Python/uv for existing harvesting/store tooling.  
**Dependencies**: `AdtTensorPackBuilder`, `AlphaTensorPackBuilder`, `TerrainVertexLattice`, existing
archive interfaces and ARRY concepts; Python Zarr v3/NumPy/PyArrow; existing content writers.  
**Storage**: Versioned, finalized per-build/map v65 stores with catalog/index and native terrain
records, typed placement metadata and retained appearance/source/PNG resources; no client archive
output or object geometry/masks. Committed v65 evidence is durable, not an expendable export cache.  
**Validation**: Per-field native round-trip and coverage checks on bounded real inputs, then Spec
284 target serialization and operator witness. Existing training metrics are not interchange proof.  
**Scale**: Selected tiles first; streaming bounded memory; full harvest is operator-owned.  
**Compatibility**: Opt-in new schema/layout/profile. No in-place v50/v60 change or mandatory trainer migration.

## Operator Intent Check

| Operator wording | Response |
|---|---|
| Existing data-harvester gets most data between versions | Reuse extraction and Python storage pipeline; factor current CLI orchestration into a reusable harvest service. |
| Dataset as decoded convergence | Both target directions consume finalized v65; native semantics precede target reduction. |
| Chunks not parsed/handled need additions | Complete per-occurrence coverage ledger and scoped new terrain handlers, including transport/store omissions. |
| Keep placement metadata; exclude geometry/masks; call it v65 | Typed placement/reference tables remain; mesh payloads, object libraries and supervision arrays are excluded. |
| Original conversion logic worked | Recover algorithms as source references, port with native real-data checks; do not replace them with inferred ML terrain. |
| Modern tooling; old project may only compile | No `WowViewer.Tool.Converter` dependency/invocation and no legacy CLI parity gate. |
| CASC support for reading | Reuse current CASC catalog backend, product/build and FileDataID resolution; verify a real CASC source through v65. |
| Keep all metadata/chunks for later viewing/models | Durable raw/native records and query indexes; versioned interpretations never replace source evidence. |
| Cell texture metadata; all tilesets PNG; terrain appearance reference | Layered cell appearance graph, native masks/parameters, source and RGBA PNG texture resources. |

## Constitution Check

| Gate | Decision |
|---|---|
| Core first / standalone repo | Core owns DTOs, Core.IO extraction/protocol/target adapters; Python under data-harvester; no legacy project dependency. |
| Python store ownership | Python reads/writes v65. New C# code consumes typed streams, not Zarr directly. |
| Streaming first | Existing nested framing is extended with explicit v65 schema/dtypes/limits; no NPZ interchange staging. |
| Frozen base | New opt-in extraction handlers/projections, not shared reader/renderer rewrites. Original Alpha writer remains frozen. |
| Storage / roots | Zarr v3 and loose content only; configured roots, repository-managed outputs; clients read-only. |
| Evidence / scope | Per-stage fidelity and missing-data gates; no object conversion, training or full-client datastore scope added here. |
| Operator runs | Broad harvest, expensive conversion and all real-client visual proof remain operator-owned. |

Pass by design; implementation/real-data gates are pending. This extends harvesting for authorized
missing chunk semantics, without authorizing unrelated changes to established readers.

## Design Decisions

### D1 — Add an opt-in profile; do not change the old training pack by default

Current extraction already provides many decoded fields, but it also rasterizes, truncates or
embeds objects. Add `V65TerrainRecord` plus typed placement metadata and field handlers above existing
reader/chunk interfaces. Where the current pack has already lost information (four-layer cap,
boolean holes, flattened liquids), obtain native values before those reductions using scoped
harvest handlers. Reuse existing decoding primitives; do not duplicate whole format parsers.

Extract `BuildEnrichedTensorPackForTile`/archive orchestration from the huge harvest CLI into
`Core.IO/Harvest/TerrainHarvestService.cs`. Keep the current CLI as a thin entry and use the same
service from converter integration. v65 must bypass object geometry/mask enrichment, not compute
it and merely discard the final arrays. Existing profiles retain their current behavior.

### D2 — Native semantics are authoritative; training rasters are views

Use native 145-sample chunk terrain, presence, basis, base-height/packing provenance, native normals,
full per-chunk layers/flags/effects and full-resolution alpha. Layers and liquids are ragged where
the source is ragged. Preserve exact low/high-resolution hole masks, all liquid layers and their
coverage/types/heights/depth/flags. No four-layer or 256-pixel reduction happens during v65 ingestion.

Retain terrain shading/colors, area/flags, material names/FDIDs and required terrain-table resolution,
flight bounds, terrain sound emitters, WDT/WDL metadata and their source distinctions. Zero, absent,
unsupported and parse failure are different. Derived images/rasters are optional named views with
source/version/availability; never use inpainted `ground_intent_height_257` as native terrain.

### D3 — Coverage spans decode, transport, persistence and export

The [coverage contract](contracts/terrain-interchange.md) starts with an occurrence inventory,
including root and selected texture/object companion metadata, WDT and WDL. Mark each item as
decoded-native / decoded-placement / decoded-derived / terrain-opaque / geometry-or-mask-excluded / structural / unsupported.
Track every stage independently: a property in the pack does not prove it reaches Zarr or export.

Add missing terrain handlers after inventory. Retain every observed terrain chunk/header/unknown
payload with source file/band, offsets/order/tag, coordinates, bytes and hash, including chunks
whose decoded tensor projection loses fields. Unknown bytes remain opaque rather than decoded
cross-version support. Preserve source evidence for later interpretations. Known object geometry/
mask payloads stay excluded; placement metadata remains included. Raw source snapshots/regions
are evidence only, never a substitute for native decoded interchange.

Structural offsets/counts are regenerated at target serialization. Placement chunks and MCNK object
reference fields are retained as typed dataset metadata, including MCRF index/uniqueID semantics.
Required undecoded semantics block preserve-features export; reviewed reductions belong to Spec 284.

### D4 — Explicit Python schema, field availability and layouts

Implement `harvester/v65/` with contracts, ingest, store, read, validate, migration and export-stream
modules. Reuse v50 staging/hash/lineage patterns and current stream decoding primitives. A new
schema ID and layout ID distinguish native records, ragged arrays, optional views and provenance;
do not inherit v60's flat/dedup ambiguity or generic first-shape stacking.

Choose stable tile keys `(sourceRevision,map,tileX,tileY)`, explicit row positions and native/ragged
field schemas. Missing fields still retain the row and validity reason; no zero fill that claims
presence. Finalization verifies hashes, offsets, counts, dependencies, allowlists and coverage.
No automatic broad consolidation or existing-corpus rewrite is part of this feature.

### D4a — CASC is a first-class source backend

Reuse current `OpenClientCatalog` CASC/MPQ selection through the extracted harvest service; do not
assume a mounted CASC source is a loose/MPQ directory. Bind CASC install/product/build/CDN-config
identity and local content state in SourceRevision. Resolve map/tile/texture/placement companion
references by path or FileDataID using current catalog APIs; retain unnamed-file identity when no
path exists. Missing listfile names do not authorize guessing a different file or client.

Validate locally available data with CDN fill disabled by default. Any existing explicitly selected
network-fill mode remains separately visible; no new network retrieval feature is required.
CASC bytes are read-only; output is v65 plus loose target content. Run a bounded actual CASC pilot
and MPQ/loose controls; mocks/builds alone cannot establish this source capability.

### D5 — Preserve typed placement metadata inside v65 before float conversion

Store typed source placement records and target-independent references in a distinct metadata
group inside v65. uniqueID/flags/name IDs are exact integers, never recovered from float arrays.
Use checked ragged offsets/counts, source revision/map/tile/record identity and explicit reference
edges. Include asset locators, transforms, bounds, flags, doodad/name sets and unresolved dependencies.
Global WMO-map placement metadata is retained too. Geometry/masks remain excluded. Optional exported
manifests are audit mirrors, not the only placement authority. Spec 284 reads these records for ID
allocation, asset conversion/remeshing/regrouping/generation and target object serialization.

### D6 — Reverse stream keeps Python as the sole Zarr owner

Python reads a finalized selection and emits a versioned typed terrain stream. C# validates schema,
layout, dtypes, shape/length bounds, hashes, availability and terminal completion before export.
Reuse the existing outer frame/inner ARRY concepts; distinguish this from V22E enrichment. Define
integer-width support explicitly, including uint64 holes; reject unknown dtypes rather than falling
back to float32. No NPZ/temporary target files and no direct C# Zarr dependency.

Missing reverse transport is new implementation work. After harvesting, terrain export can run
with the source client detached; declared terrain material dependencies must still be available.
The original client may still be needed independently by the separate object asset path.

### D7 — Migration is audited copying, not a version-label change

Identify actual old layout/schema and field coverage. Copy only faithful native arrays and validated
metadata; quarantine misaligned/zero-filled/ambiguous fields, exclude object geometry/masks and preserve eligible exact placement/reference metadata, and request
re-harvest for missing full layers, exact holes/liquids/material identity. Preserve all old stores.
Existing v60 trainers remain on their contract; v65 is interchange, not a new ML experiment.

### D8 — Durable appearance reference, cell indexes and PNG resources

The operator extends v65 into retained terrain evidence for later viewers/models. The detailed
[appearance/retention contract](contracts/appearance-retention.md) is part of this plan. Retain
original source/native records and unknown bytes, plus versioned decoded interpretations, without
automatic export eviction. Build source→map/tile/chunk/cell/layer→resource indexes and inverse
texture usage queries; no unknown field disappears because it lacks a named schema property.

Cells resolve ordered source layer identities and full native masks, not a single texture ID.
Record UV/blend/material/effect/animation/height-map parameters and relevant terrain appearance
table/context rows. Keep authored color/shadow/light inputs separate from reconstructed views.
Collect all referenced terrain textures through MPQ/loose/CASC, retain source bytes and native
format/mip metadata, and derive native-resolution RGBA PNG resources with content identity.
PNG is a reusable model/viewer representation; it does not replace source compression/palette,
mips, channel meaning or unresolved shader semantics. Existing RGB/resize exporters are insufficient.

No model/trainer implementation or new renderer is added. Versioned model/query views reference
the durable source records; target exporters apply measured per-version reductions later.

### D9 — Per-layer opacity reporting, preserved alongside source data

Follow the [layer opacity report contract](contracts/layer-opacity-report.md). Preserve all declared
layers before any texture aggregation, with raw MCLY/MCAL identity and decoder/profile/heuristic/
fixup/validity metadata. Store separate decoded-alpha and supported effective-contribution maps,
statistics/histograms/coverage and native-cell summaries. Include implicit base and unknown/zero/
fully overpainted/repeated-texture layers. A decoder's filled map is not proof all samples were observed.

Use the existing ordinary ordered alpha-over behavior as one named model, never raw-channel
normalization. Unknown height-blend/shader behavior retains source/decoded evidence with contribution
marked unknown or explicitly a reference model. Reports do not claim material-image alpha or final
rendered opacity. Existing counts that increment per pixel are not counts of missing layer records.

Persist reports in v65 as versioned metadata/derived arrays tied to immutable source/native hashes;
reader/query/model views expose them without recomputing from an altered target. The source report
unblocks later writer checks; it never depends on target output. After Spec 284 writes output,
read it independently and compare layer reports using the recorded source→target layer mapping.

## Project Structure (proposed)

`Core/Harvest/V65LayerOpacityReport.cs` owns shared DTOs;
`Core.IO/Harvest/V65LayerOpacityAnalyzer.cs` owns native/source analysis;
`harvester/v65/layer_opacity.py` owns persistence/query/export through Python-owned v65 I/O.
All retained layered fields also carry axis/shape/dtype/units/availability/source/transform metadata;
opacity-specific quantities apply to alpha layers rather than inventing opacity for unrelated fields.

```text
src/core/WowViewer.Core/Harvest/V65TerrainRecord.cs
src/core/WowViewer.Core/Harvest/V65CoverageRecord.cs
src/core/WowViewer.Core.IO/Harvest/
  TerrainHarvestService.cs  V65TerrainProjector.cs  V65TerrainStreamWriter.cs
  V65TerrainStreamReader.cs  V65ChunkCoverageCollector.cs
  V65NativeTerrainHandlers.cs  V65PlacementMetadataExtractor.cs
tools/harvest/WowViewer.Tool.Harvest/Program.cs          # delegation/profile registration only
data-harvester/src/harvester/v65/
  contracts.py  ingest.py  store.py  reader.py  validate.py  migrate.py  export_stream.py
  retained_evidence.py  indexes.py  appearance.py  texture_resources.py  model_views.py
src/core/WowViewer.Core.IO/Harvest/
  V65AppearanceMetadataExtractor.cs  V65TerrainTextureCollector.cs
data-harvester/scripts/v65_dataset.py                   # thin inspect/ingest/validate/migrate/stream CLI
data-harvester/tests/v65/
tests/WowViewer.Core.Tests/Harvest/V65/
```

Missing handlers may be split by chunk family to honor file budgets. No additions to god-classes,
no source dependency on old Converter, and no new trainer or object dataset.

## Phases and validation gates

1. Inventory current pipeline and freeze v65/native field and object-exclusion contracts.
2. US1: shared modern harvesting service, typed profile and Python ingestion/store.
3. US2: required native chunk handlers, per-stage coverage and unknown handling.
4. US3: validate geometry/mask exclusion and exact stored placement metadata.
5. US4: reverse stream, source-detached readback and Spec 284 handoff.
6. US5: durable source/unknown metadata retention and query indexes.
7. US6: layered cell appearance, full terrain PNG resources and future-consumer views.
8. US7: every-layer opacity/provenance/statistics and supported contribution reporting.
9. Audited migration, bounded real-data receipts and continuity.

Maximum ten tasks per phase; one validated phase at a time. Initial MVP is a native terrain tile
with coverage/readback and explicit placement-pending status; typed placement readiness follows
Phase 5. Native/store/reverse-stream readiness unblocks Spec 284's target adapter. SC-004 is joint
acceptance after that adapter exists, not a circular prerequisite. Broader target acceptance
remains Spec 284/operator-owned. See [tasks.md](tasks.md) and [quickstart.md](quickstart.md).

## Complexity Tracking

No new format-storage stack. v65-specific projection and reverse transport are required because
the existing training-oriented profile loses native data and C# has no current v65 store reader.
