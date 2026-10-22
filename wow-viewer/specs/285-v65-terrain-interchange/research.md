# Research: Current Harvester to v65

**2026-10-21**. Repository source inspection only. No harvest, Python job, client scan, build or
visual run. The operator clarified that placement metadata stays in v65; geometry and masks do not.

## R1 — Current extraction is reusable; the CLI is not yet a library API

**Decision**: Factor current harvest orchestration into a Core.IO service and add an opt-in profile.
**Evidence**: `src/core/WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs` exposes `Build/BuildFromBytes`;
`AlphaTensorPackBuilder.cs` builds Alpha packs. `TerrainTileTensorPack.cs` is the current shared
pack. `tools/harvest/WowViewer.Tool.Harvest/Program.cs` has `RunHarvestStream`, `OpenClientCatalog`
and private `BuildEnrichedTensorPackForTile`; it opens CASC/MPQ through current catalog interfaces.
**Rationale**: Reuse actual extraction, but do not force the Editor to call CLI-private methods.
**Alternative rejected**: Depending on old `WowViewer.Tool.Converter` or duplicating source decoders.

## R2 — Current dataset/transport is reduced, not conversion-complete

**Decision**: Native schema with independent coverage per pipeline stage.
**Evidence**: `Core.IO/Maps/RawArraySerializer.cs` defines Full/V16/V22. `WriteV22Arrays` emits
native vertex/WDL arrays but only four-layer alpha256 and cropped liquid views. It omits fields
available in Full/pack such as MCCV/MCLV/MFBO/MCMT/MAMP, terrain emitter/reference/raw chunk data.
`AdtTensorPackBuilder.ReadTextureData` already caps layers to four. `TerrainTileTensorPack.HoleMask16`
is a per-chunk boolean, not exact hole topology. Its `ToTerrainTileData` convenience path defaults
area metadata and contains empty placements; it is not a complete reconstruction contract.
**Rationale**: A reduced tensor cannot reconstruct data discarded before persistence.
**Alternative rejected**: Renaming v60 or using alpha256/height257 as native interchange authority.

## R3 — Python has storage ownership and real fidelity hazards to avoid

**Decision**: New `harvester/v65` module using current validated patterns, preserving old versions.
**Evidence**: `data-harvester/scripts/v60_build_dataset.py` invokes `harvest-stream --stream-profile
v22`; `_write_per_build_store` selects only ndarray keys, dropping `_metadata`, stacks first shapes,
zero-substitutes differing shapes and omits missing-signal rows. Flat/dedup paths use the same schema
label. `src/harvester/v60/store.py` also merges raster fields; `v50/store.py` supplies stronger
staging/finalization/lineage patterns. `v50/contracts.py` defines live v50/v60 version contracts.
**Rationale**: v65 needs typed ragged fields, row validity, exact identity and an explicit layout.
**Alternative rejected**: Reusing generic stacking and pretending missing arrays were valid zeros.

## R4 — Preserve exact placement metadata, exclude incomplete object geometry

**Decision**: Typed integer placement/reference metadata in v65; no meshes or object supervision.
**Evidence**: `TerrainTileTensorPack.PlacementMddfData/PlacementModfData` are float arrays;
`RawArraySerializer.WriteV22Arrays` casts those columns back into integer uniqueID arrays. IDs above
float32's exact integer range can already be lost. `Core/Maps/V22ModelPayload.cs` flattens some M2/
WMO geometry but does not preserve complete UV/material/batch/portal/BSP/collision semantics.
`v22_zarr_io.py` has older ragged placement and model libraries, not proof these survive v60.
**Rationale**: Extract original integer bits before the float projection; object payloads neither
belong in the requested dataset nor establish reversible asset conversion.
**Alternative rejected**: Float-cast IDs, an external-only placement authority, or dataset mesh blobs.

## R5 — Reverse transport is planned work

**Decision**: Python store read → typed stream → C# target adapter, keeping Python the Zarr owner.
**Evidence**: `raw_reader.py` reads nested ARRY payloads with `_metadata`. Its dtype lookup has a
float32 fallback; uint64/int64 support needs explicit handling. Current harvester comments identify
the absence of a general C# Zarr reader. No current v65 reverse path exists.
**Rationale**: Reuse framing, version it and validate lengths/dtypes/completion. Do not introduce
another storage implementation or NPZ/target-file staging as an interchange shortcut.
**Alternative rejected**: C# direct Zarr reads or treating V22E enrichment as the same transport.

## R6 — Scope is dataset interchange, not a legacy training revival

**Decision**: New Spec 285 owns the schema; amend Spec 284 to consume it. Preserve historical docs.
**Evidence**: Epic 254 is active dataset ownership. Spec 269 describes current v60 ingestion.
Archived 134 was amended into a controlled terrain experiment and does not authorize a broad
interchange rewrite. `docs/architecture/v22-dataset-signals-2026-06-30.md` documents an older model/
placement contract; its claims cannot be projected onto every v60 store. Quilt materialization
patches templates or emits terrain-only LK; it is not full v65/Alpha reconstruction.
**Rationale**: No training, unknown model reconstruction or general full-client datastore is needed.
**Alternative rejected**: Editing archived acceptance claims or treating an old filename as authority.

Remaining proof is assigned to tasks: selected-corpus chunk inventory, missing native handlers,
typed placement custody, forward/reverse stream/store readback and target output. No unresolved
product clarification remains after the operator's placement answer.

## R7 — Retained appearance data, cells and PNGs (operator addition)

**Decision**: Retain source/raw metadata, layered cell indexes and original/PNG texture resources;
future viewing/models consume versioned views over durable evidence.
**Evidence**: `Core/Maps/AdtTextureChunkLayer.cs` and `AdtTextureLayerDescriptor.cs` retain ordered
IDs/flags/alpha offsets/effects. `AdtTextureReader.cs` handles MTEX and some MDID resolution;
numeric identities need custody. `AdtMcalDecoder.cs` has encoding/profile/fixup state. Tensor
projection drops metadata/caps layers. The raw collector skips decoded chunks and chooses one
effective band, so it is not complete source evidence. `MopAdtChunkParser` retains raw MTXP;
complete MHID semantics were not established. GroundEffectLookup provides build-aware table refs.

`AdtTextureTrainingSupervisionExporter.cs` emits weights and a lossy dominant-index PNG.
Harvest `RunExtractTilesets` forces alpha=255 after RGB decode; viewer VLM export resizes to512
and flattens basenames. `BlpRgbReader` strips alpha; `BlpSummaryReader` exposes original format/
mip/header metadata. These are useful primitives, not preservation exports.

**Rationale**: Cells blend layers; one ID or PNG cannot preserve masks/material/auxiliary/color/
shadow/lighting/unknown parameters. Native RGBA/mips/source bytes require a bounded adapter.
Retaining evidence permits future interpretations without re-harvest; missing resources stay explicit.
**Alternatives rejected**: Dropped unknown bytes, RGB/proxy/resized PNGs, dominant-only labels,
export cleanup of source evidence or adding training/model architecture work now.

## R8 — MCAL/MCLY opacity requires complete per-layer reports

**Decision**: Preserve source/decode metadata and per-layer decoded alpha separately from supported
effective composition; persist maps/validity/statistics and include every declared layer.
**Evidence**: `AdtTextureChunkLayer` and `AdtMcalDecodedLayer` retain order/texture/flags/effects/
offset/encoding/bytes consumed/applied fixup. `AdtMcalSummaryReader` has layer/decode counts, not
full opacity statistics. `TerrainMinimapCompositor` applies ordered Lerp; the supervision exporter
implements equivalent reverse contributions, then merges repeated texture IDs. Its missing-alpha
counter increments inside a pixel loop, so it is not a missing-layer count. Decoder zero-fill/
edge/nibble fixups and heuristic encoding decisions need explicit validity/provenance.
**Rationale**: Raw alpha channels are not normalized weights; base residual and later occlusion
matter. Unknown height blending cannot be promoted to measured final contribution.
**Alternatives rejected**: Per-texture-only summaries, dropped zero layers, unqualified fixed-map
statistics or interpreting missing alpha as observed zero. See the layer report contract.
