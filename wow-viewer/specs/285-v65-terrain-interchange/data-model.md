# Data Model: v65 Terrain Interchange

**Proposed schema**: `v65-terrain-interchange-v1`; **layout**: `native-chunks-v1`.
Storage and wire schema versions are explicit and independent of game-client version numbers.

| Entity | Required semantics |
|---|---|
| SourceRevision | Backend MPQ/loose/CASC, exact build/product, CASC build/config identities where applicable, source hashes/patch order/local content state, extraction version and configured root provenance. |
| MapRecord | Map identity/basis, WDT flags/tile presence, WDL native outer/inner samples and masks, relevant terrain-table identity. Global WMO placement references link to stored placement metadata. |
| TileRecord | Stable source/map/x/y key; explicit row; file/companion hashes; chunk index; per-field availability and coverage. |
| NativeChunk | Coordinates/basis, base Z and 145 authored samples/presence, native normals, exact hole topology, area/terrain flags, colors/shading/materials and required terrain emitters. |
| LayerRecords | Ordered per-chunk span, typed texture identity/FDID/local table key, layer flags/effect/material data, full-resolution alpha offsets/shapes/encoding provenance. No fixed four-layer cap. |
| LiquidRecords | Source family/layer identity, dimensions/coverage, heights/depth, raw/resolved types, flags and required source metadata. Unified raster is an optional derived view. |
| TerrainMaterial | Exact names/FDIDs/slot/order, layer/UV/blend/material/effect metadata and source table references; linked retained source/PNG texture resources. No object mesh payload. |
| Availability | Native available / derived available / absent in source / unsupported / parse-failed / not-selected, with reason and source evidence. Numeric zero does not mean absence. |
| ChunkCoverage | Source file/scope/tag/ordinal/coordinates/hash; per-stage decode/transport/store/export state; terrain/object/structural classification; target consequence. |
| TerrainOpaque | Retained terrain bytes including unclassified metadata, with occurrence/source/band/offset/header/hash/provenance; not decoded support. Known geometry/masks excluded. |
| DerivedView | Optional height/normal/alpha/liquid/image views with transform/version, source keys, validity and resolution. Authored object-bearing imagery is labelled observation, never clean ground truth. |
| Finalization | Schema/layout, complete tile/field counts, hashes, offset integrity, excluded-object audit, coverage summary and finalized state. |
| CellAppearance | Native cell key/topology/basis, ordered layers, exact alpha sample footprint and UV/blend/material/effect parameters; summaries are versioned views. |
| TextureResource | Source bytes, path/FDID aliases, build/content hash, roles/usage, format/compression/header/native dimensions/mips/channel/color/alpha metadata and RGBA PNG resources with hashes. |
| MetadataIndex | Source→file/chunk/header/unknown metadata and map/tile/chunk/cell/layer/resource links; inverse texture usage and availability queries. |
| InterpretationView | Versioned decoder/model/view selection over retained records with source digests and sampling/transforms; never overwrites native/source data. |
| LayerOpacityReport | Per-report/chunk/layer identity, MCLY/MCAL source ranges, encoding/decode/fixup/validity, separate decoded-alpha and composition contribution maps/statistics/histograms/coverage and cell summaries; complete layer accounting. |
| FieldMetadata | Name/role, axes/layer order, shape/dtype/units, encoding/domain, source field/chunk/hash, validity/availability, decoder/transforms and version; linked to every retained field rather than values without context. |

Use typed integers for bitmasks/IDs, preserve source widths/signedness explicitly, and define wire
endianness. Canonical native arrays may widen values losslessly; retain source packing provenance.
Ragged arrays use typed offsets/counts with checked bounds; field rows never drift from tile rows.

## Placement metadata (inside v65)

Placement table identity and each record bind to the same source revision/map/tile. Records carry source
row/kind, original **uint32 bits**, asset path/FDID, exact transforms, flags/sets/bounds, owner and
chunk/reference edges with index-versus-ID semantics. Extract from original typed reader records
before float32 projection; an ID greater than `2^24` must survive exactly.

Store ragged typed placement tables and MCRF/MCRD/MCRW relationship arrays with exact integer keys,
counts and offsets. No object masks, mesh library, WMO geometry/material/portal payloads are embedded.
Asset identity/dependency references are retained metadata, not object geometry. Exported manifests
may mirror these tables for audit. Spec 284 owns generated candidates and target ID maps.

## Lifecycle and migration

`selected → extracting → staging → validated → finalized`, alternatively `blocked/failed/cancelled`.
Readers/exporters admit finalized stores only. Partial state is inspectable but cannot be exported
as complete. A selected store change invalidates the export request digest.

Migration outputs new v65 stores with per-field copied/derived/missing/rejected/re-harvest status.
Old schemas/layouts are detected from real metadata, not directory labels. Source stores are
read-only. Missing native semantics remain unavailable until a new authorized harvest supplies them.

Committed raw/native/placement/texture/PNG records survive export. No automatic purge or replacing
source evidence with interpretations; new versions/views link to retained hashes. See the
[appearance/retention contract](contracts/appearance-retention.md) for resource/cell/query rules.

The [opacity contract](contracts/layer-opacity-report.md) distinguishes stored payload, decoded
alpha, supported effective contribution and missing/unknown samples. Its source reports persist
independently of later writer outputs; never key layer identity solely by shared texture ID.
