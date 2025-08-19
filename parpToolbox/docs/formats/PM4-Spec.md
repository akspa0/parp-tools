# PM4 Specification (Canonical)

Status: Authoritative. This document consolidates verified truths about the PM4 format. It intentionally excludes software-derived/convenience fields and speculative interpretations. Non-normative implementation notes are clearly labeled.

## Scope and Principles

- This spec documents raw PM4 chunk data and confirmed inter-chunk relationships only.
- Parsing/algorithmic guidance (software-derived values, heuristics) is outside this spec.
- Process PM4 strictly per tile. Do not build a unified global scene across tiles. Treat cross-tile references as non-rendering metadata unless proven otherwise.

## Conventions (Normative)

- FourCC values on disk are little-endian. Canonical identifiers are the byte-reversed ASCII strings (e.g., bytes "REVM" → "MVER").
- Signedness: Certain numeric fields use signed sentinel values. In particular, index offsets may be `-1` to indicate a non‑rendering/container entry. Treat these as raw signed values at the spec level.

## Processing Model (Normative)

- Per-tile processing is required. Handle each PM4 file independently.
- Assembly is driven by hierarchical containers. Entries with no geometry are containers that organize geometry-bearing entries.
- Placements in `MPRL` map to geometry via confirmed relationships in `MSLK`.

## Chunk Inventory (Overview)

- MVER: Version header.
- MSHD: File header (details not standardized here).
- MSVT/MSPV: Vertex data (variants exist). MSVI/MSPI: Index data (variants exist).
- MSUR: Surface definitions (face/plane records used to navigate MSVI ranges and attributes).
- MSLK: Link records between placements and geometry/index slices.
- MPRL: Placement list (positions/identifiers used to map to `MSLK`).
- MPRR: Property references with sentinel separators.
- MSCN: Additional vertex-like data used as spatial/collision metadata (not directly indexed by faces).
- Other chunks (e.g., MDOS, MDBH/MDBF/MDBI, MDSF) exist but are out of scope unless explicitly referenced below.

## Dual Geometry System

- Structural path: `MSLK` links placements (`MPRL`) to index slices (`MSVI`/`MSPI`) and vertex pools (`MSVT`/`MSPV`). This path defines how geometry is referenced and organized, including container/group entries where `MspiFirstIndex = -1`.
- Render-surface path: `MSUR` provides face/plane records and attributes that direct how `MSVI` ranges are interpreted for rendering and diagnostics.
- Guidance (Normative): Faces are realized by applying `MSUR` interpretation to `MSVI` ranges, with link/placement context from `MSLK` and `MPRL`. Do not infer building boundaries purely from `MSUR` records.

## Confirmed Fields and Relationships (Normative)

### MSLK (Link Records)
- ParentIndex: Confirms placement mapping. It equals `MPRL.Unknown4`.
- MspiFirstIndex: Index offset into index buffer; value `-1` indicates a container/grouping node with no geometry.
- MspiIndexCount: Number of indices associated with this link. Container/group nodes have zero effective geometry.
- LinkIdPadding: Always `0xFFFF`.
- `LinkIdTileX`, `LinkIdTileY`: Present and encode the PM4 tile grid position.
- ReferenceIndex: A single 32-bit field exists. Any split into High/Low words is a software-side convenience only.
- Signedness: `MspiFirstIndex` is signed; value `-1` is the container sentinel and denotes no geometry for that link.

Notes:
- Do not rely on a `HasGeometry` flag in raw data. That is a derived convenience: `MspiFirstIndex >= 0 && MspiIndexCount > 0`.
- Do not document or depend on `HasValidTileCoordinates`, `TileCoordinate`, or `LinkSubKey` as raw fields. These are software constructs.

### MPRL (Placements)
- Unknown4: Equals `MSLK.ParentIndex` (confirmed mapping from placements to geometry links).
- Unknown6: Always `32768` (`0x8000`) in observed real placements.
- Position: 3 floats for placement position. The coordinate handedness/orientation fix is implementation-specific (see Non‑normative Notes).

### MPRR (Property References)
- Value1 = `65535` (`0xFFFF`) are sentinel separators between property sections. They do not define building/object boundaries.

### MSUR (Surface Records)
- IndexCount: Present. Useful for diagnostics/visualization. It is not a definitive object identifier.
- MSUR directs how MSVI index ranges are interpreted (face counts/offsets and attributes).
- Attribute masks/flags: Present. Exact semantics are dataset‑dependent and remain under investigation.

### MSCN (Spatial/Collision Metadata)
- MSCN provides additional vertex-like data used as spatial anchors/metadata. It is not directly indexed by face records. Treat it as non-rendering metadata unless proven otherwise.
- Usage: Treat MSCN as non‑rendering spatial metadata. Downstream correlation to external assets (e.g., WMO) is an implementation concern and outside this spec.

## Assembly Guidance (Normative Where Stated)

- Treat `MSLK.MspiFirstIndex = -1` entries as containers with no geometry.
- Map placements via `MPRL.Unknown4 ↔ MSLK.ParentIndex`.
- Faces come from `MSUR` guiding `MSVI` ranges. Do not assume global scene merges across tiles.

## Deprecated or Disproven Claims (Non‑normative list maintained for clarity)

- MPRR.Value1=65535 are building/object boundaries. Status: Deprecated. They are property separators.
- Global multi‑tile vertex pool is required for geometry assembly. Status: Deprecated for exporters; use per‑tile processing. Cross‑tile references should be treated as metadata unless new evidence proves rendering dependence.
- “Fully decoded” field sets across all chunks. Status: Overconfident. Several fields remain partially understood; confidence varies by field and dataset.
- `HasGeometry` is a raw field. Status: Incorrect. It is a software-derived convenience from index fields.
- `ReferenceIndexHigh/Low` are distinct raw fields. Status: Incorrect. They are a split view of a single 32‑bit field.

## Open Questions

- Complete field-by-field layouts for some chunks (e.g., MSHD, MDOS/MDSF) remain to be standardized once verified across datasets.
- Exact semantics of some attribute masks and composite keys require further evidence.
- Exact semantics and usage of `MSLK.ReferenceIndex`.
- Precise meaning of MSUR attribute masks/flags across regions.
- Standardized field layouts and linkage for `MDOS/MDSF` building hierarchy where present, including `MDSF → MDOS` relationships.
- Role and gating behavior (if any) of `MPRR` sections over groups of links/surfaces.

## Non‑normative Implementation Notes

- Coordinate parity: Some pipelines invert X for visualization parity. Treat axis flipping as an exporter concern, not a spec requirement.
- Convenience fields such as `HasGeometry` may be computed in software to simplify processing; they are not part of the raw spec.
- Building extraction heuristics: Some pipelines use self‑referencing `MSLK` nodes as grouping separators for building‑scale extraction. This is an implementation heuristic and not a spec requirement.
- Per‑tile exporters may group outputs by a dominant tile computed from surfaces → indices → tile mappings. This grouping is an exporter concern and does not alter raw data interpretation.
- MSCN world‑space normalization and cross‑tile aggregation (e.g., 3×3 grids) are analysis/exporter concerns; do not reinterpret raw PM4 tile boundaries in the spec.
- Duplicate elimination and degenerate‑face validation are quality safeguards implemented downstream; they are not part of the raw format.

## References

- Verified relationship: `MPRL.Unknown4 = MSLK.ParentIndex`.
- Container detection: `MSLK.MspiFirstIndex = -1` denotes container/grouping nodes (no geometry).
- `MPRR.Value1 = 65535` sentinel separators.
- `MSUR.IndexCount` is diagnostic; not a definitive object identifier.
- `MSLK.LinkIdPadding = 0xFFFF`; `LinkIdTileX/LinkIdTileY` are present.
- See also: `PM4-Chunk-Reference.md` for field inventory and byte-level notes.
- See also: `PM4-Errata.md` for dataset-specific caveats and corrections.
