# PM4/PD4 Chunk Families – Working Notes

> These notes capture current speculation and observed evidence about Blizzard’s chunk naming conventions.  They help prioritise reverse-engineering work and should evolve as understanding improves.

## Naming convention hypothesis

Blizzard appears to reserve the **first three characters** of a chunk ID for a subsystem tag:

| Prefix | Sub-system | Confirmed Chunks | Purpose (current theory) |
|--------|------------|------------------|-------------------------|
| `MSV`  | Mesh Surface **Vertex** data | MSVT (vertices), MSVI (indices), MSPV (positions) | Raw vertex/index buffers for render/collision meshes |
| `MSR`  | Mesh Surface **Reference / Relation** | MSRN (normals), MSUR (surface headers) | Per-vertex metadata + triangle-group headers |
| `MSL`  | Mesh Surface **Link**        | MSLK                     | Scene-graph/object linkage table |
| `MPR`  | Mesh **Path Region**         | MPRL, MPRR              | Server path-finding region lookup & range tables |
| `MD`   | Map **Destructible** | MDOS, MDSF, MDB*        | Phased/destructible world geometry (tile 00_00 only) |
| `MSP`  | Mesh Surface **Pointer/Partition** | MSPI                 | Vertex island / cluster indices (under investigation) |
| `MSB`  | Mesh Surface **Bounds**      | MDBI, MDBH, MDBF        | Bounding boxes & flags (supporting MD*) |

This aligns with PD4 (WMO) chunks where the same prefixes re-appear.

> Note: previous drafts speculated about *materials* being referenced in these chunks; closer inspection of the binary shows **no material IDs or paths are encoded at all** (this game build predates CASC file-IDs). Any material handling must therefore come from higher-level WMO data, not PM4/PD4 chunks themselves.

## Evidence snapshots

### MSV group
* Counts in MSVT & MSPV always match; MSVI indices fall within this range.
* MSPV occasionally stores 24-byte stride (XYZ + 3 unknown floats) – extra floats correlate with per-vertex normals.  When stride is 12 bytes, MSRN likely supplies normals.

### MSR group
* **MSRN** entry count == MSPV vertex count in every tested PM4.
* Scaling vector components by `1/8192` produces unit-length normals ≈ MSCN face normals.
* **MSUR** provides per-triangle ranges (`MSVI_first_index`, `Count`), plus 16 bytes of float data – suspected AABB or plane equation.

### MPR group
* **MPRL** maps logical tile coords → region IDs.
* **MPRR** stores per-region bounding ranges (likely min/max tile indices).
* Pairing them reconstructs a fast Region-of-Interest lookup.

### MD group
* Seen only on `development_00_00.pm4`.
* `MDOS` / `MDSF` contain many `Unknown` ints; initial comparison with live objects suggests state bits & surface-flags for phased destruction.

## Next validation steps

1. **MSRN normal proof** – compute cosine similarity against MSVT/MSPV-derived normals.
2. **Surface bounds** – decode MSUR float quad & relate to MSPI vertex island clusters.
3. **Link ↔ Surface mapping** – validate `MSLK.ReferenceIndex` ranges inside MSUR groups.
4. **PD4 alignment** – compare PD4 MSPV/MSRN data to corresponding WMO `MOVT/MONR` vertices & normals.

---
*Last updated*: 2025-07-12 (removed material speculation, renamed MD → Map Destructible Object Set)
