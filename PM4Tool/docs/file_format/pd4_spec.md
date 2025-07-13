# PD4 File Format – Verified Specification (WMO-Scoped Variant)

This document captures **decoded and tested** information for *PD4* files – the per-WMO counterpart to map-tile PM4 meshes.  Where behaviour matches PM4, the row references the PM4 spec; differences are noted explicitly.  Fields or chunks not yet decoded are marked `Unknown`.

> PD4 observations are based on leaked WMO-scoped files dated ~2008 (`StormwindBank_0000.pd4`, etc.).  Implementation status reflects loaders in `WoWToolbox.Core.v2` (2025-07-12).

---

## Global Characteristics
* Little-endian IFF container (same header layout as PM4).
* **Chunk order** differs – PD4 begins with mesh data, omits path-region chunks (`MPRL/R`).
* One PD4 exists **per WMO** instead of per map-tile.

---

## Chunk List (confirmed / observed)

| ID  | Description | Notes | Implementation |
|-----|-------------|-------|----------------|
| MVER | Version header | Same as PM4 (`Version = 4`). | ✅ Implemented |
| MSHD | Scene header  | Identical struct; numeric values differ per WMO. | ✅ Implemented |
| **Vertex / Index Data** | | | |
| MSPV | Vertex positions | **Always 12-byte stride** (`XYZ` only).  Per-vertex normals stored separately in MSRN. | ✅ Implemented |
| MSVT | Texture coordinates | Matches PM4. | ✅ Implemented |
| MSVI | Triangle indices  | Matches PM4. | ✅ Implemented |
| MSPI | Vertex island index ranges | Matches PM4 but counts differ. | ✅ Implemented |
| **Surface Metadata** | | | |
| MSUR | Surface headers | Matches PM4. | ✅ Implemented |
| MSRN | Vertex normals | `C3Vectori`; scale by **1/8192**. Present in all sampled PD4s. | ✅ Implemented |
| MSCN | Exterior boundary vertices | *Not observed* in PD4 samples – WMO collision handled elsewhere. | ⏳ N/A |
| **Scene-Graph / Links** | | | |
| MSLK | Link table | Links mesh islands to WMO doodad sets; structure identical. | ✅ Implemented |
| **Path Region / Navigation** | | | |
| MPRL | Tile lookup table | **Absent** – PD4 is WMO-scoped. | — |
| MPRR | Region ranges | **Absent**. | — |
| **Destructible Geometry** | | | |
| MD*  | Destructible chunks | Not present in any PD4 tested. | — |
| **Unknown / Unimplemented** | | | |
| MSRF | *Suspected* Mipmap / surface flag table – signature found in some PD4s. | Loader stubbed (`TODO`). | ⚠️ Pending |
| MSRT | *Suspected* runtime flags | Signature rare; structure unknown. | ⚠️ Pending |

Legend: ✅ Implemented  ⚠️ Pending  ⏳ Irrelevant/Not encountered  — Absent

---

## Implementation Notes (Core.v2)
* `PD4File` inherits `PM4File`, so any unrecognised chunk is currently **ignored** with a debug log entry – no crash.
* Batch-dump CLI writes CSVs for MSRN & standard chunks; unknown chunks are emitted as raw hex for future analysis.
* OBJ exporter uses MSRN normals directly (no embedded normals in MSPV).

---

## Divergences from wowdev.wiki (historic docs)
* wiki lists `MSPV` as 24-byte; in practice PD4 samples show 12-byte.
* Additional `MSR*` chunks hinted on wiki remain unverified; none appear in sample set.

---

_Last updated: 2025-07-12_
