# Open Questions & Unknowns (2025-07-12)

This page tracks **unanswered questions** and suspected data we may be ignoring in the current implementation.  Each item should spawn either an investigation task or a spec update once resolved.

| Area | Question | Observations / Leads | Priority |
|------|----------|----------------------|----------|
| MSPV stride-24 | What precise purpose do the extra 3 floats serve? Duplicate normals or something more? | Values match `MSRN / 8192` in all sampled tiles. Present only when `MSRN` chunk absent. | 🔶 Medium |
| MSRN in PM4 | Are there PM4 tiles that actually contain `MSRN`? | None found so far (`development_22_18`, `development_00_00`). Could be PD4-only. | 🔶 Medium |
| MSHD fields | 8× `uint32` remain unnamed. What do they control (spatial extents? tile metadata?) | Values are stable across all development tiles. | 🔷 Low |
| MSUR AABB | Four floats after `StartIndex/Count` look like bounding box mins/maxs but mapping unknown. | Used by exporters for debugging only. | 🔶 Medium |
| MSPV extra floats vs OBJ exporter | Should exporter prefer embedded normals when present and MSRN absent? | Would avoid costly recomputation and match authoring intent. | 🔷 Low |
| Unparsed MSR* chunks | Do chunks like `MSRF`, `MSRT`, etc. exist? If so, what semantics? | Search PD4 samples for additional `MSR` signatures. | 🔴 High |
| MDOS/MDSF | Full semantics of destructible object sets and state frames. | Only tile `00_00` contains these chunks. Need correlation with WMO phasing data. | 🔷 Low |
| PD4 only chunks | Confirm list of PD4-specific chunks beyond `MSRN`. | PD4File inherits PM4File; we may silently drop unknown PD4 chunks. | 🔴 High |
| MSPR / Path Regions | Relationship between `MPRL` lookup and `MPRR` ranges – full graph semantics? | Critical for inter-tile navigation. | 🔴 High |
| MSCN usage | How does the exterior boundary relate to WMO collision or AOI culling? | Requires overlay with in-game boundaries. | 🔶 Medium |

Legend: 🔴 High  🔶 Medium  🔷 Low

_Add new questions as they arise and strike-through resolved items with a brief answer + commit reference._
