# wow-viewer Format Parity Matrix

This matrix tracks the full-ownership gap between the active `MdxViewer` format surface and the current `wow-viewer` library/tooling surface.

Use it as the working backlog for the Mar 28, 2026 full-format ownership reset.

## Status Legend

- `none`: no real owned seam yet
- `summary`: detector or top-level summary only
- `partial`: some deep ownership exists, but active viewer behavior still depends on old code or third-party ownership
- `strong`: most active behavior is library-owned, but cutover/write/runtime parity is still incomplete
- `complete`: first-party ownership is the real source of truth for active behavior

## Matrix

| Family | Active viewer reality today | Current wow-viewer ownership | Key gaps to closure | First full-ownership milestone |
| --- | --- | --- | --- | --- |
| `ADT` root | Core terrain loading, chunk mesh build, alpha decode, liquids, placements | `summary` | `MCAL`, `MCLY`, texture/shadow/liquid payload semantics, write path, runtime services | library-owned `MCAL` + `MCLY` + root payload reader used by both viewer and converter |
| split `ADT` `_tex0` | texture/layer payload source for later builds | `summary` | full chunk routing, texture payload read/write ownership | shared split-file routing and paired file resolution |
| split `ADT` `_obj0` | doodad/WMO placement source for later builds | `summary` | placement payload ownership, write path, shared paired-file routing | shared `_obj0` reader/writer contracts consumed by converter and app |
| split `ADT` `_lod` | later-build terrain LOD companion | `none` | detection, reader, writer, runtime service surface | shared `_lod` file-kind and top-level reader |
| `WDT` | map bootstrap, tile occupancy, build flags, map-type routing | `summary` | full root payload ownership and write semantics where active tools depend on them | shared WDT payload reader beyond MPHD/MAIN summary |
| `WDL` | preview-only far terrain path | `none` | first-party parser and inspect/tool surface | shared WDL reader with inspect coverage |
| `WMO` root | runtime rendering and conversion depend on deep root semantics | `partial` | material semantics, texture ownership, portal/liquid/topology behavior, write parity | deepen root contracts from summary to payload detail for material and texture tables |
| `WMO` group | runtime group rendering and conversion depend on deep payloads | `partial` | standalone group ownership, liquid detail, visibility/topology behavior | shared standalone group reader/detail contracts consumed outside inspect |
| `WMO` embedded groups | Alpha monolithic roots already partially surfaced | `partial` | full parity with standalone group behavior and converter/runtime consumers | reuse one owned group payload model across embedded and standalone paths |
| `MDX` | full model parse, animation, bones, geosets, materials, emitters | `summary` | deep chunk parsing, animation tracks, runtime model contracts, write/export seams | shared deep `MDX` reader past `VERS`/`MODL`/`TEXS`/`MTLS` |
| `M2` | active viewer depends on Warcraft.NET adapter behavior | `none` | first-party parsing, runtime contracts, build/version routing, texture/material ownership | shared `M2` inspect reader and core chunk contracts |
| `BLP` | active viewer depends on pixel decode and mip use | `summary` | first-party palette/JPEG/DXT decode, mip write, export parity | shared decode service for `BLP1`/`BLP2` pixel paths |
| `PM4` | active overlay/alignment/correlation logic still partly viewer-local | `strong` | remaining semantic extraction, restore-facing services, final consumer cutover | continue `Core.PM4` extraction until `WorldScene` no longer owns active semantics |
| `DBC` / `DB2` | active viewer and converter consume a narrow subset | `partial` | breadth, schema-backed ownership, runtime service consolidation | explicit inventory of actively consumed tables and first-party access seams |
| minimap translation/support | active viewer uses md5 translate and tile cache/runtime helpers | `partial` | shared runtime seam for translation and file lookup used by app/tools | move minimap translation/runtime ownership fully under shared runtime/services |

## Program-Level Priority

1. `ADT` root + split `ADT`
2. `WMO` root/group
3. `MDX`
4. `M2`
5. `BLP`
6. `PM4` continuation
7. `DBC` / `DB2` breadth
8. `WDL`
9. minimap translation/runtime auxiliary seams

## Notes

- `PM4` ranks below terrain/model ownership only because it already has a real library home and meaningful extraction momentum.
- `BLP` and `M2` remain strategic blockers because the active viewer still relies on third-party ownership there.
- `WDL` is lower priority because the active viewer uses it narrowly today, but it is still in scope for full ownership.