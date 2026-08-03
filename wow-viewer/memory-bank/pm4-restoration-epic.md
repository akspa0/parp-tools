# PM4 Restoration Epic — pointer

Recorded: 2026-08-03. Status: three specs written, implementation not started.

**Full epic doc**: `specs/epic-pm4-restoration/epic.md` — read that before touching PM4 work.

## One-line summary

Restore object placements for a 4.0.0-era development map whose placements are lost, by decoding
what PM4 actually is: the NEGATIVE BSP — the walkable surfaces every placed object carves out.

| spec | branch | what |
|---|---|---|
| 130 | `130-pm4-remaining-decode` | undecoded fields, connective geometry, object identity |
| 129 | `129-pm4-zarr-dataset` | stored queryable form at map/tile/object levels |
| 128 | `128-pm4-negative-bsp-matching` | match on negative-space structure, not aggregate scalars |

Branches are stacked linearly; `130` contains all three. Order: 130 -> 129 -> 128.

## The three things most likely to be got wrong

1. **The PM4 C# stack already exists and is the oldest part of the codebase** — 13 research
   analyzers, 14 services, a 30-command CLI under `WowViewer.Tool.Inspect pm4`. Never hand-parse
   PM4 chunks. A Python parse outside `Pm4CoordinateService` during spec research produced a
   confident and entirely wrong conclusion; the axis order differs BETWEEN CHUNKS OF ONE FILE
   (MPRL's third axis is MSVT's first).
2. **The viewer selecting a surface instead of an object is not a UI bug.**
   `MSLK.GroupObjectId -> MPRL.Unk04` is 65,819 fits / 1,206,977 misses (~5%). Surfaces are the
   largest unit the decode can justify.
3. **CK24 = 0 spans 291 tiles and is a null sentinel**, not an object. 1,229 distinct keys, 266
   (21.6%) genuinely span 2+ tiles — which is why the dataset is object-primary, not tile-keyed.

## Leading lead

MSPV/MSPI is a second geometry stream LARGER than the decoded surface mesh (2,418,205 vs 1,930,146
index fits) attached to the same link records, with its window meaning unresolved
(indicesOnly=399,183, both=199,699, trianglesOnly=0). Right shape for the connective geometry that
seals a surface set into a negative volume. A lead, not a premise.

MPRR is the largest undecoded surface in the format (327,744 bytes in tile 0_0); neither candidate
target domain explains it.

## Related

- [[weak-signal-tile-archaeology]] — the terrain-side sibling, complete and parked. Same discipline:
  measure with the project's own tooling, verify detector power before trusting a null.
