# gillijimproject_refactor archive - pre wow-viewer reset

Archived on 2026-07-04 while trimming root continuity to reference-only truth.

## Terrain-model history

- 2026-06-19: V21 pivot kept terrain mesh route on proven V18 single-head height model with filtered-mask loss.
- V19 and V20 were explicitly abandoned as over-built detours.
- Key diagnosis then: loss gating ignored intended filtered object mask until dataset and trainer fixes landed.

## PM4 history

- 2026-06-17: surface-correlation matching replaced old hull-footprint matching.
- Key proof then: 1604 PM4 fingerprints vs 2790 WMO surface fingerprints, `P@1 = 1.3%`, `P@3 = 10.3%`, false positives like Ironforge and Darnassus removed.
- Main open gap then: WMO DB coverage and dev-map ADT reliability.

## Why this moved out

- Root continuity had become stale and duplicated active `wow-viewer` truth.
- Live root files now say only what this repo still owns: reference behavior, data paths, and archaeology.
