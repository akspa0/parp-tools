# Plan — Epic 250 Map Reconstruction, Composition & Editor Platform

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies between backlog items

```text
E-10 applier completion ──> E-11 one undo/save path ──> E-01 map save ──> E-02 project persistence
                                                   └──> E-12 clipboard migration, E-13 add-placement
E-14 journal ──> after E-11
E-16 integrity census ──> E-16 repair patterns
E-01 map save ──> E-17 tile creation, E-33 WMO writing
E-20 uniqueId reconciliation (independent)
```

E-01 consumes the existing writers (`LkAdtWriter`, `LkWdtWriter`, `AlphaWdtWriter` — frozen, §4) and
is shared with Epic 248 F-03 (modern→legacy conversion writes through the same targets).

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| E-01, E-04 | [234 spec](../archived/234-map-save-new-map/spec.md) · [236 plan](../archived/236-scene-lighting-doodad-performance/plan.md) (Phase 5) |
| E-02, E-25–E-27 | [232 plan](../archived/232-cartography-composition-project/plan.md) · [data-model](../archived/232-cartography-composition-project/data-model.md) · [contracts](../archived/232-cartography-composition-project/contracts/) |
| E-10–E-17 | [old editor-platform epic](../archived/epic-editor-platform/epic.md) and its member specs 166–177 |
| E-18 | [176 plan](../archived/176-object-transfer/plan.md) |
| E-20 | [203 spec](../archived/203-multi-phase-map-composition/spec.md) |
| E-21–E-23 | [219 plan](../archived/219-phase-layer-rotation/plan.md) · [222 plan](../archived/222-map-composition-workbench/plan.md) |
| E-30, E-31 | [236 spec](../archived/236-scene-lighting-doodad-performance/spec.md) US4 · [191 plan](../archived/191-procedural-garden-museum-generator/plan.md) |
| E-32 | [230 spec](../archived/230-reconstruction-editor/spec.md) |
| E-33 | [220 plan](../archived/220-wmo-doodad-editing/plan.md) |

## Standing constraints

- Never write into a client install tree; outputs go to a workspace/operator path (§9.3).
- `AlphaWdtWriter` is frozen (§4). New editor features are owned services, not `ViewerApp` members (§10).
- Client-load witnesses (3.3.5 client, Noggit, Alpha 0.5.3) are operator-owned.
