# Plan — Epic 249 Renderer Performance, Lighting & Correctness

**Status**: Implementation approach **not yet selected** (operator directive 2026-09-23: triage first).

## Dependencies between backlog items

```text
R-01 modern benchmark ──> R-10 per-placement WMO instancing ──> R-11 lit placements off instancing
                                                              └─> R-14 doodad instancing
R-12 batching planner ──> R-13 doodad batch planning ──> R-14
R-05 metric close-out ──> R-12 native-M2 batch key
R-15 WMO admission (independent; largest measured hitch source)
R-20 off-thread decode ──> R-21 fog residency ──> R-22 camera actor
R-03 receipt model ──> every "before/after" receipt in this epic
```

Duplicate tracks merged here: 136/138/153/202/207/236 all touched the same `ModelRenderer`/`WmoRenderer`
batching paths (the opaque/faded split landed twice); 147 and 148 described the same residency and
batching gap under different names; 150 was superseded by 152, which spun off 153.

## Design documents adopted by reference

| Item | Adopted design |
|---|---|
| R-01, R-02 | [246 spec](../archived/246-modern-m2-camera-paths-and-benchmarking/spec.md) |
| R-03 | [233 plan](../archived/233-marketing-capture-automation/plan.md) · [contracts](../archived/233-marketing-capture-automation/contracts/) |
| R-10 | [242 spec](../archived/242-wmo-instancing-performance/spec.md) |
| R-11, R-14 | [236 plan](../archived/236-scene-lighting-doodad-performance/plan.md) |
| R-12 | [202 plan](../archived/202-unified-model-batching/plan.md) · [research](../archived/202-unified-model-batching/research.md) |
| R-13, R-21 | [147 plan](../archived/147-minimap-fog-instancing/plan.md) · [148 plan](../archived/148-world-simulator/plan.md) |
| R-15 | [200 spec](../archived/200-wmo-portal-admission-fallback/spec.md) · [207 plan](../archived/207-object-draw-call-reduction/plan.md) · [153 research](../archived/153-renderer-hitch-and-batching/research.md) |
| R-20 | [204 plan](../archived/204-off-thread-asset-decode/plan.md) · [research](../archived/204-off-thread-asset-decode/research.md) |
| R-23 | [142 plan](../archived/142-world-scene-graph/plan.md) |
| R-31 | [138 WMO-doodad batching slice](../archived/138-cataclysm-renderer-evolution/wmo-doodad-batching-slice.md) |
| R-35 | [160 plan](../archived/160-skybox-rendering/plan.md) |
| R-36 | [198 plan](../archived/198-m2-shader-permutations/plan.md) |

## Standing constraints

- No new members in `WorldScene` (17,153 lines) or `ViewerApp` (16,746 lines) — AGENTS.md §10.
  Every item lands in an owned service class.
- Every performance claim needs a before/after pair from the same path, build and map (R-01/R-03).
- The profiler's static camera cannot prove a null result (memory: renderer profiler is blind).
