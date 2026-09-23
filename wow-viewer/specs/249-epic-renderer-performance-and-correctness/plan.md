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

## Approach — R-10 (operator P1, 2026-09-23; pending scope approval)

Order: measure → cheap exact fixes → re-measure → decide whether the per-instance-light path is needed.

| Step | Change | Owner (no god-class growth) | Proof |
|---|---|---|---|
| R-10a | Frame counters: lights collected, `QueryAffecting` calls + ms, WMO placements batched / lit-fallback / self-lit | existing `WorldRenderDiagnostics` + frame-history counters | operator baseline capture on 1.60.1 Azeroth |
| R-10b | `WorldWmoOpaqueBatchCandidate` gets a lit flag; `PlanOpaqueWmoBatches` batches only unlit placements; batch path uploads 0 local lights | `Core.Runtime/World/Passes/WorldObjectPassCoordinator` (+ `WorldObjectPassCoordinatorTests`); one-line call-site change at `WorldScene.cs:11415` | unit tests: deterministic partition (242 FR-004) |
| R-10c | *(spec-synced 2026-09-23)* Visible MDX is collected after the WMO pass, so the rebuild instead keeps only lights whose sphere (+256 margin) touches the view's side/near planes — exact for drawn pixels | existing method body + `FrustumCuller.TestSphereIgnoringFarPlane` | kept/collected counters |
| R-10d | Uniform-grid spatial index built once per frame in `SceneLightManager`; `QueryAffecting` returns the same lights in the same order | `SceneLightManager` (owned service); tests in a new test file | equivalence tests vs the linear scan |
| R-10e | *(conditional)* per-instance light sets for instanced WMO shells | `WmoRenderer` + shader | only if R-10a re-measure says self-lit WMOs dominate |

Risk: R-10b changes which WMOs light up only if a light reaching a placement was previously dropped —
it must not be. Batched placements are exactly those no light reaches, so their shading is unchanged.
