# Plan — Epic 249 Renderer Performance, Lighting & Correctness

**Status**: 2026-10-22 bounded R-35/R-38 fidelity planning below; other epic items retain their
existing triage/proof state. Older R-10 sections are historical, not instructions to replay them.

**Active handoff:** [evidence-first refresh](#2026-10-22--active-evidence-first-fidelity-plan).

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

## 2026-10-22 — active evidence-first fidelity plan

**Branch/context:** 286-view-controls-userguide; renderer ownership stays in this epic.
**Inputs:** [dated operator amendment](spec.md), [source research](research.md),
[evidence model](data-model.md), [contracts](contracts/render-fidelity.md), [quickstart](quickstart.md).
This section supersedes stale approach instructions only for the named refresh; no shipped code
or historical receipt is erased. [Registry](../STATUS.md:53) shows later Spec 273 work and WMO
culling quarantine; do not restore old R-10 gates or portal behavior from this plan.

### Operator Intent Check

Plan the requested ground-effect, liquid and sky investigation with available references, then
bounded evidence-backed corrections. Do not claim dense distant effects are a proven opacity
bug, no sky implementation exists, DBC vertex format equals visual material, or decomp tools
are installed. [Spec 259](../259-ground-effects-and-wmo-detail-doodads/plan.md) owns flora;
[Spec 286](../286-view-controls-userguide/plan.md) owns contextual UI. No duplicate task owner.

### Technical Context / project structure

Existing C#/.NET 10, Silk.NET OpenGL, ImGui.NET and xUnit; no new dependency/backend/storage.
Core/Runtime own shared pure policies and evidence contracts, viewer-owned services adapt
existing documents to rendering; UI stays in its existing visual-control owner. Evidence files
live below the owning spec, never a machine-local hardcoded client root. No numerical FPS target
is invented: measure same-scene before/after cost and correctness.

| Existing seam | Bounded candidate responsibility (not automatic write permission) |
|---|---|
| [WMO liquid service](../../src/viewer/WoWViewer/Rendering/WmoLiquidRenderer.cs:292) | Trace existing nibble/group family policy and selected procedural material; isolate a policy adapter if evidence warrants |
| [Standard adapter mapping](../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:2840) | Observe actual DBC family path and fallback; do not expand this oversized file with new features |
| [Vertex-format chain](../../src/core/WowViewer.Core.IO/Dbc/LiquidVertexFormatChain.cs:35) | Read-only reference for layout resolution, not a proposed reader rewrite |
| [Terrain liquid renderer](../../src/viewer/WoWViewer/Terrain/LiquidRenderer.cs:277) | Family palette/material submission after mapping is proven |
| [Atmosphere service](../../src/viewer/WoWViewer/Terrain/Scene/Atmosphere/SceneAtmosphere.cs:357), [lighting service](../../src/viewer/WoWViewer/Terrain/LightService.cs:63), [sky dome](../../src/viewer/WoWViewer/Rendering/SkyDomeRenderer.cs:49) | Exact-build row/model selection, visibility and pass-state trace; owned sky policy/presentation adapter |
| [Light resolver tests](../../tests/WowViewer.Core.Tests/BuildScopedLightDbcProfileResolverTests.cs:7), [liquid chain tests](../../tests/WowViewer.Core.Tests/Dbc/LiquidVertexFormatChainTests.cs:120) | Preserve and extend focused pure-policy coverage, not visual signoff |

### Constitution Check — before and after design

PASS in design: operator-originated scope; existing ownership/library-first; no reader/reference
writes; no app/scene god-class growth; <=10 steps per phase; repo-contained evidence;
real-client visual/performance proof operator-owned. No exceptions or silent base-tool changes.
Unknown empirical reference/appearance results are explicit gates, not unresolved design claims.

### E0 — common evidence and reference gate
1. Choose one exact build/map/asset/camera reproduction per complaint with explicit controls,
   revision, client fingerprint and baseline output; reuse existing telemetry, not a new profiler.
2. Verify configured decomp/reference availability read-only. Record source revision, exact symbol/
   file/build and citation. Treat existing external-root pointers as unverified until actually read.
3. **Gate E:** source ledger distinguishes observed/reference-confirmed/hypothesis/unknown;
   operator provides real reproduction. Missing sources block claims, not permission to invent parity.

### L0 — liquid identity, family, layout and material (no fix yet)
1. Trace one WMO and one MH2O real example for each reported wrong family; include the missing-row
   fallback and mixed-layer case when relevant. WMO group/material values and MH2O object/layout
   values remain distinct. Compare live and harvest paths, not just the easier decoder.
2. Add bounded pure-policy fixtures/tests only after real row/payload facts are recorded. Preserve
   existing missing-row/default/layout tests; identify whether defect is classification, merged
   layer loss, material selection or geometry rather than assuming reader error.
3. **Gate L:** operator approves diagnosis and narrow write set. Reader contradiction routes to
   an explicit verified-format-bug amendment; do not change readers in this lane automatically.

### L1 — one diagnosed liquid correction, then proof
1. Implement only Gate L's selected mapping/material adapter or service correction, opt-in during
   comparison where possible; leave unrelated palette, geometry and terrain-loader behavior alone.
2. Run focused tests/build; operator captures identical camera/build/settings before/after,
   including unaffected water/ocean/magma/slime controls applicable to that client.
3. **Gate LP:** criterion→real-output receipt and operator verdict before promoting the correction.
   If evidence instead demands full textured/animated liquid materials, propose a separately
   bounded amendment; that larger implementation is not implied by this refresh.

### S0 — sky/lighting source-to-pass audit
1. Trace exact-build Light/LightParams/bands/LightSkybox and fallback selection into atmosphere,
   model load and sky passes for fixed day/night samples; record existing night-visibility gate.
2. Compare observed behavior with verified same-era reference symbols; enumerate proven-good
   database choices and blank/default mode without treating schema presence as client row data.
3. **Gate S:** approve one mismatch and bounded service/policy write set; no blanket renderer,
   camera, local-light or portal rewrite.

### S1 — narrow sky correction and existing-owner controls
1. Implement only Gate S's proven selector/visibility/pass correction in owned services; retain
   current simple blank-sky mode and unchanged baseline comparison path.
2. Expose proven choices through the existing visual-quality/lighting surface and View route,
   with shared UI widgets/inventory entry; no second authoritative control home.
3. **Gate SP:** focused resolver/mesh/policy tests + operator same-camera day/night/blank/known-good
   screenshots and lighting checks; preserve unaffected legacy/default behavior. No visual claim
   from tests/build alone.

### Phase 2 handoff and order

Generate new unchecked tasks only after this plan: E0 → L0 → L1 and E0 → S0 → S1, one validated
phase at a time. Spec 259 follows its own G gates; UI U0 can proceed independently. Do not replay
Spec 273/276 work or old epic-wide triage as a prerequisite to this bounded evidence pass.
Do not mark historical or new checkboxes complete from planning. Exact checks/receipts are in
[quickstart](quickstart.md). No runtime, build or test execution is claimed by this document.
