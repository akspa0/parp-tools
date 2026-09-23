# Epic 249 — Renderer Performance, Lighting & Correctness

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 24 archived specs (split specs also contribute items; see the ledger). **No new scope**: every backlog item is scope a
> source spec already stated, cited by id (§9.1). Evidence: [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md),
> audits [C1](../archived/reconciliation-2026-09-23/audit/batch-C1.md) ·
> [C2](../archived/reconciliation-2026-09-23/audit/batch-C2.md) ·
> [G1](../archived/reconciliation-2026-09-23/audit/batch-G1.md) · [G2](../archived/reconciliation-2026-09-23/audit/batch-G2.md) ·
> [E](../archived/reconciliation-2026-09-23/audit/batch-E.md).

## Goal

Correct images at high, measured FPS on legacy **and** modern data, with every improvement
attributable to a before/after receipt.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Source |
|---|---|
| In-viewer rolling frame history + CPU stage attribution (`profile-render`, `WorldRenderDiagnostics`) | 152 Ph 0/1, 153 |
| Audio-stall scoping; MDX GPU instancing restored; hitch findings measured | 153 Ph 0–3 |
| Doodad rendering optimizations (code landed, parity proof owed) | 136 |
| M2 vs MDX render-path metric split (Phase 1) | 201 |
| Batching measurement infrastructure (Phase 0/1); opaque/faded instance split (landed under both 202 T302 and 207 Ph 1) | 202, 207 |
| Scene graph (`WorldSceneGraph`, traversal) + bounded tile admission (`DirectionalTileSelector`, `CameraTileWindowSelector`); graph selector **default-off** after a real-client A/B showed it slower; sky-dome sun/moon; LightSkybox/Stars | 142 |
| Minimap interaction state + Armed triple-click teleport on both surfaces | 147 US1 (137) |
| Quick lighting/fog controls, `FogEnd+2500` far plane, single-ray hover cards | 107 |
| Native 0.5.3 world-light direction constants; separate solar/lighting math | 106 |
| MDX/M2 shading fix + Half-Lambert; scene lights (`SceneLightManager`) from WMO `MOLT`/doodad emitters onto WMO, terrain (both shader programs) and unbatched doodads; ambient/sun contract | 236 Ph 1/2 (T010–T012) |
| Portal-aware rendering slice | 151 US1 |
| Feature-tour capture recipe (path warmup + direct framebuffer capture) | 233 US1 |

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Measurement (makes every other item provable)

| ID | Item | Source | Today |
|---|---|---|---|
| R-01 | Modern-data path-driven renderer benchmark with the legacy receipt shape | 246 FR-005 | absent (`inspect casc bench` measures reads, not frames) |
| R-02 | `MD21` CASC M2 camera tracks as playable paths (find the loss point first) | 246 FR-002 | absent |
| R-03 | Capture receipt model (completed/degraded/failed with hitch/FPS evidence) + authoring handoff / MCP transport | 233 US2 T016–T020, US3 T021–T025 | absent (type stubs only) |
| R-04 | GPU/driver timer-query attribution | 142 T048; 150 | absent |
| R-05 | M2/MDX metric close-out flight + native-M2 batch key if warranted | 201 T005/T006, Ph 2 | Phase 1 only |

### B. Draw calls & batching

| ID | Item | Source | Today |
|---|---|---|---|
| R-10 | Per-placement WMO shell instancing under scene lights (v0.6 FPS regression: ~5.5 FPS, 16,431 WMO draws on `wow_classic_beta` Azeroth) | 242 | whole-scene gate disables instancing |
| R-11 | Route light-affected opaque placements off GPU instancing onto per-placement lit draws | 236 T013 | boundary documented |
| R-12 | Unified model batching planner (`ModelBatchKey`/`ModelBatchPlanner`), native-M2 batching, era coverage, transparent batching | 202 Ph 2–6 | planner absent |
| R-13 | Doodad batch planning (compatibility key) + batch diagnostics | 147 US3/US4; 148 Ph 4 | absent |
| R-14 | Doodad instancing overhaul | 236 Ph 3 | not started |
| R-15 | WMO group admission in dense interiors (Stormwind: 7,512 groups / 80,484 draws, 100% of recorded hitches) + eliminate conservative portal fallback | 200 US1–3; 207 US2; 153 handoff; 151 T030 | measured, unfixed |
| R-16 | Skin-profile LOD | 207 US3 | absent |

### C. Streaming & residency

| ID | Item | Source | Today |
|---|---|---|---|
| R-20 | Off-thread asset decode (MoP flight: 26–68 ms hitches, 2047/2048 frames over budget) incl. async MDX decode | 204; 153 Ph 5 step 2 | absent |
| R-21 | Fog-bounded tile residency (`fogEnd` as admission radius) with residency-lease attribution | 147 US2; 148 Ph 3; 142 T054–T056 | absent — fog is render-only |
| R-22 | Camera as one authoritative world actor feeding render/audio/collision/residency | 148 Ph 2 | absent |
| R-23 | Ordered per-pass visibility lists; shared spatial/picking queries; modern instanced submission | 142 US4 T020–T022, US5 T023–T025, T057 | absent |

### D. Correctness & lighting

| ID | Item | Source | Today |
|---|---|---|---|
| R-30 | Wireframe slope-scaled bias / MDX-WMO wireframe pass; material-gated specular | 226 US1/US2 | hypothesis confirmed in code |
| R-31 | `WmoRenderer.DrawBatch → GL.DrawElements` native access violation (4 repro runs) | 138 WMO-doodad slice | unresolved |
| R-32 | Per-era terrain lighting (1.0.0+ brightness vs native) | 152 Ph 6 | not started |
| R-33 | WMO/MDX lighting-selection contract wired into `WmoRenderer`/`M2Renderer` | 143 US4 | LIT/clock plumbing done; wiring absent |
| R-34 | Native day/night coordinate-transform calibration + evidence record | 106 FR-004/FR-012 | absent |
| R-35 | Skybox rendering (full spec) | 160 | absent |
| R-36 | M2/WMO shader permutation system | 198 | absent |
| R-37 | Diagnostic render profiles | 151 US5 | absent |
| R-38 | Liquids (ocean and other layers) render with grid lines / omissions — root cause not established; needs a zoomed capture + exact map/era first | operator report, activeContext 2026-09-18 | open defect |

### E. Parked

| ID | Item | Source |
|---|---|---|
| R-90 | Shared `WowViewer.Core.Renderer` promotion / host cutover (library still a headless-tools skeleton) | 056 |
| R-91 | Broad 4.x renderer evidence epic (19 modules) | 138 |

## Operator verification owed on shipped code

136 T008/T011 (parity + I/O); 142 real-client promotion gates (T027–T029, T069, T073, T077);
152-phase-6 gate; 153 SC-007 smoothness; 233 T015 Feature Tour + video witness; 236 T003 + Gate 2
(torch/brazier spill onto WMO, terrain, doodads); 226 captures.
