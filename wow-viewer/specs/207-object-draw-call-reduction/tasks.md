# Tasks: Object Draw-Call Reduction

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

Status key: `[ ]` not started · `[x]` done · `[~]` in progress · `[O]` operator-owned

---

## Phase 0 — Baseline (blocking)

- [x] **T001** Drop fully-faded instances before submission, counted in `FullyFadedMdxCount`, with the
      skip suppressed under `IgnoreDistanceCulling`/`IgnoreVisionConeCulling` so capture flights are
      unaffected. *(Landed 2026-09-02 ahead of this spec; the existing capture-bypass test caught the
      first attempt.)*
- [x] **T002** Surface the faded-instanced count in the frame panel (`of which distance-faded`, shown
      against the instanced total). `GatedOpaqueFadeBelowThreshold` was already displayed; it should
      now read **0** for renderers that support instancing. `FullyFadedMdxCount` is collected but not
      yet surfaced -- it belongs next to the MDX visibility counts rather than the batching block.
- [ ] **T003** Count **distinct visible** model keys per frame (FR-005). The panel's existing figure
      is a *resident* count — spec 202 T102 flagged this and it is still unmeasured. This is the floor
      SC-001 is measured against.
- [O] **T004** Operator: record the fixed-camera baseline — opaque MDX submissions, instanced /
      state-hoisted / unbatched, per-gate breakdown, WMO groups considered/admitted/rejected, stage
      medians.
- [ ] **T005** **GATE**: if `GatedOpaqueFadeBelowThreshold` is a small share of submissions, stop and
      re-plan US1. The 36% figure is a geometric estimate, not a measurement.

## Phase 1 — Doodad instancing (US1) — FR-001..FR-005, FR-010, FR-011

- [x] **T101** Split `_gpuInstanceData` into opaque and faded lists in `ModelRenderer`, keyed on
      `fade >= 0.999`.
- [x] **T102** Remove the silent `return` in `QueueGpuInstance` (FR-004) — accept the instance and
      route it to the correct list. This is a latent defect regardless of Phase 1: an instance
      reaching it past the caller's gate is never drawn.
- [x] **T103** Emit two `DrawElementsInstanced` calls in `EndGpuInstanceBatch`: opaque batch (blend
      off, depth write on), then faded batch (blend on, depth write off).
- [x] **T104** Remove the `visible.OpaqueFade >= 0.999f` condition at `WorldScene.cs:11158` so faded
      instances reach the instanced arm.
- [x] **T105** Record the faded-instanced population as its own submission outcome so it is
      distinguishable from opaque-instanced in the panel (FR-011).
- [x] **T106** Unit-test the batch-splitting decision in `WowViewer.Core*` — the viewer is not
      testable from any test project, so the *decision* must live where it can be.
- [O] **T107** Operator: re-capture the fixed camera. Expect opaque MDX draws ≥10× lower (SC-001) and
      `MdxOpaqueSubmission` median ≤3 ms (SC-002).
- [O] **T108** Operator: pixel-compare the fade band for popping, z-fighting or sorting artifacts
      (SC-004 / acceptance 1.3). If present, sort the faded batch back-to-front by centroid.

## Phase 2 — WMO group admission (US2) — FR-006..FR-008

- [ ] **T201** Split the conservative-fallback counter by reason, separating "no portal data in file"
      from "portal data present, graph not built" (FR-008). Do this **first**: one counter hides two
      defects with opposite fixes.
- [ ] **T202** Evaluate each group's own bounds for frustum visibility rather than inheriting the
      placement's admission (FR-006).
- [ ] **T203** Apply projected-size rejection per group, reusing the existing MDX/terrain thresholds
      rather than inventing new ones.
- [ ] **T204** Ensure group culling runs with no portal graph present (FR-007).
- [ ] **T205** Runtime toggle for group culling (FR-010), defaulting **off** until T207 passes.
- [ ] **T206** Report groups considered / admitted / rejected with the rejecting rule named.
- [O] **T207** Operator: confirm a non-zero rejection rate (SC-004) and walk an interior — no wall,
      floor or ceiling may pop (acceptance 2.4). Expect `WmoSubmission` median ≤15 ms (SC-005).

## Phase 3 — Skin-profile LOD (US3) — FR-009

- [ ] **T301** Surface the M2 skin profile count per model (`nViews`/`ofsViews`), building on spec 193.
- [ ] **T302** Select a profile by projected size, with a single-profile asset rendering unchanged.
- [ ] **T303** Count submitted triangles per frame so SC-008 is readable.
- [ ] **T304** Replace the 80% fade annulus with LOD selection plus a hard cull once T302 holds.
- [O] **T305** Operator: confirm triangle count falls at a fixed distant camera with no silhouette
      change, and that approach transitions do not pop.

## Deferred — explicitly not in this spec

- **`SceneMaintenance`** median 0.02 / max 103.7 ms. A 5000× spread is one rare operation; it needs
  diagnosis before planning. Not a draw-call problem.
- **Spec 204 off-thread decode.** `DeferredAssetLoads` max is 4.1 ms in this scene against 204's
  442.9 ms baseline. Re-measure before reopening.
- **Portal traversal correctness.** Spec 200/151 keep it. US2 only requires that group culling not
  depend on portals.
