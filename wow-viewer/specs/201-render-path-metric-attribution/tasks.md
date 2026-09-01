# Tasks: Separating M2 and MDX Render-Path Metrics

**Spec**: [spec.md](spec.md) | **Created**: 2026-09-01

Phase 1 changes no rendering. That is deliberate: it makes the existing numbers readable
before anything is optimised on the strength of them.

Legend: `[ ]` open · `[x]` done · `[-]` in progress · **(operator)** = user-run.

---

## Phase 1 — Attribute, change nothing

- [ ] **T001** Add per-render-path opaque/transparent submission counters to
      `WorldRenderFrameStats`, keyed by applied `M2RouteType` (FR-001, FR-002). Keep the
      existing aggregate fields so FR-005's sum check is testable.
- [ ] **T002** Record submission outcome per instance as batched / unbatched / **unbatchable**,
      with the reason for unbatchable (FR-003, FR-004). `M2Renderer.RequiresUnbatchedWorldRender`
      and `SupportsGpuInstancedOpaque` already carry the signal — surface it rather than
      re-deriving it.
- [ ] **T003** Unit-test FR-005: per-path counts sum exactly to the aggregate totals. This is
      what proves the change is a decomposition and not a redefinition.
- [ ] **T004** Update the frame-history panel to show the per-path breakdown, and stop labelling
      M2-routed models as `MDX` in the panel and the status bar (FR-006).
- [ ] **T005** **(operator)** Re-fly the Wandering Isle route with batching **off**. Record the
      per-path split of the 13,180 unbatched opaque draws. **This number decides whether Phase 2
      is worth doing** (SC-002).
- [ ] **T006** **(operator)** Confirm FR-007: frame-time distribution for the same route is
      unchanged by adding attribution (SC-004).

**Phase 1 exit**: "100% unbatched" resolves into per-path counts, and the share that is
structurally unbatchable is a number.

---

## Phase 2 — Native-route M2 batching

**Gated on T005.** If native-route M2 is a small share of the unbatched count, record that and
stop — the spec's job was to find out.

- [ ] **T101** Design the backend-specific batch key the `M2Renderer` comment names as missing.
      The native runtime backend owns a different shader/state path, so the key must include
      whatever state makes two instances co-submittable.
- [ ] **T102** Implement opaque batching for native-route M2 behind a runtime toggle (FR-008),
      mirroring spec 153 US3 so both can be compared in one session.
- [ ] **T103** Verify appearance is unchanged versus the unbatched path on a reference scene.
- [ ] **T104** **(operator)** Re-fly the route. SC-005: per-path unbatched counts for
      native-route M2 fall; frame time recorded.

---

## Notes

- Do **not** re-open spec 153 US3's batching for legacy-backed models. This spec adds the path
  the native backend lacks; 153 keeps the one it has.
- Attribution must use the **applied** route. A model whose primary route failed and fell back
  drew on the fallback, and that is what the metric has to say.
- `MDX 13180/20115 (429 ok/0 fail)` — the gap between placed and drawn is large and unexplained.
  Not this spec's scope, but worth a look while the counters are open.
- Known test baseline: `WowViewer.Core.Tests` has **9 pre-existing failures**. Compare to 9.
