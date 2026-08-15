# Implementation Plan: Renderer frame-time stability, flattened pipeline, and focused view modes

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-15 | **Spec**: [spec.md](spec.md)

## Summary

v0.5.3 makes the renderer render properly. Three defects share one root: the renderer treats the
world as a tree of discrete per-object nodes walked every frame, instead of retained ordered lists
of draw work processed in passes. That produces the gallop (per-frame allocation and rejection cost),
the universal-scene cost (every overlay always live), and the UI sprawl (every surface reachable from
everywhere, with no owner).

The approach is deliberately ordered so that nothing is optimized on a guess:

1. **The viewer records its own behavior over time**, and automated tests assert on that record. This
   is the #1 deliverable. External CLI profiling was the previous approach and did not surface this;
   the detector has to live in the running viewer. The instrumentation already exists — 18 per-stage
   timers per frame — and is discarded every frame because nothing retains it.
2. **Baseline and attribute** the real hitch pattern, and measure which churn surfaces dominate.
3. **Kill the churn** — small, local, revertible fixes to structures that rebuild per frame or hold
   caches with a one-frame lifetime. Possibly a large share of the win at a fraction of the risk.
4. **Flatten** the per-frame path into retained ordered draw lists and explicit passes, measured
   against an already-cleaned baseline so its own contribution is honest.
5. **Scope** scene construction to a focused view mode.
6. **Fix era lighting** (independent; can run in parallel from day one).
7. **Give every UI surface one owner**, following the view-mode structure.

Every phase is independently revertible and leaves the viewer runnable. Phases 3 and 4 must still
earn their place after Phase 2 is measured; the architecture rewrite is not assumed.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET.OpenGL (current backend), ImGui.NET (viewer shell)

**Storage**: N/A for this feature; measurement reports are JSON on disk

**Testing**: xUnit for pure contracts; **in-viewer rolling runtime-stat record as the primary
measurement**, driven by automated camera-path runs and asserted on directly; external CLI profiling
retained only as one driver of that same recorder, not as a separate measurement system;
maintainer-owned interactive proof for visual sign-off

**Target Platform**: Windows x64 primary; Linux/macOS via the cross-platform viewer target

**Project Type**: Desktop viewer + CLI tooling

**Performance Goals**: Frame-time *stability* first — eliminate periodic hitches during camera
movement. Absolute FPS is secondary and machine-specific. Concrete thresholds are set from the
Phase 1 baseline, not guessed here.

**Constraints**: No FPS or performance claim from build output alone. No optimization accepted
without before/after measurement from a harness proven able to detect the defect. Viewer must remain
runnable at the end of every phase.

**Scale/Scope**: `WorldScene.cs` ~15,456 lines; `ViewerApp*.cs` ~35,685 lines with 214 `Draw*`
methods; scene graph ~2,742 lines across 12 files; resident tile sets up to 25 ADT tiles, each an
isolated graph.

## Constitution Check

*GATE: must pass before Phase 0, re-check after each phase.*

| Principle | Status | Note |
| --- | --- | --- |
| III. Real-Data Validation | PASS | All measurement runs against staged real clients at `H:\CLIENTS`; no synthetic-only claim of success. |
| VI. No Game Client Path Assumptions | PASS | Client roots stay configuration; trajectories name a client/build/map as data, never hardcoded. |
| Read-Only Reference Codebase | PASS | No work in `gillijimproject_refactor/`. |
| One Phase at a Time | PASS | Phase gates are explicit below; each ends at a checkpoint. |
| Bite-Sized Plans | AT RISK | This feature is large. Mitigated by hard phase gates, per-phase revertibility, and a rule that Phase 2 and Phase 4 land one reversible change at a time. |
| Spec Docs Are Source of Truth | PASS | Spec 152 owns requirements; this plan owns sequencing only. |

**Complexity note**: this plan touches the renderer, the scene representation, and the UI. That
breadth is justified because the three symptoms share one cause, and fixing them separately would
mean rewriting the same call paths three times. The risk is controlled by phase gates, not by
narrowing scope to something that would not fix the defect.

## Project Structure

### Documentation (this feature)

```text
specs/152-renderer-frame-stability/
├── spec.md              # Requirements (written)
├── plan.md              # This file
├── research.md          # Phase 1 output — hitch mechanism, churn costs, Ghidra lane
├── data-model.md        # Phase 3 output — draw list / pass / view-mode entities
├── quickstart.md        # How to run a measurement and read a report
├── contracts/           # Frame-record, camera-path, era-profile, draw-list contracts
└── checklists/
    └── requirements.md  # Spec quality checklist (written)
```

### Source code touched

```text
wow-viewer/
├── src/core/WowViewer.Core.Runtime/World/
│   ├── WorldRenderFrameStats.cs             # Exists: TotalCpuMs + 18 stage timers, already per-frame
│   ├── (new) WorldRenderFrameHistory.cs     # Phase 0: fixed-capacity ring buffer + percentiles
│   └── SceneGraph/
│       ├── WorldSceneTraversal.cs           # Phase 2: diagnostics behind a switch; buffer reuse
│       └── WorldSceneNode.cs                # Phase 4: retained list representation
├── src/core/WowViewer.Core.Renderer/Scene/
│   └── SceneRenderer.cs                     # Phase 4: pass-oriented submission
├── src/viewer/WoWViewer/Terrain/
│   ├── WorldScene.cs                        # Phase 0 record hook; Phase 2 churn (C3–C6); Phase 4/5
│   ├── TerrainRenderer.cs                   # Phase 4: pass participation
│   └── TerrainLighting.cs                   # Phase 6: era profile selection
├── src/viewer/WoWViewer/
│   ├── ViewerApp_Investigation.cs           # Phase 0: in-viewer history view (US1b)
│   └── ViewerApp*.cs                        # Phase 7: single-owner routing
└── tools/validation-capture/WowViewer.Tool.ValidationCapture/
    └── ProductionWorldSceneProfiler.cs      # Phase 0: retargeted to drive the in-process recorder
```

**Structure Decision**: the record lives in `Core.Runtime` beside the existing
`WorldRenderFrameStats`, so the viewer, the tests, and any CLI driver all read one measurement
system rather than three. This is the correction from the previous approach, where the external
validation-capture tool was its own parallel measurement path. The flattened representation belongs
next to the current scene graph so both can coexist behind a switch during migration. Era lighting
stays in the viewer's terrain layer where the current model lives.

---

## Phase 0 — In-viewer runtime stats over time, and automated tests on them (US1, US1b)

**Gate to enter**: none. This is first, always. **This is the #1 deliverable of v0.5.3.**

**Why this shape**: the previous approach was external CLI profiling — spin up a separate short-lived
process, render a handful of frames, dump JSON. That has already been tried and did not surface this
defect. The detector must live where the defect lives: inside the running viewer, accumulating over a
real session.

The important discovery is that **the instrumentation already exists and is thrown away**.
`WorldRenderFrameStats` already carries `TotalCpuMs` plus 18 per-stage timers, produced every frame.
`LastRenderFrameStats` keeps exactly one frame; there is no history anywhere in the codebase. So this
phase is mostly *retention and analysis*, not new instrumentation.

**Work**

1. **Rolling frame history in-process.** A fixed-capacity ring buffer of `WorldRenderFrameStats`,
   always recording, sized to cover periodic behavior (several seconds of frames). Fixed memory,
   zero per-frame allocation — the recorder must not become churn surface C8.
2. **Statistics over the window**: median, max, p95, p99, and over-threshold counts, for total frame
   time *and* each of the 18 stage timers. Per-stage percentiles are what turn "a hitch happened"
   into "the hitch was in terrain upload".
3. **Hitch detection and marking** against a threshold, retaining the frame index and dominant stage.
4. **Injected synthetic stall** of known magnitude at a known frame — the detector-power check. If the
   record does not flag it correctly, the detector is not trusted and nothing proceeds.
5. **Camera path driving for automated tests.** Defined paths, at minimum one tile-boundary crossing
   and one continuous heading change, plus a stationary control that is explicitly labelled as such.
6. **Assertable + exportable record**, stamped with client root, build, map, path, and frame counts,
   so before/after comparisons are only made between comparable runs.
7. **Noise floor**: repeat identical runs, report run-to-run spread; nothing smaller than this is ever
   called an improvement.
8. **Recorder overhead measurement**, reported with every record.
9. **In-viewer history view** (US1b): recent frame times over time with hitches marked and per-stage
   breakdown on inspection. Recording continues whether or not the view is open.
10. **Retarget the external CLI path** to drive this same recorder rather than being a parallel
    measurement system. It becomes one driver among several, not the source of truth.

**Exit criteria**

- Injected stall of known size is flagged at the correct frame with correct magnitude in 100% of
  verification runs (SC-001).
- Repeated identical runs produce a stated noise floor.
- Stationary records are labelled as incapable of demonstrating movement behavior.
- Recorder overhead is measured and reported.
- An automated test drives a camera path and asserts on the accumulated record.

**Deliverable**: `contracts/frame-record.md`, `contracts/camera-path.md`, `quickstart.md`.

**Revert**: additive instrumentation and one UI view; no renderer behavior changed.

---

## Phase 1 — Baseline and attribute the real gallop (US2)

**Gate to enter**: Phase 0 exit criteria met.

**Work**

1. Record baselines on a tile-crossing path and a heading-change path, for Alpha 0.5.3 and one late
   client (3.3.5 or 4.0.0.12635), same map and path each time.
2. Confirm or refute the structural hypothesis from the spec's evidence section by measuring:
   allocation volume and **GC collection counts per frame** (the direct test of the allocation-churn
   theory — do hitch frames coincide with Gen0 collections?); frames coinciding with resident-set
   changes; time inside traversal versus submission versus streaming.
3. **Measure each churn surface C1–C7 individually** so Phase 2 is ordered by real cost rather than by
   how obvious the code looked. Include cache hit rates for C6 — the prediction is a near-total miss
   rate across frames, since the cache is cleared each frame.
4. Classify each flagged hitch as CPU, GPU/driver, or I/O/streaming. Where the split cannot be made
   with available instrumentation, say so in the report rather than guessing.
5. Set the concrete hitch threshold, noise floor, and the SC-004 improvement margin from this data.

**Exit criteria**

- The gallop is reproduced as a measured pattern (SC-002).
- Every flagged hitch carries an attribution or an explicit "cannot separate" (SC-003).
- Thresholds deferred by the spec checklist are now numbers backed by measurement.

**Deliverable**: `research.md` with the baseline, the attribution, and an explicit statement of
whether the scene-graph hypothesis is confirmed, partially confirmed, or refuted.

**Decision point**: if the hypothesis is refuted, Phase 3 is re-planned against the real cause before
any flattening work begins. The plan must not proceed on momentum.

---

## Phase 2 — Kill the churn (US5 partial, FR-039..FR-044)

**Gate to enter**: Phase 1 has measured which churn surfaces actually dominate.

**Why this is its own phase, before any architecture work**: every item below is a small, local,
independently revertible change to an existing structure. None of them require the flattened
representation. If the gallop is allocation-driven, a meaningful fraction of the win may land here —
at a fraction of the risk of an architecture rewrite. Doing this first also means the flattening work
in Phase 4 is measured against an already-cleaned baseline, so its own contribution is honest.

**One reversible change at a time, each measured against the Phase 1 baseline.** Ordered cheapest and
safest first:

1. **C2 — Stop the rejected-subtree walk.** Put rejected-node collection and per-kind attribution
   behind a diagnostic switch. This removes a full recursive walk of every culled region from every
   production frame, and it is the single most perverse cost found: today, culling more costs more.
2. **C3 — Stop rebuilding the active-graph set.** The active graph list and its `HashSet` change only
   on residency change; retain them and invalidate on change instead of rebuilding per frame.
3. **C1 — Retain traversal buffers.** Reuse the visible/rejected lists and diagnostics objects across
   frames instead of allocating per graph per frame.
4. **C6 — Make the renderer caches actually caches.** They are cleared every frame in `Reset()`,
   giving them a one-frame lifetime. Key them by a stable identity resolved once (not per-frame
   `OrdinalIgnoreCase` string hashing), let them live across frames, and invalidate on residency or
   asset change. If a structure must remain per-frame scratch, rename it so it stops claiming to be a
   cache.
5. **C4 — Retain the batching scratch structures.** The `Dictionary<IModelRenderer, List<Matrix4x4>>`,
   the three `HashSet`s, and the per-batch lists allocated inside loops become reused buffers cleared
   in place. This is the batching pass — added to improve performance — currently rebuilding all of
   its own intermediates every frame.
6. **C5 — Reuse the transparent sort buffer** rather than reallocating it per frame.
7. **Cache observability (FR-044).** Expose hit rate, size, and eviction cause for every retained
   cache, so "the cache is working" becomes a number rather than an assumption.

**Exit criteria**

- Every C1–C7 surface is either retained/incremental or has a written justification for remaining
  per-frame.
- No structure named a cache has a single-frame lifetime.
- Cache effectiveness is observable.
- Measured improvement against the Phase 1 baseline exceeds the noise floor, or the individual change
  is reverted.

**Note**: if Phase 2 alone brings frame-time variance within target, Phases 3 and 4 are re-evaluated
rather than executed on momentum. The architecture rewrite must still earn its place.

---

## Phase 3 — Design the flattened representation (US5 design)

**Gate to enter**: Phase 1 confirms an allocation/traversal-dominated cause.

**Work**

1. Define the retained draw-list model: stable per-tile buffers, ordered by pass, rebuilt incrementally
   on residency change only.
2. Define the pass structure explicitly (for example: terrain opaque, WMO opaque, doodad instanced,
   transparent, overlay/diagnostic). Passes are the unit of ordering; objects are entries in a pass.
3. Define how culling produces a *range or mask* over an existing list rather than a walk that
   allocates and that pays per rejected descendant.
4. Define the diagnostic switch: attribution data is produced only when requested.
5. Define the staleness contract: a draw list cannot be consumed while inconsistent with residency.

**Exit criteria**: `data-model.md` and `contracts/draw-list.md` reviewed against FR-021..FR-027.

**Deliverable**: design only. No production code changes.

---

## Phase 4 — Flatten the hot path (US5 implementation)

**Gate to enter**: Phase 2 design complete; Phase 1 baseline recorded.

**Rule for this phase: one reversible change at a time, each measured.**

Ordered by expected value against risk, cheapest and safest first:

1. **Move diagnostics off the production path.** Rejected-node collection and per-kind attribution
   become opt-in. This alone removes the recursive rejected-subtree walk from every frame. Measure.
2. **Remove per-frame caller allocation.** Reuse the active-graph list and set instead of building
   them per frame. Measure.
3. **Reuse traversal buffers.** Retain the visible/rejected lists and diagnostics objects across
   frames instead of allocating per graph per frame. Measure.
4. **Introduce the retained draw list** behind a switch, with the existing path still available.
   Measure both on the same trajectory.
5. **Make culling range-based** so rejection cost stops scaling with subtree size. Measure.
6. **Make rebuilds incremental** on residency change. Measure.

**Exit criteria**

- Steady-state frames allocate nothing proportional to node or tile count (SC-008).
- Rejection cost no longer scales with rejected subtree size (SC-009).
- Production frames do no diagnostic work unless requested (SC-010).
- Frame-time variance beats the Phase 1 baseline by more than the noise floor (SC-004).
- Visual equivalence confirmed against the previous path (FR-026).

**Revert**: each step is a separate commit; any step failing its measurement is reverted individually.

---

## Phase 5 — Focused view modes (US6)

**Gate to enter**: Phase 3 landed and measured.

**Work**

1. Define the mode set: terrain (ADT), model (M2/MDX), WMO, PM4. Modes select which passes and which
   content the scene builds.
2. Scope scene construction to the active mode; release the previous mode's content on switch without
   a world reload.
3. Make per-object discrete state conditional on the mode requiring picking, rather than always-on.
4. Keep cross-mode facts available through the sidebars without building the other mode's scene.
5. Report per-frame work per mode so modes are comparable.

**Exit criteria**: each mode measurably reduces per-frame work versus the combined view (SC-011);
mode switching does not leak residency or double-load.

---

## Phase 6 — Per-era terrain lighting (US3, independent)

**Gate to enter**: none. **This phase may start immediately and run in parallel with Phase 0/1.**
It touches no code the other phases depend on, and it fixes a visible defect on its own.

**Work**

1. Confirm by measurement that the 1.0.0+ darkness is a lighting-model era gap, not an asset or
   decode defect. This is currently an inference and must be verified before the fix is designed.
2. Define the era profile contract across 0.5.3 → 4.0.x, reusing the era-gating pattern already
   established for minimap generation rather than inventing a second scheme.
3. Select the profile from the active build; report which profile was selected.
4. Flag unprofiled builds explicitly and name the fallback; never silently apply the Alpha model.
5. Keep exact-build `Light*` DBC values authoritative where present; the era profile must not override
   them.
6. Carry provenance, not just values.

**Exit criteria**: brightness on a 1.0.0+ client matches the native client within the stated
tolerance with evidence recorded (SC-005); 100% of builds resolve to a named profile or are reported
unprofiled (SC-006).

---

## Phase 7 — One owner per UI surface (US7)

**Gate to enter**: Phase 4 mode structure decided — the mode set determines the correct organization,
so doing this earlier would mean doing it twice.

**Work**

1. Produce the full panel inventory: every `Draw*` content method and every route that reaches it.
   Start from the measured 214 methods / 71 multi-route figure.
2. Separate genuine reusable widgets from duplicated content panels. Widgets may stay shared.
3. Assign exactly one owning route per content panel, following the view-mode structure.
4. Retire duplicate routes with redirects to the owner, never by silently removing access.
5. Reconcile behavioral differences between duplicate routes deliberately, recording the decision.

**Exit criteria**: zero multiply-routed content panels (SC-012); sidebar organization follows the
active mode (FR-036).

---

## Supporting lane — Ghidra: how the native client bounded per-frame work

**Runs alongside Phases 1–3, not as a gate.**

Purpose is to learn the native client's *shape* of per-frame work — how it batched, ordered, and
bounded submission — to inform the pass structure in Phase 3. Static RE evidence from a staged binary
is explicitly permitted by the constitution and must cite the build it came from.

This is a learning lane. It is not a port; no original client code is copied. If it produces nothing
actionable, Phase 2 proceeds on measurement alone.

---

## Sequencing summary

| Phase | Depends on | Can start now | Reverts independently |
| --- | --- | --- | --- |
| 0 In-viewer runtime stats over time | — | **yes — #1 priority** | yes |
| 1 Baseline + attribution + churn measurement | 0 | no | yes |
| 2 Kill the churn (C1–C7) | 1 | no | yes, per step |
| 3 Flattening design | 2 | no | n/a (docs) |
| 4 Flatten hot path | 3 | no | yes, per step |
| 5 View modes | 4 | no | yes |
| 6 Era lighting | — | **yes, in parallel** | yes |
| 7 UI ownership | 5 | no | yes |
| Ghidra lane | — | yes | n/a |

The two phases that can start immediately are **0** (the detector, which is the #1 deliverable) and
**6** (era lighting, which is independent of everything else and fixes a visible defect on its own).

## Risks

| Risk | Mitigation |
| --- | --- |
| The scene-graph hypothesis is wrong and flattening does not fix the gallop | Phase 1 is an explicit decision point that can refute it; Phase 2 attacks churn directly and cheaply first, and Phases 3-4 must re-earn their place after Phase 2 is measured |
| A large rewrite leaves the viewer unusable mid-flight | Every phase ends runnable; Phases 2 and 4 land one reversible change at a time behind a switch |
| Measurement noise is mistaken for improvement | Noise floor is a Phase 0/1 deliverable and gates every acceptance |
| Flattening changes visual output subtly | Visual equivalence is an explicit exit criterion, and the old path stays available during migration |
| Scope sprawl across renderer + UI | Phase gates; UI work is last and inherits structure rather than inventing it |
| The recorder becomes a churn source itself | FR-002 requires fixed memory and zero per-frame allocation; recorder overhead is measured and reported with every record |

## Explicitly out of scope

- Porting the original client renderer.
- Backend change (Vulkan/WebGL) — the constitution names Silk.NET.OpenGL as current.
- Visual-fidelity work beyond era lighting correctness.
- Any FPS claim from build output alone.
