# Implementation Plan: Renderer frame-time stability, flattened pipeline, and focused view modes

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-15 | **Spec**: [spec.md](spec.md)

## Summary

v0.5.3 makes the renderer render properly. Three defects share one root: the renderer treats the
world as a tree of discrete per-object nodes walked every frame, instead of retained ordered lists
of draw work processed in passes. That produces the gallop (per-frame allocation and rejection cost),
the universal-scene cost (every overlay always live), and the UI sprawl (every surface reachable from
everywhere, with no owner).

The approach is deliberately ordered so that nothing is optimized on a guess:

1. **Make the detector able to see the defect, and prove it.** The existing harness is blind.
2. **Baseline and attribute** the real hitch pattern.
3. **Flatten** the per-frame path into retained ordered draw lists and explicit passes.
4. **Scope** scene construction to a focused view mode.
5. **Fix era lighting** (independent; can run in parallel from day one).
6. **Give every UI surface one owner**, following the view-mode structure.

Every phase is independently revertible and leaves the viewer runnable.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET.OpenGL (current backend), ImGui.NET (viewer shell)

**Storage**: N/A for this feature; measurement reports are JSON on disk

**Testing**: xUnit for pure contracts; `WowViewer.Tool.ValidationCapture profile-render` for
measurement; maintainer-owned interactive proof for visual sign-off

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
|---|---|---|
| III. Real-Data Validation | PASS | All measurement runs against staged real clients at `H:\CLIENTS`; no synthetic-only claim of success. |
| VI. No Game Client Path Assumptions | PASS | Client roots stay configuration; trajectories name a client/build/map as data, never hardcoded. |
| Read-Only Reference Codebase | PASS | No work in `gillijimproject_refactor/`. |
| One Phase at a Time | PASS | Phase gates are explicit below; each ends at a checkpoint. |
| Bite-Sized Plans | AT RISK | This feature is large. Mitigated by hard phase gates, per-phase revertibility, and a rule that Phase 3+ lands one reversible change at a time. |
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
├── research.md          # Phase 0 output — hitch mechanism + Ghidra lane
├── data-model.md        # Phase 2 output — draw list / pass / view-mode entities
├── quickstart.md        # How to run a measurement and read a report
├── contracts/           # Trajectory, run-report, era-profile, draw-list contracts
└── checklists/
    └── requirements.md  # Spec quality checklist (written)
```

### Source code touched

```text
wow-viewer/
├── tools/validation-capture/WowViewer.Tool.ValidationCapture/
│   ├── ProductionWorldSceneProfiler.cs      # Phase 0: trajectories, per-frame timing, hitch stats
│   └── ValidationCaptureCommand.cs          # Phase 0: new options
├── src/core/WowViewer.Core.Runtime/World/SceneGraph/
│   ├── WorldSceneTraversal.cs               # Phase 1/3: diagnostics off the hot path; flattening
│   └── WorldSceneNode.cs                    # Phase 3: retained list representation
├── src/core/WowViewer.Core.Renderer/Scene/
│   └── SceneRenderer.cs                     # Phase 3: pass-oriented submission
├── src/viewer/WoWViewer/Terrain/
│   ├── WorldScene.cs                        # Phase 3/4: hot path; later decomposition
│   ├── TerrainRenderer.cs                   # Phase 3: pass participation
│   └── TerrainLighting.cs                   # Phase 5: era profile selection
└── src/viewer/WoWViewer/
    └── ViewerApp*.cs                        # Phase 6: single-owner routing
```

**Structure Decision**: measurement lives in the existing validation-capture tool so profiling stays
headless and scriptable. The flattened representation belongs in `Core.Runtime` next to the current
scene graph so both can coexist behind a switch during migration. Era lighting stays in the viewer's
terrain layer where the current model lives.

---

## Phase 0 — Make the detector able to see the defect (US1)

**Gate to enter**: none. This is first, always.

**Why first**: the current harness renders a stationary camera for 12 frames and reports no timing
distribution. It cannot observe the defect. Every optimization measured against it would be noise.

**Work**

1. Add named camera trajectories to the profiler: at minimum `stationary` (kept, honestly labelled),
   `linear-crossing` (crosses at least one ADT boundary), and `orbit` (continuous heading change,
   which the current residency policy reacts to).
2. Capture per-frame wall-clock time for every measured frame, plus the existing per-stage stats.
3. Report median, max, p95, p99, and a count of frames over a hitch threshold, alongside the raw
   per-frame series.
4. Add `--inject-hitch-ms <n> --inject-hitch-frame <i>` to stall a known frame by a known amount.
5. Add a noise-floor procedure: repeat an identical run N times, report run-to-run spread.
6. Raise the default measured-frame count so at least one full tile-crossing cycle is observed.
7. Stamp every report with client root, build, map, trajectory, frame counts, and trajectory type.

**Exit criteria**

- Injected hitch of known size is flagged at the correct frame index, within tolerance, in 100% of
  verification runs (SC-001).
- Two identical runs produce a stated noise floor.
- A stationary run is labelled as incapable of demonstrating movement behavior.

**Deliverable**: `contracts/trajectory.md`, `contracts/run-report.md`, `quickstart.md`.

**Revert**: profiler-only changes; no renderer code touched.

---

## Phase 1 — Baseline and attribute the real gallop (US2)

**Gate to enter**: Phase 0 exit criteria met.

**Work**

1. Record baselines on `linear-crossing` and `orbit` for Alpha 0.5.3 and one late client (3.3.5 or
   4.0.0.12635), same map and trajectory each time.
2. Confirm or refute the structural hypothesis from the spec's evidence section by measuring:
   allocation volume and GC collection counts per frame; frames coinciding with resident-set changes;
   time inside traversal versus submission versus streaming.
3. Classify each flagged hitch as CPU, GPU/driver, or I/O/streaming. Where the split cannot be made
   with available instrumentation, say so in the report rather than guessing.
4. Set the concrete hitch threshold, noise floor, and the SC-004 improvement margin from this data.

**Exit criteria**

- The gallop is reproduced as a measured pattern (SC-002).
- Every flagged hitch carries an attribution or an explicit "cannot separate" (SC-003).
- Thresholds deferred by the spec checklist are now numbers backed by measurement.

**Deliverable**: `research.md` with the baseline, the attribution, and an explicit statement of
whether the scene-graph hypothesis is confirmed, partially confirmed, or refuted.

**Decision point**: if the hypothesis is refuted, Phase 3 is re-planned against the real cause before
any flattening work begins. The plan must not proceed on momentum.

---

## Phase 2 — Design the flattened representation (US5 design)

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

## Phase 3 — Flatten the hot path (US5 implementation)

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

## Phase 4 — Focused view modes (US6)

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

## Phase 5 — Per-era terrain lighting (US3, independent)

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

## Phase 6 — One owner per UI surface (US7)

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

**Runs alongside Phase 1–2, not as a gate.**

Purpose is to learn the native client's *shape* of per-frame work — how it batched, ordered, and
bounded submission — to inform the pass structure in Phase 2. Static RE evidence from a staged binary
is explicitly permitted by the constitution and must cite the build it came from.

This is a learning lane. It is not a port; no original client code is copied. If it produces nothing
actionable, Phase 2 proceeds on measurement alone.

---

## Sequencing summary

| Phase | Depends on | Can start now | Reverts independently |
|---|---|---|---|
| 0 Detector power | — | yes | yes |
| 1 Baseline + attribution | 0 | no | yes |
| 2 Flattening design | 1 | no | n/a (docs) |
| 3 Flatten hot path | 2 | no | yes, per step |
| 4 View modes | 3 | no | yes |
| 5 Era lighting | — | **yes, in parallel** | yes |
| 6 UI ownership | 4 | no | yes |
| Ghidra lane | — | yes | n/a |

## Risks

| Risk | Mitigation |
|---|---|
| The scene-graph hypothesis is wrong and flattening does not fix the gallop | Phase 1 is an explicit decision point that can refute it and force a re-plan; flattening is not started until measurement supports it |
| A large rewrite leaves the viewer unusable mid-flight | Every phase ends runnable; Phase 3 lands one reversible change at a time behind a switch |
| Measurement noise is mistaken for improvement | Noise floor is a Phase 0 deliverable and gates every acceptance |
| Flattening changes visual output subtly | Visual equivalence is an explicit exit criterion, and the old path stays available during migration |
| Scope sprawl across renderer + UI | Phase gates; UI work is last and inherits structure rather than inventing it |
| Optimizing the profiler's own overhead into the measurement | Profiler overhead is characterized in Phase 0 and reported with every run |

## Explicitly out of scope

- Porting the original client renderer.
- Backend change (Vulkan/WebGL) — the constitution names Silk.NET.OpenGL as current.
- Visual-fidelity work beyond era lighting correctness.
- Any FPS claim from build output alone.
