# Implementation Plan: Alpha 0.5.3 Renderer Performance Evidence and Optimization

**Branch**: `150-alpha-renderer-performance` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

## Summary

Use the 0.5.3 client as a build-scoped performance control and evidence source. First make the
existing production OpenGL path explain its CPU, GPU/driver, visibility, submission, state, and
residency costs. In parallel, record native Ghidra observations for the exact behaviors that may
explain its frame pacing. Then implement one reversible viewer-side optimization for the measured
dominant owner, preserving the current path for unsupported or correctness-sensitive content.

The likely high-value hints are coarse admission before per-object work, stable retained resources,
compatible opaque submission groups, limited state changes, and distance/far-horizon reduction. None
of those is accepted as a 0.5.3 fact until the native evidence ledger anchors it.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WorldScene`, `TerrainRenderer`, `WorldRenderDiagnostics`,
`WorldObjectVisibilityCollector`, `WorldObjectPassCoordinator`, `profile-render`, Silk.NET OpenGL,
and the read-only Ghidra program for `WoWClient.exe` 0.5.3.3368

**Storage**: Machine-readable diagnostic JSON and checked-in evidence/spec documents; no proprietary
client data or captures in the repository

**Testing**: Focused `WowViewer.Core.Tests`, validation-capture report tests, Debug build, and
user-owned 0.5.3 visual/FPS/native-client comparison

**Target Platform**: Windows desktop viewer and hidden OpenGL validation-capture host

**Project Type**: Desktop application with shared runtime and renderer libraries

**Performance Target**: First accepted experiment reduces its selected measured owner by at least
15 percent or 5 ms at the declared control scene, without increasing total p95 CPU frame time or
breaking visual correctness

**Constraints**: No original client code port; no renderer backend replacement; no whole-map load;
no unbounded per-frame allocations; keep current paths and fallbacks; do not generalize one Alpha
finding to later builds

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after design.*

* **PASS — Evidence first**: no optimization is selected before a repeatable production report and
  a native evidence row or explicit unknown state exist.
* **PASS — Library-first**: counters and reusable performance contracts belong in core/runtime owners;
  viewer UI only displays them.
* **PASS — Source boundary**: Ghidra disassembly and old renderer behavior are evidence only; no
  original client or legacy viewer code is copied into the implementation.
* **PASS — Build scope**: Alpha 0.5.3.3368 is the first proof target; later builds require their own
  evidence and are not silently covered.
* **PASS — Reversible changes**: each optimization keeps an A/B boundary and a correctness fallback.
* **PASS — User-owned proof**: real native FPS, interactive viewer visuals, and GPU-driver timing
  remain explicit user gates.

## Existing Owners and Reuse Boundary

| Concern | Existing owner | Plan use |
|---|---|---|
| Production world frame | `src/viewer/WoWViewer/Terrain/WorldScene.cs` | Keep pass order; add only attribution/scratch reuse needed by the selected experiment |
| Terrain submission | `src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` | Reuse retained tile VAO, texture-array, cull, and draw counters |
| Object admission | `src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` | Measure before changing distance/cone/projected-size rules |
| Opaque grouping | `src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` | Preserve current batch eligibility and add no global regrouping without evidence |
| Report schema | `src/core/WowViewer.Core.Runtime/World/WorldRenderDiagnostics.cs` | Extend versioned report only when a field is needed by a selected gate |
| Capture harness | `tools/validation-capture/WowViewer.Tool.ValidationCapture/ValidationCaptureCommand.cs` | Reuse `profile-render` and its hidden production OpenGL path |
| Native evidence | `memory-bank/workstream-alpha053-renderer-performance.md` | Record address/function/observation/confidence; never copy implementation code |

## Phase 0 — Native Evidence and Baseline (P1)

**Goal**: Produce the evidence ledger and stable current-renderer baseline before changing source.

1. Confirm the exact 0.5.3.3368 binary/program identity and record read-only Ghidra anchors for the
   world render loop, terrain/chunk submission, object visibility/draw distance, resource/state setup,
   and far-distance or LOD behavior. If an anchor cannot be recovered, mark it unknown.
2. Run `profile-render` on one fixed 0.5.3 outdoor control scene with settled warmup and repeated
   samples. Record map/tile/camera/residency/build identity and source revision.
3. Compare stage timing and workload fields against the screenshot/runtime counters already visible
   in the viewer; do not treat 23 FPS or 41.2 ms from one interactive frame as a benchmark.
4. Select exactly one dominant owner and write the proposed experiment, expected counter movement,
   fallback path, and stop condition.

**Exit**: A checked-in evidence ledger, baseline report shape, and one selected owner exist. No source
optimization begins while the dominant owner is unknown.

## Phase 1 — Attribution Contract (P1)

**Goal**: Make the selected cost and its pressure observable without changing rendering behavior.

1. Extend the shared diagnostic model only for missing counters required by the selected owner, keeping
   CPU timing, GPU/driver timing, and unavailable timing distinct.
2. Add focused report-contract tests for field presence, unavailable GPU timing, stable stage names,
   workload/counter consistency, and dominant-owner selection.
3. Thread the new counters through the production `WorldScene`/renderer owner and `profile-render`.
4. Display the result in existing Runtime Stats or diagnostic output without adding per-frame UI work
   to the production render path.

**Exit**: Two unchanged-source profiles identify the same owner and show its pressure counters.

## Phase 2 — One Reversible Optimization (P1)

**Goal**: Reduce the selected owner while preserving the current path and visual behavior.

1. Implement one bounded change, selected from measured evidence: scratch-buffer reuse, cull/admission
   ordering, compatible opaque grouping, state/uniform reduction, retained resource reuse, or a
   build-scoped terrain/object LOD reduction.
2. Add a runtime or validation-capture switch that makes the old and new paths directly comparable.
3. Route unsupported, transparent, animated, particle/ribbon, or material-sensitive content through
   the existing fallback and count it.
4. Add focused tests for the changed decision and for fallback selection.
5. Re-run the same baseline/post-change profile and reject the experiment if the selected owner does
   not improve or total p95 CPU time regresses.

**Exit**: The first experiment either passes the measurable gate or is disabled with its negative
result recorded. No second optimization is bundled into this phase.

## Phase 3 — Build-Scoped Visibility/LOD Follow-through (P2)

**Goal**: Apply a second native-informed improvement only if Phase 2 proves the measurement and A/B
workflow.

1. Use the evidence ledger to distinguish 0.5.3 chunk/object distance behavior from later-build LOD
   behavior.
2. Extend existing `WorldTerrainLodSelector`, WDL, object admission, or render-bucket contracts only
   where the 0.5.3 control scene and counters demonstrate a real opportunity.
3. Keep Alpha terrain holes/liquids, WMO visibility, M2/MDX fallback, and tile residency in the same
   correctness matrix.
4. Validate one new distance bucket or admission rule independently before combining it with Phase 2.

**Exit**: A separate report demonstrates the selected distance/visibility change and its visual gate;
otherwise the phase remains documented as an evidence gap.

## Phase 4 — Handoff and User Proof (P1)

1. Run focused tests, `git diff --check`, and the Debug build.
2. Prepare PowerShell-ready `profile-render` commands for the configured 0.5.3 client root and
   control map; the user runs the real-client and interactive comparisons.
3. Record whether each timing is viewer CPU, viewer GPU/driver, native FPS, or user-observed display
   timing.
4. Update the performance workstream, active context, status router, and this spec with completed,
   unproven, and next bounded work.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Native 0.5.3 behavior is under-symbolized | Preserve address/byte/control-flow anchors and label unknowns; do not infer from later clients |
| CPU report hides GPU bottleneck | Add optional GPU/driver timing and retain an explicit unavailable state |
| Batching changes visual order or material behavior | Restrict first change to compatible opaque work and keep fallback routing |
| LOD improves FPS by hiding content | Compare visible/submitted counts and fixed captures, not FPS alone |
| Performance is dominated by client reads/asset settling | Separate deferred reads, pending work, and settled-frame samples |
| Existing 149/148 dirty work is accidentally mixed in | Keep source edits disjoint and preserve unrelated worktree changes |

## Out of Scope

- Porting, copying, or translating original client code.
- Vulkan, compute shaders, renderer backend replacement, or a full shared-renderer rewrite.
- Player/game mode movement, PM4 region/matching UI, audio/music behavior, or native callback work.
- Cross-era claims based solely on the 0.5.3 control scene.
