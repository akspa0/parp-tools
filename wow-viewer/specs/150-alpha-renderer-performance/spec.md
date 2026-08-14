# Feature Specification: Alpha 0.5.3 Renderer Performance Evidence and Optimization

**Feature Branch**: `150-alpha-renderer-performance`

**Created**: 2026-08-14

**Status**: Draft

**Input**: User description: "Figure out how to get the performance of the 0.5.3 OpenGL renderer in our renderer, without porting any original code and only taking hints from the original client."

## Scope

This feature is a build-scoped performance and evidence lane for Alpha 0.5.3. The original client
is an evidence source and control, not a code source. The work must preserve the current viewer
render path as a correctness fallback and must not become a renderer rewrite, an audio change, a
game-mode movement feature, or a cross-era generalization.

## User Scenarios & Testing

### User Story 1 - Identify the actual frame owner (Priority: P1)

When viewing a real 0.5.3 world, the maintainer can capture a repeatable frame report that separates
CPU visibility, terrain, object submission, transparent sorting, liquid, overlay, asset loading,
GPU timing when available, and their workload counts. The report identifies the dominant cost before
an optimization is selected.

**Why this priority**: A faster native client does not tell us which current viewer stage is slow.
Without attribution, batching, LOD, or shader changes are guesses and can make correctness worse.

**Independent Test**: Run the production render profile against the same 0.5.3 map, tile, camera,
resident-tile policy, warmup, and frame count twice; the reports contain the same stage inventory and
identify a reproducible dominant owner within an agreed variance.

**Acceptance Scenarios**:

1. **Given** a configured 0.5.3 client and a fixed camera route, **when** the profile is run after
   warmup, **then** it records CPU stage durations, visible/submitted counts, draw/state pressure,
   residency, pending work, and the capture identity in a machine-readable report.
2. **Given** GPU timer queries or an equivalent backend timer are unavailable, **when** the profile
   runs, **then** it labels GPU attribution as unavailable rather than presenting CPU submission time
   as GPU time.
3. **Given** repeated profiles with unchanged source and scene state, **when** the reports are
   compared, **then** the dominant stage and workload are stable enough to select one next slice.

### User Story 2 - Recover useful native performance contracts (Priority: P1)

When inspecting the open 0.5.3 client in Ghidra, the maintainer can record what the client actually
does for terrain/chunk admission, object visibility and distance, resource lifetime, render-state
grouping, and any LOD or far-horizon path. Each fact is tied to a function or data reference and is
classified as proven, inferred, or unknown.

**Why this priority**: The client can provide valuable constraints without making the viewer depend on
its implementation or accidentally importing behavior from a different build.

**Independent Test**: Review the evidence ledger and confirm every proposed optimization has either a
0.5.3 function/data anchor or an explicit reason it is a viewer-side experiment rather than a native
client claim.

**Acceptance Scenarios**:

1. **Given** a proposed culling, batching, resource, or LOD change, **when** its evidence row is
   reviewed, **then** the row names the 0.5.3 anchor, observed behavior, confidence, and viewer-side
   translation without copying original implementation code.
2. **Given** a native behavior cannot be recovered from the open program, **when** the ledger is
   updated, **then** it remains unknown and is not used as a correctness claim.
3. **Given** evidence from a later client or an old reference renderer, **when** it is considered,
   **then** it is labeled as comparative context and cannot silently become a 0.5.3 contract.

### User Story 3 - Apply one reversible performance improvement (Priority: P1)

When the dominant owner is known, the maintainer can apply one bounded optimization that reduces
work for the same visible 0.5.3 scene while preserving the existing path as a fallback and retaining
diagnostics for visible, submitted, culled, and deferred work.

**Why this priority**: The goal is useful frame time, but only after the evidence gate prevents a
large speculative rewrite.

**Independent Test**: Compare baseline and post-change profiles using the same scene and camera. The
selected stage improves materially, the visible result remains equivalent within the declared capture
tolerance, and the optimization can be disabled for A/B diagnosis.

**Acceptance Scenarios**:

1. **Given** a measured dominant stage, **when** one optimization is enabled, **then** the report
   identifies the changed path and its before/after workload and timing rather than only reporting FPS.
2. **Given** unsupported geometry, transparent/material-sensitive content, or an unsettled asset,
   **when** the optimized path cannot safely handle it, **then** the existing fallback path renders it
   and records the fallback reason.
3. **Given** a camera turn, tile seam, map switch, or asset reload, **when** the optimization is
   exercised, **then** resource ownership and residency remain bounded and no stale draw submission
   survives the source change.

### User Story 4 - Prove the result on the real control scene (Priority: P2)

When a bounded optimization is ready, the maintainer can compare the viewer against its own baseline
and the user can compare the same 0.5.3 scene against the native client without conflating source
parity, visual parity, CPU time, GPU time, or audible/runtime behavior.

**Why this priority**: Compilation and synthetic workload tests can validate contracts, but they cannot
prove real-client frame pacing or visual parity.

**Independent Test**: Run focused tests and a viewer profile locally, then run the documented native
client and interactive viewer comparison with identical map, camera, resolution, and display settings.

**Acceptance Scenarios**:

1. **Given** the same 0.5.3 map/camera state, **when** baseline and optimized viewer captures are
   compared, **then** terrain, holes, liquids, WMOs, and M2/MDX objects remain present and their
   documented visual differences are explained.
2. **Given** a real native-versus-viewer comparison, **when** timing is reported, **then** the report
   names whether each number is native client FPS, viewer CPU frame time, viewer GPU time, or a
   user-observed display measurement.
3. **Given** the optimization is not a net win, **when** the gate fails, **then** the change remains
   disabled or is reverted without weakening the correctness path.

### Edge Cases

- A 0.5.3 map may have many resident tiles but only a small camera-visible set; residency must not be
  mistaken for submission.
- A render may be CPU-bound, GPU-bound, driver-bound, or asset-I/O-bound; missing timer evidence must
  remain explicit.
- Shared models can have different transforms, animation state, material flags, transparency, or
  particle/ribbon requirements; only compatible instances may share a submission path.
- A tile or model can enter or leave the bounded residency window during a camera turn; GPU resources
  must not be used after unload.
- Alpha terrain, WMO, M2/MDX, liquid, overlays, and debug surfaces have different correctness and
  culling rules; one aggregate "draw call" number is insufficient.
- A native function may be named only by address or have ambiguous control flow; the evidence ledger
  must preserve uncertainty instead of inventing a semantic name.
- A later-build optimization may look attractive but may depend on formats or client systems absent
  in 0.5.3; build identity is part of every evidence row.

## Requirements

### Functional Requirements

- **FR-001**: The performance lane MUST target Alpha 0.5.3.3368 first and MUST record the map, tile or
  route, client build, camera/residency policy, resolution, warmup, frame count, and source revision
  for every comparison.
- **FR-002**: The viewer MUST expose separate CPU stage timing and workload counts for terrain, WDL,
  liquid, WMO visibility/submission, MDX visibility/animation/submission, transparent sorting,
  overlays, scene maintenance, deferred asset loading, and pending residency work.
- **FR-003**: The performance report MUST distinguish CPU submission time from GPU/driver timing and
  MUST label unavailable GPU attribution explicitly.
- **FR-004**: The report MUST include enough pressure counters to explain a frame: visible and culled
  terrain chunks, visible object instances, opaque/transparent batch and fallback counts, WMO group/
  liquid/doodad submissions, draw calls, uniform/state/texture pressure when available, and pending
  asset work.
- **FR-005**: Native-client evidence MUST be stored as build-scoped observations with an anchor,
  behavior, confidence, and viewer-side implication; original client source or implementation code
  MUST NOT be copied into the viewer.
- **FR-006**: Every optimization MUST be selected from a measured owner or a clearly labeled native
  evidence gap, MUST have a reversible enable/disable boundary, and MUST preserve a correctness
  fallback for unsupported content.
- **FR-007**: The first optimization phase MUST change one dominant owner at a time and MUST preserve
  before/after counters so batching, culling, LOD, resource reuse, and state reduction cannot be
  credited for one another's effects.
- **FR-008**: Terrain and object visibility MUST remain bounded by the existing residency and camera
  admission policy; performance work MUST NOT load the whole map merely to improve a frame profile.
- **FR-009**: Performance work MUST preserve the current Alpha terrain coordinate, hole, liquid,
  WMO, M2/MDX, and texture correctness contracts unless a separate evidence-backed change is approved.
- **FR-010**: Focused automated coverage MUST verify report schema stability, dominant-owner selection,
  unavailable GPU timing, reversible fallback selection, and workload/counter consistency.
- **FR-011**: Real-client visual and FPS comparison MUST remain a separate user-owned proof gate and
  MUST NOT be claimed from compilation, unit tests, or headless CPU reports alone.
- **FR-012**: Player/game-mode movement, audio behavior, PM4 matching/region UI, renderer backend
  replacement, Vulkan, compute shaders, and cross-era performance claims MUST remain out of scope.

### Key Entities

- **NativeRenderEvidence**: One build-scoped observation from the 0.5.3 client, including its anchor,
  observed behavior, confidence, and viewer-side implication.
- **RenderPerformanceSample**: One production viewer frame or warmup-set summary containing identity,
  CPU/GPU timing classification, stage durations, workload counts, resource pressure, and findings.
- **OptimizationExperiment**: One reversible viewer-side change with a baseline sample, post-change
  sample, selected owner, fallback behavior, and acceptance decision.
- **RenderControlScene**: The fixed map/camera/residency/resolution setup used for repeatable comparison.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A repeatable 0.5.3 control scene produces two machine-readable reports with identical
  stage and workload fields and a reproducible dominant owner within the documented variance.
- **SC-002**: The native evidence ledger contains at least one anchored observation for each proposed
  optimization, or explicitly records that the native behavior is unknown and keeps the proposal
  experimental.
- **SC-003**: The first accepted optimization reduces the selected owner by at least 15 percent or
  5 ms at the chosen control scene, whichever is the smaller threshold, without increasing total
  p95 CPU frame time.
- **SC-004**: Baseline and optimized captures preserve all previously visible Alpha terrain, liquid,
  WMO, and M2/MDX classes at the same camera state, with any difference explained by the report.
- **SC-005**: Unsupported or fallback-routed content remains visible in diagnostics and does not
  silently bypass the correctness path.
- **SC-006**: No final report describes CPU submission time as GPU time, and no final handoff claims
  native-client FPS parity without user-run native and viewer evidence.

## Assumptions

- The approved client library and the open Ghidra program for `H:\CLIENTS\Vanilla\0.x\0_5_3_3368`
  are available for user-owned native comparison; paths remain runtime/configuration inputs and are
  not hardcoded into source.
- The existing `profile-render` validation command is the first repeatable production-render capture
  surface and may be extended rather than replaced.
- Current retained tile VAOs, texture arrays, bounded streaming, object visibility collectors, and
  opaque instancing are reusable foundations; they are not treated as proof that the whole renderer
  is fast.
- A CPU-only result is still useful when GPU timing is unavailable, provided the limitation is stated.
- The first accepted slice may improve one owner only; a broad renderer modernization remains a later
  separately approved phase.
