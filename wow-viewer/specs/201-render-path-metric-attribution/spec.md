# Feature Specification: Separating M2 and MDX Render-Path Metrics

**Feature Branch**: `201-render-path-metric-attribution`

**Created**: 2026-09-01

**Status**: Draft

**Input**: Operator observation, 2026-09-01: *"we list M2 as MDX — which is the wrong way to
track metrics of two different render paths."*

## Context

The frame-history panel reports `MDX opaque: 0 batched, 13180 unbatched (100.00% unbatched)`.
That counter does not mean what its label says.

`ResolveVisibleMdxRenderer` returns an `IModelRenderer`, which is **either** `M2Renderer`
**or** `MdxRenderer`. Both increment the same `OpaqueBatchedMdxCount` /
`OpaqueUnbatchedMdxCount` fields on `WorldRenderFrameStats`. The status bar's `MDX
13180/20115` is likewise every placed model regardless of path. Two different render paths,
one bucket.

The project already has the correct axis and does not use it here. `M2RouteType` distinguishes
five routes — `AdapterSkin`, `AdapterEmbeddedProfile`, `NativeEmbeddedProfile`,
`ConversionFallback`, and `MdxDirect` (the legacy loader for genuine non-M2 models) — and
`M2RouteDecision` already records the primary and applied route per model.

**What the conflation hides.** `M2Renderer` delegates batching to an inner legacy renderer:

```csharp
public bool RequiresUnbatchedWorldRender
    => _legacyRenderer is null || _legacyRenderer.RequiresUnbatchedWorldRender;

public bool SupportsGpuInstancedOpaque
    => _legacyRenderer is IGpuInstancedModelRenderer gpuRenderer
       && gpuRenderer.SupportsGpuInstancedOpaque;
```

An M2 on the **native runtime route** has no `_legacyRenderer`, so it reports
`RequiresUnbatchedWorldRender = true` and `SupportsGpuInstancedOpaque = false`
**unconditionally**. Its own comment says why: *"The native runtime backend owns a different
shader/state path, so keep it isolated until a backend-specific batch key exists."*

So an unknown share of those 13,180 unbatched draws is **structurally unbatchable today**, and
spec 153 US3's toggle cannot touch them. The single-bucket metric is what makes that
invisible: the panel says "100% unbatched" and implies one fix, when there are two causes with
two different fixes and no way to tell their sizes apart.

This spec is a **precondition for the measurements specs 153, 198 and 200 depend on**. Until
the counters separate the paths, before/after comparisons on any of them are unreadable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The panel says which path the draws came from (Priority: P1)

An operator reading submission efficiency sees opaque and transparent counts broken out by
render path, so "100% unbatched" resolves into which path produced them.

**Why this priority**: Every performance conclusion currently drawn from this panel is
unattributable. This is the smallest change that makes the existing numbers mean something.

**Independent Test**: Fly a route and read per-path counts. Delivers a correct diagnosis with
no batching behaviour changed.

**Acceptance Scenarios**:

1. **Given** a frame containing both M2- and MDX-routed models, **When** the panel is read,
   **Then** batched and unbatched counts are reported per render path and sum to the totals.
2. **Given** a model on a route that cannot batch, **When** it is counted, **Then** it is
   distinguished from a model that *could* batch but did not.

---

### User Story 2 - Unbatchable is distinguished from unbatched (Priority: P1)

The report separates draws that are unbatched because batching is switched off from draws that
are unbatched because their route has no batch path at all.

**Why this priority**: Equal to US1, and the point of it. "Unbatched" currently conflates a
toggle state with a structural gap. Only the first is fixed by flipping the toggle.

**Independent Test**: Read the counts with the batching toggle off and on; the structurally
unbatchable count must not move.

**Acceptance Scenarios**:

1. **Given** the batching toggle is enabled, **When** counts are read, **Then** draws on
   routes with no batch path remain counted as unbatchable, not as a batching failure.
2. **Given** a route reports `RequiresUnbatchedWorldRender`, **When** it is counted, **Then**
   the reason is recorded.

---

### User Story 3 - The native-M2 batch path exists (Priority: P2)

M2 instances on the native runtime route participate in opaque batching, via the
backend-specific batch key their own code comments say is missing.

**Why this priority**: This is the actual performance work, and US1/US2 must come first — the
size of the prize is currently unknown. If native-route M2s turn out to be a small share of
the 13,180, this is not worth doing and the measurement will say so.

**Independent Test**: Fly a fixed route before and after; compare per-path unbatched counts and
frame time.

**Acceptance Scenarios**:

1. **Given** an M2 on the native runtime route with no legacy renderer, **When** opaque
   batching is enabled, **Then** it is submitted through a batch rather than per instance.
2. **Given** batching is enabled for native-route M2s, **When** the scene renders, **Then**
   appearance is unchanged from the unbatched path.

---

### Edge Cases

- A model whose applied route differs from its primary route — the metric must attribute to
  the **applied** route, since that is what actually drew.
- A route that batches some passes and not others.
- Models that fail to load (`MDX 13180/20115 (429 ok/0 fail)` suggests a large gap between
  placed and drawn) — counted separately from either path, not silently absent.
- Changing the toggle mid-session must not corrupt the per-path attribution.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Submission counters MUST be reported per render path, not as a single `MDX`
  bucket.
- **FR-002**: Attribution MUST use the **applied** route, not the intended one.
- **FR-003**: The report MUST distinguish *unbatched* (batching available, not applied) from
  *unbatchable* (no batch path exists for that route).
- **FR-004**: Where a draw is unbatchable, the reason MUST be recorded.
- **FR-005**: Per-path counts MUST sum to the existing totals, so the change is provably a
  decomposition and not a redefinition.
- **FR-006**: The label `MDX` MUST NOT be used for counts that include M2-routed models,
  anywhere in the UI or the stats contract.
- **FR-007**: Adding attribution MUST NOT change what is drawn or how it is batched.
- **FR-008**: Any native-route batching added under US3 MUST be switchable at runtime for
  same-session before/after, matching how spec 153 US3's toggle behaves.

### Key Entities

- **Render path**: The applied `M2RouteType` for an instance — the four M2 routes plus
  `MdxDirect`.
- **Submission outcome**: Per instance — batched, unbatched, or unbatchable-with-reason.
- **Per-path submission stats**: Opaque and transparent counts per path, summing to the
  existing totals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a flown route, opaque and transparent submission counts are reported per
  render path, and the per-path counts sum exactly to the previously reported totals.
- **SC-002**: The share of the measured 13,180 unbatched opaque draws attributable to
  native-route M2 is known as a number.
- **SC-003**: No counter or label in the UI reports M2-routed models as `MDX`.
- **SC-004**: With attribution added and batching unchanged, a flown route produces identical
  frame-time distribution to before — proving FR-007.
- **SC-005**: If US3 is implemented, per-path unbatched counts for native-route M2 fall, and
  the appearance of a reference scene is unchanged.

## Assumptions

- `M2RouteType` and `M2RouteDecision` are the right axis and already exist; this spec uses
  them rather than inventing a parallel classification.
- The Wandering Isle flight (2048 frames, median 84.44 ms, 13,180 unbatched opaque) is the
  baseline; re-flights are operator-run.
- Whether native-route M2s dominate the unbatched count is **genuinely unknown**. US3 is
  written so that "they are a small share, do not bother" is a legitimate outcome of SC-002.
- Spec 153 US3 continues to own batching for legacy-backed models; this spec adds the path the
  native backend lacks, and does not re-open the existing one.
