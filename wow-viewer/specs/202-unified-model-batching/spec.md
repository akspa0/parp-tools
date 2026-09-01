# Feature Specification: Unified Model Batching Across Render Paths and Eras

**Feature Branch**: `202-unified-model-batching`

**Created**: 2026-09-01

**Status**: Draft

**Input**: Operator, 2026-09-01: *"We must fix our batching of doodads/MDX/M2 data, as that is
what kills performance in the viewer, for ALL versions of data. It's the one outstanding issue
that really needs an overhaul."*

## Context

Measured, operator flight over a 5.0.1 map (Wandering Isle), 2026-09-01:

```
2048 frames   median 84.44 ms   p95 130.81   p99 139.62   max 222.16
1820 of 2048 frames over 33.3 ms       CPU 129.0 ms       7 FPS
opaque:      0 batched, 13180 unbatched (100%)
transparent: 0 batched,   260 unbatched (100%)
```

At a median of 84 ms this is not hitching — the renderer is uniformly slow, and one draw call
per instance is why. `UNACCOUNTED median 0.15 ms` confirms the stage timers cover the frame,
so the attribution to submission is sound.

Four pieces of work already touch this and none of them own it end to end:

| Spec | Owns | State |
|---|---|---|
| 136 | M2 doodad performance | 9/11; both open tasks are operator measurement |
| 153 US3 | Opaque **MDX** batching | shipped, behind a runtime toggle |
| 201 | Per-path metric attribution + native-M2 batch key | planned |
| **202 (this)** | **One batching architecture across every path and era** | — |

**Why the existing pieces are not enough.** Batching is currently a property each renderer
opts into. `M2Renderer` delegates the decision to an inner legacy renderer and, when there
isn't one, refuses:

```csharp
public bool RequiresUnbatchedWorldRender
    => _legacyRenderer is null || _legacyRenderer.RequiresUnbatchedWorldRender;
```

Its comment states the gap plainly: *"The native runtime backend owns a different shader/state
path, so keep it isolated until a backend-specific batch key exists."* So whether an instance
batches depends on which loader happened to construct it — which is an implementation
accident, not a rendering decision.

That is why this is an overhaul rather than another per-renderer fix, and why the operator
reports it across **all** data versions: 0.5.3 MDX, LK, Cata and MoP models each arrive through
different routes, and each route has its own answer to "can I batch". There is no single place
that decides.

**Dependency**: spec 201 must land first. Its counters currently report M2 and MDX in one
`MDX` bucket, so the size of each cause is unknown and any before/after here would be
unreadable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Batching is decided by render state, not by loader (Priority: P1)

Two instances that could share a draw do share one, regardless of which loader produced them
or which era the model came from.

**Why this priority**: This is the overhaul. Every other story is a consequence.

**Independent Test**: Load a scene with instances of the same model reaching the renderer by
different routes; confirm they submit as one batch.

**Acceptance Scenarios**:

1. **Given** two instances whose render state is identical, **When** the frame submits,
   **Then** they share a draw call regardless of applied route.
2. **Given** two instances whose render state differs, **When** the frame submits, **Then**
   they are in different batches and the differing state is nameable.
3. **Given** a route with no batching today, **When** the frame submits, **Then** it batches
   on the same terms as every other route, or reports why it cannot.

---

### User Story 2 - It works on every era we support (Priority: P1)

0.5.3, LK, Cata and MoP content all batch. The operator's statement is that this kills
performance for **all** versions, so a fix that only helps 5.0.1 has not delivered.

**Why this priority**: Equal to US1. Era coverage is the acceptance boundary, not a follow-up.

**Independent Test**: Fly a fixed route on one map per era and compare unbatched counts.

**Acceptance Scenarios**:

1. **Given** a 0.5.3 scene, **When** it renders, **Then** its unbatched count falls and its
   appearance is unchanged.
2. **Given** the same for LK, Cata and MoP scenes, **Then** the same holds.

---

### User Story 3 - Transparent instances batch too (Priority: P2)

The 260 unbatched transparent draws are addressed, within the ordering constraints transparency
imposes.

**Why this priority**: Two orders of magnitude smaller than the opaque count, and constrained
by sort order. Real, but not where the win is.

**Independent Test**: Compare transparent batched/unbatched counts on a fixed route.

**Acceptance Scenarios**:

1. **Given** transparent instances that can share a draw without violating sort order, **When**
   the frame submits, **Then** they are batched.
2. **Given** batching would change draw order, **When** the frame submits, **Then** order wins
   and the instance stays unbatched, counted with that reason.

---

### Edge Cases

- Animated instances whose bone state differs per instance — must not be silently merged.
- A model visible through both a WMO doodad set and a terrain MDDF placement in the same frame.
- Instances differing only by fade alpha or by fog parameters.
- A batch large enough to exceed a uniform or instance-buffer limit — must split, not overflow.
- The `MDX 13180/20115 (429 ok/0 fail)` gap: ~7,000 placed models are not drawn and nothing
  reports a failure. Whether they are culled, unloaded, or lost is unknown and must not be
  assumed benign.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: One component MUST decide batchability, from render state, for every model
  instance regardless of applied route or era.
- **FR-002**: The batch key MUST be derived from the state that actually makes two draws
  combinable; anything excluded from the key MUST be justified in the design.
- **FR-003**: A route that cannot batch MUST report a reason. "No batch path implemented" is a
  valid reason and MUST be distinguishable from "state differs".
- **FR-004**: Batching MUST NOT change rendered appearance. Any visible difference is a defect.
- **FR-005**: Batching MUST NOT change draw order where order is observable.
- **FR-006**: The system MUST be switchable at runtime for same-session before/after
  comparison, matching spec 153 US3's toggle behaviour.
- **FR-007**: Per-path, per-era batched/unbatched/unbatchable counts MUST be reported, building
  on spec 201's attribution rather than a second scheme.
- **FR-008**: The change MUST NOT regress 0.5.3 rendering, which is this project's primary lane.

### Key Entities

- **Batch key**: The render state that determines whether two instances can share a draw.
- **Batchability decision**: Per instance — batched, unbatched-by-state, or
  unbatchable-with-reason.
- **Batch**: A set of instances sharing one draw, with its key and instance count.
- **Route/era coverage report**: Batched vs unbatched per applied route and per era.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the Wandering Isle baseline route, opaque unbatched draws fall substantially
  from 13,180, and the residual is attributed by reason.
- **SC-002**: Median frame time on that route improves measurably from the 84.44 ms baseline.
- **SC-003**: A scene per era (0.5.3, LK, Cata, MoP) shows reduced unbatched counts.
- **SC-004**: Capture comparison shows no appearance change on a reference scene per era.
- **SC-005**: Every remaining unbatched draw has a reason, so what is left is a list rather
  than a mystery.
- **SC-006**: Disabling batching reproduces the pre-change output exactly.

## Assumptions

- Spec 201 lands first and supplies per-path attribution; this spec consumes it.
- Spec 153 US3's existing legacy-MDX batching is absorbed into the unified decision rather
  than kept as a parallel mechanism — one owner, per Constitution II.
- Spec 136's remaining tasks are operator measurement and are unblocked by this work rather
  than blocking it.
- Flights and capture comparisons are operator-run.
- The magnitude of the win is **not** assumed. SC-001 requires the residual to be explained,
  so a smaller-than-hoped improvement is a reportable result rather than a failure.
