# Feature Specification: Per-Placement WMO Shell Instancing Under Scene Lights

<!-- reconciliation-2026-09-23 -->
> **ARCHIVED 2026-09-23 — open residue folded into an epic.** Entire spec open. Successor: [Epic 249](../../249-epic-renderer-performance-and-correctness/spec.md). Status lines and checkboxes below are historical and were audited against the code ([audit](../reconciliation-2026-09-23/audit/batch-C2.md)); they are not implementation authority.

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-18

**Status**: Draft (operator-directed 2026-09-18). Opened from an operator-reported regression on the
v0.6.0-alpha build; **not** implemented in that build.

**Depends on**: [Spec 236](../236-scene-lighting-doodad-performance/spec.md) (emitted-light scene
lighting, which introduced the gate), [Spec 239](../239-modern-client-assets/spec.md) /
[Spec 240](../240-format-conformance/spec.md) (the modern data surfaced by the defect)

**Input**: operator report on the v0.6.0-alpha build — performance is "atrocious with the new support
of the newer data": ~5.5 FPS, ~230 ms uncapped frame, `WMO draw calls 16,431`, WMO pass ≈5,493 ms of
7,716 ms, on `wow_classic_beta` 1.60.1 `Azeroth` in the viewer.

## Context

Spec 236 Phase 2 added scene-emitted point lights (MDX `LITE`, M2 `LITE`, WMO `MOLT`, WMO-internal
doodad lights) and, to avoid uploading one approximated light set for a whole batch, disabled WMO
shell GPU instancing whenever **any** scene light is active. The gate is a whole-scene test:
`_sceneLightManager.Count == 0` (`WorldScene.cs`, WMO opaque batching candidate).

Modern 1.60.1 surfaces carry emitted lights almost everywhere, so the gate is effectively always
false and **every** WMO opaque placement takes the per-placement fallback path. Measured result on the
reported map: 16,431 WMO draw calls and a ~230 ms frame.

The per-placement test the gate *should* use already exists: the shell lighting path transforms the
placement's world AABB and asks `SceneLightManager.QueryAffecting(boundsMin, boundsMax, …)`, which
returns zero when no light's attenuation reaches the placement (`WmoRenderer`). This spec moves the
batching decision from "is any light active anywhere" to "does any light reach this placement".

## User Scenarios & Testing

### User Story 1 — Unlit WMO placements batch again (Priority: P1)

A user opens a modern CASC map and flies over ordinary buildings that no light reaches. Those
placements are submitted through the instanced batch path as they were before scene lighting landed,
while buildings inside a light's reach keep per-placement light evaluation.

**Why this priority**: it is the reported regression; without it the v0.6 modern-data lane is not
usable at interactive frame rates.

**Independent Test**: on an unlit stretch of the reported map, WMO draw calls fall to the pre-236
baseline level, and a placement inside a light's attenuation radius still receives that light.

**Acceptance Scenarios**:

1. **Given** a frame in which no active light's attenuation reaches a WMO placement, **When** the
   frame renders, **Then** that placement is submitted through the instanced batch path.
2. **Given** an active light whose attenuation sphere intersects a placement's world bounds, **When**
   the frame renders, **Then** that placement uses the per-placement path and receives the light.
3. **Given** identical lighting and placement state, **When** the frame renders twice, **Then** the
   batched/unbatched partition is identical.

### User Story 2 — No lighting regression (Priority: P1)

Shading of lit geometry is unchanged from the current build.

**Independent Test**: capture a scene containing a light in both builds; the lit placement's shading
matches.

### User Story 3 — Measured improvement with a receipt (Priority: P2)

The change ships with before/after frame-time and draw-call numbers on the same map and camera.

**Independent Test**: operator capture pair with the frame-history counters recorded in the spec's
`evidence/`.

## Requirements

- **FR-001**: The WMO opaque instancing decision MUST be made per placement, from whether any active
  light's attenuation reaches that placement's world bounds.
- **FR-002**: A placement reached by one or more active lights MUST keep the current per-placement
  light path; its lighting MUST NOT be approximated by a shared batch light set.
- **FR-003**: A placement no active light reaches MUST be eligible for the instanced batch path
  whenever the renderer otherwise supports GPU-instanced opaques.
- **FR-004**: The batched/unbatched partition MUST be deterministic for identical lighting and
  placement state.
- **FR-005**: The change MUST NOT alter terrain, M2/MDX, portal, or format-reader/writer behavior.
- **FR-006**: New logic MUST follow AGENTS.md §10 — no new members in the `WorldScene` or `ViewerApp`
  god classes; use an owned service or an existing service method.
- **FR-007**: A receipt MUST be recorded (files, exact commands, exit status, and a criterion→evidence
  table) including a before/after measurement on real `wow_classic_beta` data. Build/test output alone
  does not satisfy the runtime/FPS criteria.

## Success Criteria

- **SC-001**: On the reported map and camera, WMO draw calls fall from the measured ~16,431 to a level
  comparable with the pre-236 baseline for unlit geometry, with unchanged visible content.
- **SC-002**: Frame time on that map and camera improves measurably (target: an operator-observed
  improvement of at least 2× FPS from the reported ~5.5), recorded in the receipt.
- **SC-003**: Lit placement shading is visually unchanged in a side-by-side capture.
- **SC-004**: The viewer build is clean and focused tests pass with no new failures.

## Key Entities

- **Scene light**: an active emitted point light (position, colour, intensity, attenuation start/end).
- **WMO placement**: one placed WMO instance with a world-space bounds box and a model key.
- **Batch candidate**: a WMO placement plus its eligibility for the instanced path.

## Assumptions & open questions

- The whole-scene gate is the dominant cost. Other modern-data costs (8 terrain layers, per-shader WMO
  materials) stay unoptimized until measured separately.
- Whether a batch of placements that are lit *identically* could still be instanced with a shared light
  set is deliberately out of scope; correctness first.
- Operator note (2026-09-18): refresh the vendored libraries and re-read the wowdev.wiki for new
  documentation of this build's makeup before deeper format work. That research is separate from this
  fix and does not gate it.
