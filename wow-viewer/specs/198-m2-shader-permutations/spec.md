# Feature Specification: M2 and WMO Shader Permutation System

**Feature Branch**: `198-m2-shader-permutations`

**Created**: 2026-09-01

**Status**: Draft

**Input**: User description: "M2/WMO shader permutation system: implement the client's vertex (Diffuse_*) x pixel (Combiners_*) shader pair selection for M2 doodads and WMO, replacing the single hardcoded program, to cut per-batch CPU material work and render doodads with correct texture combiners."

## Context

Measured against `Wow.exe` 5.0.1.15464 on 2026-09-01 and recorded in
[143-world-context-lighting/research.md](../143-world-context-lighting/research.md) section 3.

The client selects a **shader pair** per render batch. Both halves are named in the binary:
16 vertex programs (`Diffuse_T1`, `Diffuse_T1_Env_T2`, `Diffuse_EdgeFade_T1_T2`, …
`0x00d722c4`–`0x00d72858`) and 31 pixel programs (`Combiners_Mod`,
`Combiners_Opaque_Mod2xNA_Alpha_UnshAlpha`, `Combiners_Mod_Masked_Dual_Crossfade`, …
`0x00d7286c`–`0x00d72bcc`). WMO carries its own six (`MapObjDiffuse`, `MapObjSpecular`,
`MapObjTwoLayerDiffuse`, `MapObjDiffuseEmissive`, `MapObjTwoLayerDiffuseOpaque`,
`MapObjTwoLayerDiffuseEmissive`, `0x00dee4b4`–`0x00dee5c8`).

Our renderers have **no permutation system**. A search across `src/` for `Combiners_` or
`Diffuse_T1` returns nothing. Every M2 batch goes through one program
(`M2Renderer.cs`), and materials are flattened to a three-value classification —
`Opaque` / `Cutout` / `Blended` (`M2MaterialPassProfile.M2PassClass`). The per-batch
texture-combiner semantics the client resolves once, at shader-select time, are either
approximated on the CPU or not applied at all.

This spec covers **shader selection and the combiner semantics that follow from it**. It does
not cover lighting inputs (specs to follow) or draw-call batching (spec 136, 9/11, remaining
tasks are operator measurement; and spec 153 US3, which owns opaque MDX batching).

**Do not treat this spec as a performance fix.** An operator flight over a 5.0.1 map on
2026-09-01 measured the renderer at median 84.44 ms / 7 FPS with **13,180 opaque draw calls,
100% unbatched**. That is the bottleneck and it is not shader selection.

Two corrections to an earlier reading of that number, both recorded in spec 201: the counter
labelled `MDX` actually aggregates **both** the M2 and MDX render paths, and M2 instances on
the native runtime route are **structurally unbatchable** today
(`M2Renderer.RequiresUnbatchedWorldRender` returns true whenever there is no inner legacy
renderer). So spec 153 US3's toggle does **not** necessarily fix those draws, and how many of
them it can fix is not yet known. This spec is about rendering doodads *correctly*.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Doodads render with the material semantics the client uses (Priority: P1)

An operator loads a map with dense doodads and sees each model's material behave the way it
does in the real client: environment-mapped metal reflects, edge-faded foliage fades at
silhouettes, dual-crossfade materials blend between their layers, and two-layer materials
show both layers rather than one.

**Why this priority**: This is the visible defect. Everything else in this spec is a
consequence of selecting the right program. Without it, doodads are wrong regardless of how
fast they draw.

**Independent Test**: Load a scene containing at least one model per resolved permutation and
compare captures against the same scene in the real client. Delivers correct doodad
appearance with no other part of this spec implemented.

**Acceptance Scenarios**:

1. **Given** an M2 batch whose material resolves to an environment-mapped permutation,
   **When** the scene renders, **Then** the batch is drawn by a program implementing that
   permutation's combiner, not the generic program.
2. **Given** an M2 batch whose material resolves to a permutation that is not yet
   implemented, **When** the scene renders, **Then** the batch falls back to the current
   generic program, renders without error, and the unimplemented permutation is reported.
3. **Given** a model that renders correctly today, **When** the permutation system is
   enabled, **Then** its appearance does not regress.

---

### User Story 2 - Per-batch material work is measured, not assumed (Priority: P3)

An operator profiling a doodad-dense scene can see what per-batch material and combiner setup
actually costs on the CPU, before and after permutation selection, and the answer is recorded
whether or not it is favourable.

**Why this priority**: **Demoted from P2 to P3 on 2026-09-01 by measurement.** A real flight
over a 5.0.1 map (Wandering Isle, 2048 frames, median 84.44 ms, CPU 129.0 ms, 7 FPS) reported
**13,180 opaque MDX draw calls, 100% unbatched — one draw call per instance**. Draw-call count,
not per-batch combiner work, is where the CPU is going. Spec 153 US3 already owns that fix and
it exists behind a runtime toggle. This spec's value is correctness and appearance (US1); the
CPU effect here may be small, and claiming otherwise ahead of the measurement is how a spec
acquires a success criterion it cannot meet.

**Independent Test**: Capture per-stage CPU timings for a fixed camera and scene before and
after, using the measurement path spec 136 established. Requires US1.

**Acceptance Scenarios**:

1. **Given** a fixed scene and camera, **When** CPU stage timings are captured before and
   after enabling permutation selection, **Then** per-batch material setup CPU time is
   reported for both and the comparison is recorded — including a null or negative result.
2. **Given** the permutation set grows, **When** program count increases, **Then** shader
   compilation happens once per permutation rather than per batch or per frame.

---

### User Story 3 - WMO uses its own shader set (Priority: P3)

WMO surfaces render through the six map-object programs the client names, so specular and
emissive surfaces inside buildings stop being flattened to diffuse.

**Why this priority**: Smaller, well-bounded set, and independent of the M2 work. Deferred
because doodads are the operator's stated priority and carry the larger visible error.

**Independent Test**: Load a WMO containing specular and emissive materials and compare
against the real client. Independent of US1 and US2.

**Acceptance Scenarios**:

1. **Given** a WMO material declaring a specular or emissive shader, **When** the group
   renders, **Then** the corresponding map-object program is used.

---

### Edge Cases

- A model declares a shader index outside the known permutation table — must fall back and be
  reported, never crash or silently render as a wrong permutation.
- A permutation resolves but the model supplies fewer texture units than it needs.
- Era differences: 0.5.3/LK-era models predate this shader table. They must keep rendering
  exactly as they do today; permutation selection must not be applied to eras that do not
  carry the data.
- Shader compilation fails on a driver for one permutation — that permutation falls back
  without taking down the frame.
- A batch's resolved permutation changes between frames (animated material) — selection must
  not thrash program state per draw.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST resolve a vertex/pixel program pair per render batch from the
  batch's own material data, rather than applying one program to all batches.
- **FR-002**: The resolved pair MUST be derived from data present in the model file. Where a
  field's meaning is not established by measurement, the system MUST report it as
  unresolved rather than assign a guessed permutation.
- **FR-003**: The system MUST provide a fallback program for any batch whose permutation is
  unresolved or unimplemented, so that no batch fails to render.
- **FR-004**: The system MUST report which permutations were requested, which were
  implemented, and which fell back — as counts per frame or per scene load, queryable
  without a debugger.
- **FR-005**: Programs MUST be compiled at most once per permutation for the lifetime of the
  renderer, not per batch, per frame, or per model.
- **FR-006**: The system MUST NOT alter rendering for eras whose models do not carry the
  shader table; those batches continue through the existing path unchanged.
- **FR-007**: The permutation path MUST be switchable off at runtime, returning rendering to
  the current single-program behaviour, so parity and performance can be compared in one
  session.
- **FR-008**: Enabling the permutation path MUST NOT change draw-call batching or culling
  behaviour established by spec 136.
- **FR-009**: The system MUST record, for each implemented permutation, the native name it
  corresponds to (e.g. `Combiners_Opaque_Mod2xNA_Alpha`) so implementations remain traceable
  to the measured table.
- **FR-010**: WMO surfaces MUST resolve among the six map-object programs, with the same
  fallback and reporting rules as M2.

### Key Entities

- **Shader permutation**: A named vertex program paired with a named pixel program, matching
  a name in the client's table. Carries its native name, the texture-unit count it requires,
  and whether it is implemented.
- **Permutation request**: What one render batch asks for — the resolved pair, or an
  indication that it could not be resolved.
- **Permutation registry**: The set of compiled programs, keyed by permutation, with the
  fallback and the per-permutation implemented/unimplemented status.
- **Selection report**: Per-scene counts of requested, implemented, fallen-back, and
  unresolved permutations, with the offending model paths for the last two.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a named reference scene, every M2 batch either renders through a program
  matching its resolved permutation or is counted in the fallback report — no batch is
  silently drawn by the wrong permutation.
- **SC-002**: Side-by-side captures of the reference scene against the real client show the
  material behaviours that motivated this work — environment mapping, edge fade, dual
  crossfade, two-layer blending — present rather than absent.
- **SC-003**: No model that renders correctly before the change renders differently after it,
  except where the difference is a correction verified against the real client.
- **SC-004**: Per-batch material setup CPU time in the reference scene is measured before and
  after, and the change is recorded with the measurement method. **This is a measurement, not
  a target** — no reduction is claimed or required. Measured 2026-09-01: the dominant renderer
  cost on a real 5.0.1 flight is 13,180 unbatched opaque MDX draw calls, which spec 153 US3
  owns, not shader selection.
- **SC-005**: Turning the permutation path off restores the pre-change rendering exactly,
  verified by pixel comparison of the reference scene.
- **SC-006**: The fallback report names every permutation the reference corpus requested but
  which is not implemented, so the remaining work is enumerable rather than estimated.

## Assumptions

- The permutation table measured from 5.0.1.15464 applies to the 4.3.4–5.1 range this
  project targets for split ADTs. Earlier eras are handled by FR-006 rather than assumed
  compatible.
- The reference scene and real-client comparison captures are operator-run, consistent with
  this project's rule that visual proof against a real client belongs to the operator.
- Spec 136's measurement path is available for SC-004; this spec does not build a second one.
- Implementing all 16 x 31 M2 combinations is not required. The corpus determines which
  permutations actually occur; FR-004 and SC-006 make the remainder enumerable, and the
  unimplemented ones fall back safely.
- Lighting inputs to these programs remain whatever the renderer supplies today. Replacing
  the single directional light with the client's indexed light set is deliberately a separate
  spec, sequenced after this one so the shaders are written once.
