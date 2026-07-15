# Feature Specification: Native Day/Night Lighting Fidelity

**Feature Branch**: `106-native-daynight-lighting`

**Created**: 2026-07-15

**Status**: Planned

**Input**: Match the viewer and synthetic terrain capture path to the exact native world day/night lighting system, so one minimap input can be reconstructed reliably across supported time-of-day appearances.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reproduce a selected native outdoor lighting state (Priority: P1)

A user selects an exact supported client build, map, and time of day. The viewer presents the same outdoor sun direction, direct and ambient appearance, fog, and sky data source as the native client, with provenance showing where every value came from.

**Why this priority**: Synthetic terrain images are only useful when their appearance follows the real rendering system rather than an undocumented approximation.

**Independent Test**: For a selected build and global outdoor light, evaluate the same declared times in the viewer and in the native client, then compare the recorded direction and color/fog state and a controlled top-down image.

**Acceptance Scenarios**:

1. **Given** an exact-build global lighting source and a declared time of day, **When** the viewer renders an outdoor terrain tile, **Then** it uses the build's timed color, fog, sky, and world-light direction contracts without mixing sources.
2. **Given** a build whose direction model has not been recovered, **When** a user requests client-exact lighting, **Then** the viewer rejects that request rather than silently substituting an authored sun direction.

---

### User Story 2 - Build time-diverse synthetic examples safely (Priority: P1)

A dataset author can generate related clean-synthetic terrain examples across declared times of day, knowing that each row records its renderer and lighting provenance and cannot be confused with an already-lit minimap.

**Why this priority**: The image-only terrain model must learn from appearance variation without accidentally treating an arbitrary lighting approximation or a double-lit capture as ground truth.

**Independent Test**: Generate a source group at multiple declared times, inspect its sidecars and split assignment, and verify that malformed or mixed-source records are rejected.

**Acceptance Scenarios**:

1. **Given** an unlit owned/licensed terrain source, **When** multiple lighting variants are produced, **Then** they share one source-group identity and one train/validation partition while retaining their individual time and lighting provenance.
2. **Given** an already-lit minimap, missing provenance, or a stale capture sidecar, **When** it is submitted to the store builder, **Then** the builder rejects it.

---

### User Story 3 - Calibrate the viewer coordinate transform once per supported renderer path (Priority: P2)

A maintainer can perform one controlled native/viewer top-down comparison that proves the mapping between the native world-light vector and the viewer terrain coordinate system; the resulting calibrated mapping is reusable for all times in that build path.

**Why this priority**: Native direction recovery supplies a client-space vector. A single empirical comparison is still required to prove the viewer axis/sign transform, but it is not a search for an unknown sun azimuth.

**Independent Test**: Use a terrain tile with visible directional relief and MCSH, compare a native and viewer capture at one declared time, and show that the locked transform predicts the same shadow/light orientation at at least two other declared times.

**Acceptance Scenarios**:

1. **Given** the exact 0.5.3 native timed direction model, **When** a calibration capture is accepted, **Then** the viewer stores a versioned coordinate transform and its evidence rather than an arbitrary per-time direction.
2. **Given** a proposed transform that fails a held-out time comparison, **When** validation runs, **Then** the profile remains uncalibrated and is unavailable for client-exact capture.

---

### User Story 4 - Keep shadow systems distinct (Priority: P2)

A viewer user sees terrain lighting driven by the moving day/night world light while dynamic/unit shadow projection remains an explicitly separate fixed-angle system.

**Why this priority**: Reusing the fixed shadow-projection constant for the world sun creates incorrect time-of-day images and invalid synthetic supervision.

**Independent Test**: Evaluate the world-light state at multiple times and assert that its elevation changes while the optional dynamic-shadow projection setting remains fixed.

**Acceptance Scenarios**:

1. **Given** a selected time of day, **When** terrain lighting is evaluated, **Then** the world light uses the timed direction model rather than the fixed shadow-projection angle.
2. **Given** dynamic/unit shadows are enabled, **When** their projection is evaluated, **Then** their fixed-angle setting is recorded independently from the world-light state.

### Edge Cases

- A light file has colors but no recovered exact-build direction model: client-exact capture fails closed; a visibly labeled authored profile may still be used only where authored output is permitted.
- A native capture cannot be made because a sandbox/map is unavailable: calibration remains pending; no coordinate transform is promoted from guesswork.
- A map has local/zone light records whose spatial transform is not proven: only the unique global/default clear group is eligible for client-exact minimap capture.
- A time value is outside the native day cycle or not finite: evaluation rejects it rather than extrapolating silently.
- A source image is already lit or has no rights/provenance assertion: it is never re-lit for clean synthetic training data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST represent world-light direction separately from direct color, ambient color, fog, sky, and shadow-projection settings.
- **FR-002**: For each supported client build, client-exact world-light direction MUST come from a versioned, evidence-bound direction model; it MUST NOT be inferred from LIT or DBC color records.
- **FR-003**: The exact 0.5.3 model MUST preserve its recovered constant light-ray azimuth of 225 degrees, its timed polar-angle samples of 110 and 127 degrees, and the documented conversion between light-ray and source direction.
- **FR-004**: The viewer MUST use a single versioned native-to-viewer coordinate transform for every time evaluated under a calibrated profile, and MUST retain the capture evidence that calibrated it.
- **FR-005**: The fixed 45-degree shadow-projection constant MUST remain a distinct dynamic-shadow setting and MUST NOT supply the outdoor world-light direction.
- **FR-006**: A client-exact outdoor profile MUST use one coherent exact-build color/fog/sky source and MUST not silently mix LIT and DBC records.
- **FR-007**: Client-exact capture MUST be unavailable until both a direction model and its native-to-viewer transform are proven for the selected build/render path.
- **FR-008**: Every capture sidecar and synthetic dataset row MUST record build identity, time, direction-model revision, coordinate-transform revision, color-source identity, shadow mode, and source rights class.
- **FR-009**: Synthetic variants of the same underlying source MUST share a source-group identity and data partition; generated output MUST never be double-lit.
- **FR-010**: The capture/store pipeline MUST reject missing, stale, non-finite, mixed-source, or uncalibrated client-exact lighting provenance.
- **FR-011**: Tests MUST cover native timed direction evaluation, ray/source inversion, coordinate-transform application, world-light/shadow-projection separation, profile fail-closed behavior, and sidecar provenance validation.
- **FR-012**: A controlled native/viewer image-comparison procedure MUST verify a calibrated transform at one lock time and at least two held-out times before that profile is considered client-exact.

### Key Entities

- **World-light direction model**: Evidence-bound timed native light-ray data and its documented source-direction interpretation for one client build.
- **Coordinate-transform calibration**: Versioned mapping from native world-light coordinates into the viewer terrain coordinate system, supported by controlled image comparisons.
- **Lighting profile**: Coherent selected source for timed direct/ambient/fog/sky values plus a direction model, shadow mode, evidence state, and revisions.
- **Capture provenance sidecar**: Hash-bound description of a rendered terrain image, including all lighting and source-data identities required for validation.
- **Source group**: All time-of-day variants rendered from one unlit source; kept together in one data split.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A calibrated supported profile evaluates finite, reproducible world-light vectors for 100% of declared cycle samples and reports its evidence revision for each result.
- **SC-002**: For the controlled calibration tile, the accepted transform matches native light/shadow orientation at the lock time and at least two held-out declared times, with comparison artifacts retained for review.
- **SC-003**: 100% of client-exact captures contain a complete, hash-bound lighting sidecar; intentionally missing, stale, mixed-source, or uncalibrated cases are rejected by automated validation.
- **SC-004**: A time-diverse synthetic source group never spans train and validation partitions and never accepts an already-lit input for re-lighting.
- **SC-005**: Automated tests demonstrate that varying world-light elevation across the 0.5.3 cycle cannot be replaced by the fixed 45-degree shadow-projection setting.

## Assumptions

- The user owns native-client capture execution and may run the controlled calibration comparison when the prepared procedure is ready.
- LIT is the exact early-build color source and the Light* DBC chain is the later-build color source; neither stores a world-light direction record.
- Exact direction recovery is build-scoped. The initial implementation target is 0.5.3.3368, with other builds added only after equivalent evidence is recorded.
- Local-zone lighting, sky-band altitude placement, and native MCSH attenuation remain separately proven follow-up work; they are not guessed as part of this feature.
- This feature links Spec 032's renderer parity and Spec 103's image-only synthetic provenance, without reopening their unrelated backlog.
