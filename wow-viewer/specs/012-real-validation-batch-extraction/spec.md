# Feature Specification: Real Validation Batch Extraction

**Feature Branch**: `012-real-validation-batch-extraction`

**Created**: 2026-05-23

**Status**: In Progress

**Input**: User direction: the real renderer-truth path is the existing MdxViewer validation batch, not the preview-only `WorldGpuPreviewRenderer` path in `WowViewer.App`. The next preparation step should use Spec Kit and turn the extraction direction into a concrete implementation-ready checklist.

## Problem Statement

The current renderer-truth capture lane for terrain dataset work is real, but it is still owned by the legacy MdxViewer app surface.

That legacy path already does the hard parts that matter:

- it renders actual world terrain plus world objects
- it emits multiple capture families rather than one pretty screenshot
- it waits for streaming and deferred object loading to settle before capture
- it derives downstream artifacts such as `object_visibility_mask` and `no_object_minimap`

The problem is not that renderer-truth capture does not exist. The problem is that the only proven path still depends on the old app shell, while the forward ownership belongs in `wow-viewer`.

A preview-only terrain renderer is not sufficient for this lane. The replacement must preserve the real validation-batch behavior rather than substituting a simplified top-down preview.

## Goal

Create a `wow-viewer`-owned headless validation-capture contract that reproduces the real renderer-truth batch behavior closely enough to replace the current legacy-driven capture lane for bounded dataset validation and later batch automation.

The first target is not full viewer parity. The first target is a faithful headless validation-capture slice that can drive the real world renderer, produce the real capture families, apply deterministic policy overrides, and emit the same downstream artifact family used by the current dataset workflow.

## Current Implementation Status

- Phases 1 through 4 are now landed for the bounded proof surface.
- `ValidationWorldSceneAdapter` owns the current hidden-window OpenGL render/readback path behind `IValidationWorldSceneAdapter`.
- `WowViewer.Tool.ValidationCapture capture --gpu-viewer-style` has bounded real-data proof on:
	- staged `0_5_3_3368 / Azeroth_30_48`
	- staged `3_3_5_12340 / Azeroth_30_48`
- Both anchors currently complete `4/4` variants for:
	- `primary`
	- `noliquids`
	- `noobjects`
	- `objectsonly`
- Phase 5.1 is now also landed for the bounded proof surface: the tool emits compatible `images/<tile>_object_visibility_mask.png` and `images/<tile>_no_objects.png` artifacts under the dataset root.
- The next open work is broader cutover, not first artifact emission: keep docs honest about what remains legacy-only, replace the current broader MdxViewer batch automation, and later remove the temporary `WorldGpuPreviewRenderer` backend reuse.

## User Scenarios & Testing

### User Story 1 - One Tile Can Be Captured Headlessly With Real Renderer Behavior (Priority: P1)

A terrain researcher wants to run renderer-truth capture for one known tile without launching or depending on the legacy MdxViewer shell, while still getting the real world-rendered result rather than a simplified terrain preview.

**Why this priority**: If one tile cannot be reproduced faithfully, there is no basis for batch or dataset-pipeline cutover.

**Independent Test**: A bounded single-tile run produces the expected validation families for one known proof tile using the `wow-viewer`-owned headless path.

**Acceptance Scenarios**:

1. **Given** a staged client root, a build label, a map, and a tile coordinate, **When** the new headless validation path runs, **Then** it produces the same bounded capture-family set expected by the legacy validation batch.
2. **Given** a tile containing visible WMOs and doodads, **When** the headless path renders it, **Then** the captured output reflects real world-object rendering rather than terrain-only preview markers or placeholders.
3. **Given** the bounded proof anchors already used by the current workflow, **When** the new path is compared against them, **Then** it is credible enough to serve as the first replacement candidate for that bounded proof surface.

---

### User Story 2 - Capture Timing Is Deterministic And Does Not Depend On Manual Delay Guessing (Priority: P1)

A terrain researcher wants the capture path to wait for the world scene to be ready instead of guessing when streaming and deferred loading might be done.

**Why this priority**: The current legacy batch already encodes settle logic. Replacing it with fire-and-pray capture timing would regress the only credible renderer-truth lane.

**Independent Test**: A bounded headless run exposes readiness or timeout outcomes that match the intended settle semantics.

**Acceptance Scenarios**:

1. **Given** a tile is still streaming or world-object loads are pending, **When** the headless capture path evaluates readiness, **Then** it waits rather than capturing immediately.
2. **Given** the target framebuffer is not yet large enough for the requested validation resolution, **When** readiness is evaluated, **Then** the capture does not complete until the framebuffer is valid or a timeout is reported.
3. **Given** readiness does not converge in time, **When** the bounded capture run ends, **Then** the result reports a timeout outcome explicitly instead of silently writing misleading output.

---

### User Story 3 - Dataset Workflow Can Consume Replacement Artifacts Without Invoking MdxViewer (Priority: P2)

A terrain researcher wants the dataset pipeline to keep using renderer-truth artifacts such as `object_visibility_mask` and `no_object_minimap`, but they want the generating tool to live in `wow-viewer` rather than the legacy viewer host.

**Why this priority**: The actual workflow value is not a screenshot command. The value is replacing the legacy batch as a dataset dependency.

**Independent Test**: The new headless path emits downstream artifact families that the existing dataset side can consume after a bounded integration step.

**Acceptance Scenarios**:

1. **Given** the new capture families are present for a bounded tile set, **When** downstream artifact derivation runs, **Then** it can produce `object_visibility_mask` and `no_object_minimap` equivalents without requiring the legacy app.
2. **Given** build-specific artifact policy differs between early and later client eras, **When** downstream derivation runs, **Then** the build-policy branch remains explicit rather than implicit.
3. **Given** the `wow-viewer` path only covers one bounded proof surface at first, **When** continuity docs describe the result, **Then** they state that proof boundary honestly instead of implying full pipeline cutover.

### Edge Cases

- What happens when terrain tiles are loaded but deferred world-object loads are still pending? The readiness contract must keep those states separate.
- What happens when requested output resolution exceeds the currently valid framebuffer size? The capture path must not silently write undersized output.
- What happens when early client builds and later client builds require different object-artifact derivation policy? The build-policy branch must remain explicit and testable.
- What happens when only a bounded subset of builds is proven initially? The contract must preserve that proof boundary instead of implying whole-corpus closure.
- What happens when some capture variants succeed and others time out? The batch contract must surface per-variant outcomes rather than assuming all-or-nothing success.

## Requirements

### Functional Requirements

- **FR-001**: `012-real-validation-batch-extraction` MUST define the next bounded replacement lane for legacy renderer-truth capture ownership.
- **FR-002**: The first `wow-viewer` replacement slice MUST target the real validation-batch behavior rather than a preview-only terrain renderer.
- **FR-003**: The replacement contract MUST support the same bounded capture-family concept currently used by the legacy validation batch, including primary and object-suppressed variants.
- **FR-004**: The replacement contract MUST include deterministic scene-policy overrides for bounded validation capture so repeated runs do not depend on ad hoc operator state.
- **FR-005**: The replacement contract MUST include explicit readiness evaluation for streaming, object loading, framebuffer validity, and settle-frame timing before capture completes.
- **FR-006**: The replacement path MUST be able to render real terrain and real world-object content for the bounded proof surface rather than placeholder or marker-only object output.
- **FR-007**: The first bounded replacement slice MUST expose headless execution suitable for scripted validation use.
- **FR-008**: The replacement lane MUST preserve downstream artifact ownership for `object_visibility_mask` and `no_object_minimap`, whether generated in the same tool or in a bounded follow-up step.
- **FR-009**: The contract MUST define explicit per-run or per-variant outcomes so timeouts and partial success do not look like clean proof.
- **FR-010**: The first bounded replacement slice MUST preserve honest proof language about which builds and tiles have real validation parity evidence.
- **FR-011**: Operator-facing planning for this lane MUST distinguish shared validation-batch contracts from the headless executable host that consumes them.
- **FR-012**: Continuity docs MUST route future renderer-truth extraction work to this spec instead of the preview-only world path.

### Key Entities

- **Validation Capture Batch**: one requested set of tile captures plus deterministic capture policy and result reporting.
- **Validation Capture Variant**: one visibility family within the batch, such as the primary image or an object-suppressed image.
- **Validation Scene Policy**: the deterministic render and visibility overrides that make validation capture reproducible.
- **Validation Readiness State**: the reported state of tile loading, object loading, framebuffer readiness, and settle progress before capture.
- **Derived Validation Artifact Set**: downstream artifacts such as `object_visibility_mask` and `no_object_minimap` produced from the captured families.
- **Bounded Proof Surface**: the specific build and tile anchors on which replacement parity is proven before broader expansion.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A Speckit package exists that defines the real validation-batch extraction lane as a dedicated feature rather than a loose architecture note.
- **SC-002**: The plan breaks the lane into independently validatable steps that do not depend on the fake preview path.
- **SC-003**: The tasks file names exact `wow-viewer` target projects and files for the first implementation slice.
- **SC-004**: The first implementation slice is small enough to prove one bounded tile with real renderer-truth family output before any wider batch automation, and that bounded proof now exists on the staged `0_5_3_3368` and `3_3_5_12340` anchors for `Azeroth_30_48`.
- **SC-005**: Continuity surfaces explicitly treat the legacy MdxViewer validation batch as the source reference and the new `wow-viewer` headless path as the replacement target.
