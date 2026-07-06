# Feature Specification: 093 Render Performance And WMO Liquid Audit

**Feature Branch**: `093-render-performance-liquid-audit`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "Do a proper audit to figure out why viewer performance is terrible. Suspect asset instancing/batching is not happening. WMO liquids look opaque and terrible instead of transparent."

## User Scenarios & Testing

### User Story 1 - Identify The Actual Frame Hotspot (Priority: P1)

As a viewer user, I need the Runtime Stats surface to show enough object-render pressure to distinguish terrain, WMO, MDX, transparent, liquid, overlay, and asset-load costs, so performance fixes target the real bottleneck.

**Why this priority**: If the renderer is slow because WMO submission explodes into many group/material/liquid draws, fixing MDX batching first is wasted effort.

**Independent Test**: Load a dense city map, open Runtime Stats, and capture frame numbers for total CPU, WMO visibility/submission, WMO transparent submission, WMO draw composition, MDX visibility/submission, terrain, liquid, overlay, and asset-load cost.

**Acceptance Scenarios**:
1. **Given** a world map is loaded, **When** Runtime Stats is visible, **Then** WMO opaque and transparent submission costs are reported separately.
2. **Given** visible WMO instances are rendered, **When** Runtime Stats is visible, **Then** WMO batch, fallback, liquid, doodad, and group-submission counts are visible.
3. **Given** MDX transparent and WMO transparent work happen in the same sorted pass, **When** stats are reported, **Then** the WMO and MDX costs are not collapsed into a misleading MDX-only number.

### User Story 2 - Prove Or Reject The Batching Hypothesis (Priority: P1)

As a maintainer, I need the audit to show whether poor performance comes from missing instancing, per-group WMO draw calls, transparent sorting, asset loading, or overlays, so each fix can be validated independently.

**Why this priority**: The likely defect is not a single "slow renderer" but a set of submission patterns that need different fixes.

**Independent Test**: Compare Runtime Stats with object visibility enabled and disabled, WMO visibility enabled and disabled, and overlays disabled.

### User Story 3 - Make WMO Liquids Inspectable And Correctable (Priority: P1)

As a viewer user, I need WMO liquids to be measured and classified separately from terrain liquid and normal WMO shell rendering, so opacity and shader issues can be fixed without guessing.

**Why this priority**: Current WMO MLIQ rendering uses a simple flat color pass. It may be blending correctly at the GL state level while still looking opaque because the shader/material behavior is incomplete.

**Independent Test**: Load a WMO with MLIQ data and verify Runtime Stats reports WMO liquid draw count; then compare visual output before and after any shader/material change.

## Edge Cases

- Dense city maps may have many WMO placements using the same root WMO, many unique WMO roots, or many groups inside a few roots.
- Transparent WMO shell batches, WMO MLIQ surfaces, WMO doodads, and MDX transparent models are sorted together at scene level and must not be mislabeled.
- A WMO liquid can blend at the GL level and still look wrong if the shader is flat, too saturated, missing material settings, or drawn in the wrong order.
- GPU timing can differ from CPU submission timing; CPU counters are the first gate, not final GPU proof.

## Requirements

### Functional Requirements

- **FR-001**: Runtime Stats MUST separately report WMO opaque submission time and WMO transparent submission time.
- **FR-002**: Runtime Stats MUST report WMO draw-call pressure, including batch draws, fallback group draws, liquid draws, doodad submissions, and visible group submissions.
- **FR-003**: Runtime Stats MUST keep MDX transparent submission time scoped to MDX work only.
- **FR-004**: The audit MUST document whether the current "batched MDX" count means true GPU instancing or shared-shader submission.
- **FR-005**: The audit MUST identify whether WMO liquids are failing because of GL blend state, shader/material behavior, draw ordering, or missing data.
- **FR-006**: The first code slice MUST be diagnostic-only and MUST NOT change WMO batching architecture or liquid visuals.

### Key Entities

- **Frame Cost Snapshot**: CPU timings and submission counts for one rendered world frame.
- **WMO Draw Pressure**: Number of WMO batch draws, fallback group draws, liquid draws, doodad submissions, and visible group submissions.
- **WMO Liquid Pass**: The MLIQ render path inside `WmoRenderer`, separate from terrain liquid.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A dense map frame can be described with WMO opaque time, WMO transparent time, MDX opaque time, MDX transparent time, terrain time, liquid time, overlay time, and asset-load time.
- **SC-002**: A dense map frame can be described with WMO draw composition numbers, not only visible WMO placement count.
- **SC-003**: The next optimization slice is selected from measured Runtime Stats data rather than assumption.
- **SC-004**: The feature builds with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Assumptions

- The first real proof target should be staged `4_0_0_11927` Stormwind or another dense city map under `output/tmp/wowarchive-clients/`.
- NVIDIA Nsight Graphics is useful later for GL/GPU timing, but CPU submission and memory counters should be captured first.
- WMO liquid visual correctness is separate from terrain liquid correctness.
