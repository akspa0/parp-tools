# Feature Specification: M2 Render Parity Recovery

**Feature Branch**: `035-m2-render-parity-recovery`

**Created**: 2026-06-01

**Status**: Draft

**Input**: User report: world M2 instances (especially trees and alpha-cutout doodads) are loaded and selectable but not consistently rendered after the MdxViewer -> WoWViewer migration. Bounding boxes prove placement and culling admission, but visual output is missing. The team needs a refactor plan that restores 3.x-era M2 render behavior without reintroducing migration drift.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stable World Doodad Visibility (Priority: P1)

As a world-view operator, I can load a 3.3.5 map tile and consistently see world M2 doodads (trees and transparent/cutout assets) where their placements and bounding boxes exist.

**Why this priority**: This is the blocking regression: current world exploration and validation are unreliable if placements render only as bounds.

**Independent Test**: Load a known 3.3.5 tile with high tree density and verify that visible M2 geometry appears for sampled placements that are currently selectable and bounded.

**Acceptance Scenarios**:

1. **Given** a staged 3.3.5 client and a tile with tree doodads, **When** the world scene loads and doodads are enabled, **Then** sampled tree M2 placements render visible geometry instead of only bounds.
2. **Given** a selected M2 placement in world scene, **When** the object is hovered/inspected, **Then** selection, bounds, and rendered mesh co-exist for the same placement.

---

### User Story 2 - Deterministic M2 Load/Render Routing (Priority: P1)

As a renderer engineer, I can trace and validate a single deterministic M2 world path (load -> skin select -> material/pass route -> draw) so regressions can be reproduced and fixed without guesswork.

**Why this priority**: The current seam has multiple fallback paths and drift between adapter/runtime/converter behavior; we need one proven route with explicit fallback contracts.

**Independent Test**: Run a probe on a target M2 model and produce stable diagnostics for skin selection, section/material classification, and pass assignment that match runtime behavior.

**Acceptance Scenarios**:

1. **Given** a world M2 model path and build profile, **When** a probe is run, **Then** it reports the exact selected skin, geometry summary, material blend classification, and declared pass routing.
2. **Given** the same model and inputs, **When** probe and runtime are executed repeatedly, **Then** they produce consistent route diagnostics (no random fallback oscillation).

---

### User Story 3 - Controlled Compatibility and Refactor Boundaries (Priority: P2)

As a maintainer, I can evolve M2 handling in `wow-viewer` with explicit compatibility boundaries so migration updates do not silently undo working world-render behavior.

**Why this priority**: The current pain pattern is repeated rewiring and rollback churn; a bounded architecture contract prevents accidental regressions.

**Independent Test**: Change one M2 seam (adapter, runtime renderer, or converter fallback policy) and verify contract checks fail fast when behavior leaves approved boundaries.

**Acceptance Scenarios**:

1. **Given** an M2 path policy change, **When** build/tests/probes run, **Then** contract checks identify route-level behavior drift before manual world QA.
2. **Given** a new M2 build-profile rule, **When** it lands, **Then** architecture docs and spec tasks reflect the new rule in the same change set.

---

### Edge Cases

- M2 models that have only alpha-cutout layers in world rendering.
- M2 models with missing or unresolved external `.skin` companions.
- M2 models with fallback conversion paths that produce geometry but divergent material semantics.
- Build-profile differences (early/legacy vs 3.3.5-style routing) with different skin/profile ownership.
- Models with zero bones or static sequences that still require correct material/pass behavior.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST render world M2 placements consistently with their admitted visibility and selection state for supported staged builds.
- **FR-002**: System MUST enforce a deterministic world M2 route contract that explicitly records load path, skin source, material classification, and pass routing.
- **FR-003**: System MUST preserve a compatibility fallback path for models that cannot use the primary route, and MUST expose when fallback is used.
- **FR-004**: System MUST provide regression diagnostics that can be executed headlessly against staged clients and specific model paths.
- **FR-005**: System MUST keep M2 route ownership in `wow-viewer` runtime/library surfaces and avoid reintroducing legacy cross-repo ownership.
- **FR-006**: System MUST define and enforce cutout/transparent world-pass semantics for adapted M2 materials so alpha-based assets do not disappear.
- **FR-007**: System MUST include bounded parity checks against known world samples (including tree-heavy tiles) before declaring the route stable.

### Key Entities *(include if feature involves data)*

- **M2RouteDecision**: Captures selected route type, selected skin candidate, fallback reason (if any), and build-profile context.
- **M2MaterialPassProfile**: Captures per-material blend/cutout class and resulting world render-pass assignment.
- **M2ParitySample**: Captures reference model path, placement context, expected render visibility state, and probe/runtime evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a defined 3.3.5 parity sample set, at least 95% of previously “bounds-only” world M2 placements render visible geometry after the fix.
- **SC-002**: Probe and runtime route diagnostics match for 100% of parity sample models in the same run configuration.
- **SC-003**: Route regressions are detectable by automated checks without requiring manual visual debugging for first detection.
- **SC-004**: M2 route ownership documentation and spec artifacts are updated in the same delivery that changes route behavior.

## Assumptions

- Staged clients under `output/tmp/wowarchive-clients/` remain the authoritative real-data source.
- Initial parity closure target is 3.3.5 world usage; broader build coverage follows after this recovery slice.
- Existing `WoWViewer` probe surfaces can be extended rather than replaced.
- Compatibility fallbacks remain necessary during transition, but must be explicitly declared and measured.
