# Feature Specification: Renderer Improvements Convergence

**Feature Branch**: `036-renderer-improvements`

**Created**: 2026-06-01

**Status**: Draft

**Input**: User description: "Converge renderer improvement plans 030 through 032 into a single wow-viewer renderer modernization feature plan." Updated 2026-06-01 to prioritize terrain/world rendering FPS improvement for live `3.3.5.12340` map views.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Renderer Owner Plan (Priority: P1)

As a maintainer, I can read one renderer-improvements feature pack that owns the active plan for terrain, WMO, lighting, sky, fog, liquid, and viewer integration work, instead of chasing overlapping partial plans across specs 030, 031, and 032.

**Why this priority**: Renderer work is currently split across parallel plans with overlapping scope. Without one owner, implementation order and proof boundaries will keep drifting.

**Independent Test**: Read `specs/036-renderer-improvements/spec.md` and `plan.md` and verify they define one owner plan, a phase order, and a traceable mapping back to specs 030-032.

**Acceptance Scenarios**:

1. **Given** a renderer engineer starts new work, **When** they open the convergence feature pack, **Then** they can identify the active owner plan and the correct phase to work on first.
2. **Given** a slice from spec 030, 031, or 032, **When** a maintainer traces it in the convergence plan, **Then** they can see which convergence phase owns it and whether it remains in scope.
3. **Given** the old source plans remain in the repo, **When** a reader opens them, **Then** they are clearly framed as source slices feeding the convergence owner plan.

---

### User Story 2 - Bounded Library-First Renderer Phases (Priority: P1)

As a renderer engineer, I can implement renderer modernization as bounded, dependency-ordered phases in `wow-viewer` libraries and viewer host surfaces, so terrain, WMO, lighting, and liquid work lands in a coherent order.

**Why this priority**: The renderer work is too broad to execute safely without a strict dependency order and clear ownership split between `Core.IO`, `Core.Runtime`, and viewer host code.

**Independent Test**: Review the convergence `plan.md` and confirm each phase has one concern, explicit dependencies, at most 10 steps, and real target paths under `wow-viewer/`.

**Acceptance Scenarios**:

1. **Given** a terrain-topology slice, **When** the engineer checks the convergence plan, **Then** it is scheduled after lighting foundations only where truly required and before terrain LOD or liquid polish work that depends on it.
2. **Given** a WMO pass-architecture slice, **When** the engineer checks the convergence plan, **Then** it lands under `WowViewer.Core.Runtime` ownership with viewer host wiring clearly separated.
3. **Given** a viewer-only debug or inspection surface, **When** it is planned, **Then** the convergence plan distinguishes it from shared runtime/library work and keeps the app host thin.
4. **Given** a runtime control surface extracted from 3.3.5 (`terrainLOD`, `mapObjLightLOD`, `terrainAlphaBitDepth`, `MaxLights`, `projectedTextures`, `waterLOD`, and M2 runtime optimization flags), **When** implementation phases are defined, **Then** each control has an owning layer and explicit proof checkpoint.

---

### User Story 3 - Live Terrain/World Frame Pacing (Priority: P1)

As an operator, I can move through live staged `3.3.5.12340` outdoor world scenes with stable terrain/world rendering performance, so the viewer remains usable during normal map exploration instead of collapsing under avoidable frame-time spikes.

**Why this priority**: Renderer modernization only helps if live world traversal remains responsive. Terrain/world frame pacing is now the first requested execution lane and needs explicit ownership, validation routes, and proof criteria.

**Independent Test**: Run the defined staged `3.3.5.12340` terrain/world traversal routes and verify that performance evidence distinguishes steady-state render cost from streaming spikes while showing measurable improvement against the pre-optimization baseline.

**Acceptance Scenarios**:

1. **Given** a staged `3.3.5.12340` outdoor route with terrain, WMOs, MDXs, and liquids in view, **When** the operator traverses the route after initial load settling, **Then** frame pacing remains within the documented live-view target instead of degrading into prolonged low-FPS stalls.
2. **Given** a dense terrain/world scene, **When** performance evidence is captured, **Then** the report separates steady-state world-render cost from deferred asset-load spikes.
3. **Given** a bounded performance slice lands, **When** the same validation route is replayed on the same machine, **Then** the measured terrain/world frame time improves by the documented threshold relative to the stored baseline.

---

### User Story 4 - Shared Validation and Out-of-Scope Boundaries (Priority: P2)

As an operator, I can validate each renderer-improvements phase against staged clients using repeatable proof surfaces, and I can tell which adjacent issues stay outside this convergence plan.

**Why this priority**: Without explicit validation and scope boundaries, renderer work tends to absorb unrelated regressions and gets declared done without ground-truth proof.

**Independent Test**: Review `quickstart.md` and verify it defines staged-client validation surfaces, evidence expectations, and explicit non-goals such as separate M2 parity recovery work.

**Acceptance Scenarios**:

1. **Given** a completed convergence phase, **When** the operator checks the validation notes, **Then** they can run the listed commands and compare the expected evidence against staged client output.
2. **Given** an M2-specific regression like route drift or animation parity, **When** the operator checks this convergence feature, **Then** it is identified as a dependency or adjacent track instead of being silently absorbed.
3. **Given** a proposed renderer change that crosses terrain, WMO, and viewer layers, **When** the maintainer checks the convergence plan, **Then** they can see which proofs must pass before moving to the next phase.
4. **Given** parity drift in a staged-client run, **When** diagnostics are reviewed, **Then** telemetry logs expose runtime control values (terrain/light/liquid/fog/M2 flags) before visual-comparison signoff.
5. **Given** a staged `3.3.5.12340` world with rivers/oceans and lava in the same validation route, **When** liquid rendering is compared against source classification evidence, **Then** river/ocean water does not render as magma due to hard-coded MCNK-flag-only assumptions.

---

### Edge Cases

- Specs 030, 031, and 032 contain overlapping scope but different proof owners and phase assumptions.
- Terrain topology and WMO pass architecture may have build-era differences between Alpha-era and 3.3.5 validation targets.
- M2 recovery work from spec 035 affects world-scene confidence but is not the owner plan for terrain/WMO modernization.
- Viewer host integration may require bounded `WowViewer.App` work without shifting canonical ownership away from shared libraries.
- Some source-plan steps may need to be deferred or dropped if they duplicate a lower-level prerequisite already covered elsewhere in the convergence plan.
- Runtime control surfaces from native 3.3.5 can silently alter output quality/perf (`terrainLOD`, `waterLOD`, `projectedTextures`, `mapObjLightLOD`, M2 optimization flags) and must be captured as explicit phase checkpoints.
- In `3.3.5`, MCNK liquid bits alone are not sufficient to guarantee final visual liquid-family selection; build-aware liquid type resolution can require per-build data-table interpretation.
- Cold-start routes can mix legitimate first-load stalls with avoidable steady-state render cost; validation must separate those two classes of slowdown.
- A dense city-edge route can be bound either by visibility admission or deferred asset-load churn, so performance evidence must show which lane owns the regression before optimization is declared successful.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST define a single active renderer-improvements owner plan in `specs/036-renderer-improvements/`.
- **FR-002**: The convergence feature MUST map the relevant work from specs `030-wmo-render-pass-architecture`, `031-terrain-cell-awareness`, and `032-native-renderer-parity` into one dependency-ordered phase plan.
- **FR-003**: The convergence plan MUST preserve library-first ownership boundaries across `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, and viewer host code.
- **FR-004**: The convergence plan MUST define explicit validation surfaces that use staged clients under `output/tmp/wowarchive-clients/`.
- **FR-005**: The convergence plan MUST state which adjacent work remains out of scope, including separate M2 recovery work tracked elsewhere.
- **FR-006**: The convergence plan MUST keep each phase independently validatable and limited to bite-sized steps.
- **FR-007**: The convergence plan MUST identify which source-plan documents remain reference inputs and which new document becomes the active owner.
- **FR-008**: The source plans in specs 030-032 MUST point readers to the new convergence owner plan.
- **FR-009**: The convergence plan MUST include a runtime-controls inventory (terrain, lighting, fog, liquid, projected-texture, and M2 optimization controls) derived from staged `3.3.5.12340` Ghidra evidence.
- **FR-010**: Each convergence phase MUST define telemetry checkpoints that record relevant runtime control values before visual parity signoff.
- **FR-011**: M2 runtime findings included in this feature MUST remain bounded to convergence dependencies and diagnostics; full M2 parity ownership remains with spec 035.
- **FR-012**: The convergence plan MUST include a build-aware liquid classification contract so staged `3.3.5` water/ocean/river surfaces are not misclassified as magma when MCNK flags are ambiguous or incomplete.
- **FR-013**: Liquid rendering validation MUST include per-build classification evidence inputs (such as liquid-type table lookups or equivalent build-resolved metadata), and MUST NOT rely on one hard-coded liquid-family mapping across all builds.
- **FR-014**: The convergence feature MUST define a terrain/world rendering performance lane for live staged `3.3.5.12340` map traversal.
- **FR-015**: Terrain/world performance validation MUST distinguish steady-state rendering cost from deferred asset-load or first-load streaming spikes.
- **FR-016**: Performance proof routes MUST include at least one sparse outdoor scene, one mixed outdoor scene, and one dense terrain/world scene under staged `3.3.5.12340`.
- **FR-017**: Performance evidence MUST record frame pacing plus world-scene workload indicators such as visible terrain coverage, visible object counts, and pending asset-load pressure for each proof route.
- **FR-018**: The first bounded performance slice under this feature MUST target one dominant live-view terrain/world cost lane at a time and define a before/after comparison route.
- **FR-019**: Terrain/world performance improvements MUST preserve existing visual ownership and validation boundaries instead of bypassing correctness checks for lighting, fog, liquid, or object visibility.

### Key Entities *(include if feature involves data)*

- **RendererConvergencePhase**: A bounded modernization phase with one primary concern, explicit dependencies, target code paths, and validation gates.
- **RendererCapabilitySlice**: A specific renderer behavior or subsystem carried over from specs 030-032, such as WMO pass dispatch, terrain topology, lighting evaluation, or liquid type routing.
- **SourceSpecMapping**: A traceable mapping from a source plan section or requirement in spec 030/031/032 to a convergence phase in spec 036.
- **RendererValidationScenario**: A staged-client proof case that defines map/build context, expected subsystem behavior, and evidence requirements.
- **RendererOwnershipBoundary**: A rule that states which layer owns a behavior: shared I/O, runtime, or viewer host.
- **LiquidClassificationEvidence**: Build-scoped liquid-family evidence used to resolve rendered liquid type (for example MCNK/MH2O fields plus per-build liquid-type table semantics).
- **TerrainWorldPerformanceRoute**: A repeatable staged `3.3.5.12340` traversal path used to compare live-view terrain/world frame pacing before and after a bounded optimization slice.
- **RendererPerformanceEvidence**: A proof artifact that combines frame pacing measurements with scene workload indicators so a regression can be attributed to steady-state rendering, visibility admission, or deferred asset loading.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One convergence spec and one convergence plan exist under `specs/036-renderer-improvements/` and clearly identify themselves as the active owner for renderer improvements spanning specs 030-032.
- **SC-002**: Every major renderer slice from specs 030-032 is either mapped into a convergence phase or explicitly marked out of scope with rationale.
- **SC-003**: No convergence phase contains more than 10 implementation steps.
- **SC-004**: Validation guidance exists for each convergence phase and uses staged-client proof surfaces rather than abstract claims.
- **SC-005**: Specs 030, 031, and 032 each contain a visible note directing readers to the convergence owner plan.
- **SC-006**: `spec.md` + `plan.md` enumerate runtime controls for terrain/lighting/fog/liquid/M2 and map each to a phase-level proof checkpoint.
- **SC-007**: Phase validation guidance includes telemetry artifacts (structured log/table outputs) in addition to screenshots.
- **SC-008**: For staged `3.3.5.12340` validation scenarios that include both river/ocean water and magma/slime, rendered liquid families match the source classification evidence, with zero river/ocean samples rendered as magma unless the evidence explicitly marks them as magma.
- **SC-009**: After the first bounded terrain/world performance slice, the primary staged `3.3.5.12340` mixed outdoor route shows at least a 20% reduction in median frame time relative to its pre-slice baseline on the same validation machine.
- **SC-010**: On the same route, no post-settling traversal segment longer than 10 seconds spends more than 2 consecutive seconds below 30 FPS unless the evidence attributes the drop to first-load streaming still in progress.
- **SC-011**: Performance evidence exists for sparse, mixed, and dense staged `3.3.5.12340` routes, and each artifact explicitly labels whether the dominant cost is steady-state rendering or deferred asset loading.

## Assumptions

- Specs 030, 031, and 032 remain useful as source material and should not be deleted or rewritten as if they were the new owner plan.
- The current renderer recovery work in spec 035 improves immediate M2 confidence but remains a separate feature track.
- `wow-viewer` remains the only implementation target; `gillijimproject_refactor` stays read-only reference-only.
- Initial convergence is a planning/documentation feature, not a promise to implement every renderer slice in one session.
- Ghidra evidence from staged `3.3.5.12340` is treated as authoritative for initial runtime-control dependency modeling in this convergence feature.
- Cross-build liquid-family semantics can differ enough that a single global liquid-type hard-code is not assumed to be safe.
- Terrain/world FPS improvement for this lane is evaluated on a fixed validation machine and compared against local before/after baselines rather than a cross-machine absolute benchmark.
- The first live-view performance slice can focus on terrain/world traversal behavior without absorbing unrelated UI, editor, or full M2 parity work.
