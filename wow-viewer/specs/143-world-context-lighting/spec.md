# Feature Specification: World Context And Lighting Parity

**Feature Branch**: `143-world-context-lighting`

**Created**: 2026-08-11

**Status**: Draft

**Input**: User description: "Resolve ADT and WMO area IDs to DBC AreaNames, model the camera as a controllable player head, and restore evidence-gated WMO and MDX lighting and shader parity across supported WoW builds."

**Coordination**: This feature owns the cross-cutting world-context contract. Spec 138 owns the 4.x renderer evidence/profile matrix, Spec 106 owns native day/night lighting, and Spec 142 owns scene residency and graph traversal. This feature must consume those contracts rather than replace them.

## User Scenarios & Testing

### User Story 1 - Resolve The Current World Area (Priority: P1)

As a viewer user, I want the status bar to identify the area beneath the camera using the client data, so that `Area: Unknown` is an exceptional diagnostic state rather than the normal result.

**Why this priority**: The current AreaName display is not trustworthy, which makes navigation, screenshots, and renderer debugging harder across every map and zone.

**Independent Test**: Load representative maps from at least one early client, one 1.x/3.x client, and one 4.x client; place the camera over known ADT chunks; verify the displayed raw area ID, map context, and AreaName against the active client table.

**Acceptance Scenarios**:

1. **Given** a camera over an ADT chunk with a valid area ID, **when** the viewer updates world context, **then** the status bar shows the matching localized AreaName and raw area ID.
2. **Given** a map whose DBC/DBD layout uses a different column name or row shape, **when** the build profile is loaded, **then** the same logical area lookup succeeds without hardcoded numeric column assumptions.
3. **Given** an area ID that is absent, zero, malformed, or not valid for the active map, **when** the lookup runs, **then** the viewer shows an explicit unresolved reason and does not invent a zone name.

### User Story 2 - Identify WMO Interior Context (Priority: P1)

As a viewer user, I want the viewer to identify the WMO area while the camera is inside a building or other WMO, so that interior navigation and screenshots carry the same context the game would use.

**Why this priority**: WMO interiors currently lose their area identity, and the viewer needs a reliable camera-owned context before interior lighting can be selected correctly.

**Independent Test**: Enter representative WMO interiors with known WMO area IDs, move across group and portal boundaries, and verify that the active WMO area name changes only when the camera enters or leaves the relevant volume.

**Acceptance Scenarios**:

1. **Given** a camera inside a WMO group with a valid WMO area ID, **when** world context is evaluated, **then** the status bar and diagnostics show the WMO area name, raw ID, WMO identity, and group/source that supplied it.
2. **Given** a camera outside the WMO or in a WMO with no usable area ID, **when** context is evaluated, **then** the viewer falls back to the ADT area context and records why WMO context was unavailable.
3. **Given** overlapping or uncertain WMO bounds, **when** more than one candidate contains the camera, **then** the selection rule is deterministic and the diagnostics expose the candidate count and confidence rather than randomly changing names.

### User Story 3 - Navigate As A Player Head (Priority: P1)

As a museum-simulation user, I want camera movement and context evaluation to behave like the head of a controllable player, so that looking through a world feels spatially grounded rather than like moving an unconstrained editor camera.

**Why this priority**: Area and WMO context are camera-dependent, and a head-oriented camera gives the renderer a stable basis for interior lighting, fog, portal visibility, and future player-facing inspection.

**Independent Test**: Use the camera controls to move and look through open terrain, WMO entrances, interiors, stairs, and portals; verify that eye position, yaw/pitch, context selection, and visible content update from the same camera state.

**Acceptance Scenarios**:

1. **Given** a loaded world, **when** the user moves or looks around, **then** world context, WMO containment, fog, and visibility use the camera eye position and orientation from that frame.
2. **Given** the user changes between first-person inspection and an elevated museum view, **when** the camera mode changes, **then** the player-head state remains explicit and reversible; no hidden positional offset is applied without being shown in diagnostics.
3. **Given** a camera state is saved or restored, **when** the viewer resumes the scene, **then** position, orientation, mode, and active world-context source are restored consistently.

### User Story 4 - Restore WMO And MDX/M2 Lighting (Priority: P1)

As a viewer user, I want WMO interiors and placed MDX/M2 assets to use the lighting information intended by their client era, so that buildings, props, and models retain depth and material character instead of appearing flat-lit.

**Why this priority**: Flat lighting makes WMO interiors visually incorrect and hides the quality of the source assets. Lighting must be fixed as a scene contract, not as isolated brighter colors.

**Independent Test**: Render representative outdoor, WMO-interior, WMO-exterior, and model-heavy scenes with lighting diagnostics enabled; compare directional, ambient, baked/vertex, fog, and point-light contributions against the available client evidence and a documented fallback result.

**Acceptance Scenarios**:

1. **Given** a WMO with valid vertex, baked, or local-light information, **when** it is rendered inside and outside the WMO, **then** the lighting contribution is visible, stable, and attributable to a named source.
2. **Given** a placed MDX/M2 with valid scene-lighting inputs, **when** it is rendered in terrain and WMO contexts, **then** it receives the appropriate directional, ambient, fog, and local-light inputs without losing animation or batching correctness.
3. **Given** a material or shader path that is unsupported or unverified for the active build, **when** the asset is rendered, **then** the viewer uses an explicit fallback and reports missing, unsupported, malformed, or visually unverified status; it does not silently claim parity.
4. **Given** a build with observed BLS/effect evidence, **when** a compatible shader path is enabled, **then** the viewer uses the evidence-backed behavior or an explicitly documented equivalent rather than a hardcoded one-size-fits-all shader.

### User Story 5 - Preserve Cross-Era And Performance Boundaries (Priority: P2)

As a viewer developer, I want context and lighting improvements to remain profile-scoped and measurable, so that fixing 4.x interiors does not regress early clients or destroy the stable frame-time path.

**Why this priority**: The viewer spans incompatible client eras and already has severe dense-scene performance risks. Visual improvement without profile and performance evidence is not releasable.

**Independent Test**: Run focused checks and witnessed real-client scenes for early, 1.x/3.x, and 4.x profiles, recording context correctness, lighting source coverage, frame time, and visible-instance counts before and after each bounded slice.

**Acceptance Scenarios**:

1. **Given** a client profile without a newer area, lighting, or shader signal, **when** it is rendered, **then** the established profile fallback remains active and is labeled as such.
2. **Given** a bounded context or lighting change, **when** the same camera state is compared before and after, **then** the report identifies any CPU, GPU, draw-submission, or streaming regression instead of attributing it to visual quality.

### Edge Cases

- ADT area ID is zero, uses a legacy alias, or resolves to a row belonging to another map.
- AreaTable has duplicate aliases, missing localized text, or a build-specific primary key column.
- Camera is inside a WMO but no WMO group, portal, or area record is valid.
- Camera overlaps WMO bounds at an entrance, roof, skybox, or malformed group volume.
- WMO vertex colors are black/zero, baked lightmaps are absent, or local light records are malformed.
- MDX/M2 is animated, transparent, particle-bearing, or otherwise incompatible with a shared lighting/batching path.
- BLS/effect evidence exists for one build but not for another, or the graphics backend cannot reproduce a source effect.
- A lighting change improves appearance while increasing frame time or making camera movement unstable.

## Requirements

### Functional Requirements

- **FR-001**: The viewer MUST read the area identifier associated with the current camera location from the active world data and expose the raw identifier and source record in diagnostics.
- **FR-002**: The viewer MUST resolve that identifier against the active client’s AreaTable/DBD contract using build/profile-aware logical columns and map context; it MUST NOT hardcode area names, numeric column positions, or one build’s aliases for all builds.
- **FR-003**: The viewer MUST publish a native-UI-equivalent `SubzoneText` display result for the current context, using the resolved leaf/subzone name when present and the parent zone when no leaf exists; the status bar MUST display that result and provide an explicit unresolved state containing the reason when the identifier or row cannot be trusted.
- **FR-004**: The viewer MUST decode WMOAreaID for the WMO/group that contains the camera and MUST expose the WMO identity, group, raw identifier, resolved name, and confidence/source of the result.
- **FR-005**: The viewer MUST use deterministic precedence: valid camera-contained WMO context first, then valid ADT context, then explicit unresolved context; leaving a WMO MUST restore the parent terrain context.
- **FR-006**: The viewer MUST maintain one authoritative camera-rig state containing eye position, orientation, movement mode, and any player-head offset; context, fog, portal visibility, and object lighting MUST consume that state for the same frame.
- **FR-007**: The camera rig MUST support reversible player-head inspection controls without requiring a full gameplay, character, collision, or NPC simulation system.
- **FR-008**: The renderer MUST define attributable lighting inputs for terrain, WMO shells, WMO doodads, MDX/M2 placements, and effects, including ambient, directional, baked/vertex, fog, and local/point contributions where the active profile proves them.
- **FR-009**: WMO interiors MUST NOT use an unconditional flat-light fallback when valid vertex, baked, or local-light data exists; any fallback MUST name the missing or unsupported source.
- **FR-010**: MDX/M2 lighting MUST preserve animation, material, transparency, particle, and batching semantics while consuming the scene lighting contract.
- **FR-011**: Shader/effect selection MUST be build/profile-scoped and MUST use observed BLS/effect behavior or an explicitly documented equivalent; unsupported paths MUST remain visible in diagnostics.
- **FR-012**: Each implementation slice MUST include focused automated checks, representative real-file validation, and a before/after render or frame-time record appropriate to its proof level.
- **FR-013**: Existing early, 1.x/3.x, and 4.x profile paths MUST remain available; a newer area or lighting capability MUST NOT become a universal assumption.
- **FR-014**: Heavy client harvesting, GPU work, and long-running real-scene captures MUST remain user-run operations with exact PowerShell commands handed off after bounded code and validation work is prepared.
- **FR-015**: The viewer MUST distinguish observed, inferred, fallback, unsupported, malformed, and visually unverified context/lighting/shader states in diagnostics and release notes.
- **FR-016**: For the observed negative-count pre-alpha version-2 partial layout, the shared reader MUST decode the embedded 64-byte Global Light header, legacy prefix, two 9-track data sets, and their float bands as a profile-scoped shape; it MUST NOT reinterpret the embedded header's `-1` fields as track lengths or relax the normal 0..32 track-length contract for later layouts.

### Key Entities

- **World context**: The camera-owned current identity containing ADT area, optional WMO area, raw IDs, resolved names, source paths, precedence, and confidence.
- **Area lookup contract**: The build/profile-scoped mapping from raw world identifiers to localized AreaName records and map/parent relationships, plus the native-UI-equivalent `SubzoneText` and parent `ZoneText` display roles.
- **Player-head camera rig**: The controllable eye state containing position, orientation, mode, and explicit offsets consumed by world queries and rendering.
- **Scene lighting contract**: The per-frame set of ambient, directional, baked/vertex, fog, local/point, and effect inputs with source/proof state.
- **Shader/effect profile**: A build-scoped rendering behavior with source evidence, supported materials, fallback behavior, and visual verification status.
- **Context/lighting evidence record**: A named client/build/scene/camera record containing raw inputs, resolved outputs, proof level, and performance measurements.

## Success Criteria

### Measurable Outcomes

- **SC-001**: On a validation set containing at least 12 known ADT locations across three client eras, 100% of valid area IDs resolve to the expected AreaName and 100% of invalid/missing cases show an explicit reason rather than a fabricated name.
- **SC-002**: On at least six known WMO interior/exterior transitions, the viewer selects the expected WMO or ADT context in every witnessed transition and never changes context nondeterministically while the camera is stationary.
- **SC-003**: The player-head camera state is the single recorded source for position/orientation/context/visibility in 100% of the camera-rig validation traces, with no unexplained hidden offset.
- **SC-004**: At least three representative WMO interiors and three model-heavy scenes show attributable non-flat lighting contributions where source data exists, while unsupported cases are explicitly labeled.
- **SC-005**: Lighting and shader changes produce no more than a 10% regression in p95 frame time on the named baseline scene unless an approved evidence record explains the tradeoff; older-client comparison scenes show no new missing-content failure.
- **SC-006**: Focused profile/context/lighting checks pass for one early client, one 1.x/3.x client, and one 4.x client before the feature advances beyond its corresponding phase gate.
- **SC-007**: Every accepted context or lighting result records client build, map/WMO identity, camera state, source inputs, fallback/proof state, and before/after performance evidence.

## Assumptions

- The existing AreaTableService, DBC/DBD readers, terrain adapters, WMO read models, LightService, and renderer statistics are reused or extended only where a proven contract gap exists.
- ADT and WMO area identifiers are client data, not values to be hardcoded from known maps or zones.
- The initial camera-rig slice is an inspection/museum camera with player-head semantics; full player body animation, collision, combat, NPC behavior, and gameplay are out of scope.
- Spec 138 remains the evidence owner for Cataclysm/4.x renderer evolution and BLS/effect research; this spec coordinates its use rather than declaring unverified shader parity.
- Spec 142 remains the owner of scene residency, graph traversal, and dense-scene performance admission.
- Real-client validation uses configured approved client roots such as `H:\CLIENTS`; proprietary client files, harvested corpora, and generated outputs are not committed.
- The user runs heavy captures, broad client sweeps, GPU work, and long performance tests.

## Out Of Scope For The Initial Feature

- Full gameplay/player simulation, collision, NPCs, quests, or combat.
- Replacing all existing WMO/M2 shaders in one change.
- Treating illustrative BLS/decompiled behavior as production truth without build-scoped evidence.
- A universal shader rewrite, renderer rewrite, or removal of current fallback paths.
- Training runs, broad data harvesting, or shipping proprietary assets.
