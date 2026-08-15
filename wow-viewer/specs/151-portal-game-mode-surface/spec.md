# Feature Specification: Portal-Aware Rendering, Game Mode, and Simple Viewer Surface

**Feature Branch**: `151-portal-game-mode-surface`
**Created**: 2026-08-14
**Status**: Draft
**Input**: User request: use the 0.5.3 client as a clean-room reference to improve WMO portal culling, add an opt-in character-head game mode with basic physics, provide a simpler interactive viewer surface, and reduce the cost of raw diagnostic work.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Do not render WMO interiors from outside (Priority: P1)

As a viewer user looking at a world from outside a building, I want WMO interior groups to be rejected unless the camera can see them through the building's portal graph, so large buildings do not consume the same render work as their full interiors.

**Why this priority**: This is the most direct renderer performance improvement and can be validated independently with existing WMO group and portal data.

**Independent Test**: Load a WMO with multiple groups and portals, capture visibility for an exterior camera and an interior camera, and verify that exterior-only views do not submit interior groups while an interior view still traverses the reachable portal groups. Repeat with missing or malformed portal data and verify the conservative fallback keeps visible geometry.

### User Story 2 - Use a simple interactive viewer surface (Priority: P1)

As a user who wants to look around a map rather than inspect raw game data, I want a low-information surface with the core load, camera, game-mode, audio, and overlay controls available without the diagnostic workbench taking over the screen.

**Why this priority**: The current information-rich surface performs work and presents detail that is not needed for ordinary viewing; a separate surface makes the intended interaction path explicit without removing the data explorer.

**Independent Test**: Start the viewer in the simple surface, load a supported map or asset, move the camera, switch game mode, and toggle audio/region overlays. Verify that raw payload panels, correlation tables, per-frame diagnostics, and verbose logs are not rendered or refreshed by default. Switch to the advanced surface and verify those tools remain available.

### User Story 3 - Explore from a character's head (Priority: P2)

As a user who wants a more relatable view of the world, I want an explicit game mode that anchors the camera to the selected character model's head and keeps the existing editor/free camera available when game mode is off.

**Why this priority**: It adds a useful way to experience the scene while remaining opt-in and independent from the data-explorer workflow.

**Independent Test**: Select or spawn a supported character model, enable game mode, and verify that the camera follows a finite head anchor with a documented fallback when no head or model is available. Disable game mode and verify the prior editor camera resumes without changing its stored pose.

### User Story 4 - Move with bounded game-like physics (Priority: P2)

As a user in game mode, I want walking, running, gravity, grounding, jumping, and basic collision response so camera movement has an understandable scale and does not freely fly through the terrain.

**Why this priority**: Movement is useful only after the head-camera contract exists, so it follows the camera slice in priority.

**Independent Test**: On a fixture with terrain and simple object collision, simulate walking, running, a jump, a slope, a step, and a fall. Verify finite positions, bounded time-step behavior, grounded transitions, no movement through blocking collision, and a stable fallback when collision data is unavailable.

### User Story 5 - Keep interactive diagnostics cheap while preserving forensic tools (Priority: P2)

As a user trying to maintain a responsive frame rate, I want an interactive diagnostic profile that suppresses raw logging and expensive debug-route refreshes while retaining errors and essential counters, with a deliberate forensic profile for deep inspection.

**Why this priority**: Logging and diagnostic work are cross-cutting costs; making the policy explicit lets the renderer optimization be measured without accidentally deleting investigation capability.

**Independent Test**: Run the same controlled scene under interactive and forensic profiles. Verify that interactive mode does not emit raw payloads or refresh expensive inspection routes every frame, while forensic mode restores those routes. Compare stage timings/counters and confirm errors remain visible in both profiles.

### Edge Cases

- A WMO has no portal records, invalid portal references, degenerate portal polygons, incomplete group bounds, or an unknown flag. The renderer must use a visible conservative fallback rather than hide geometry.
- The camera is exactly on a portal plane, crosses a portal in one update, or is near a root/group bound. Visibility must not oscillate or produce an empty frame because of a precision boundary.
- A portal graph is cyclic, exceeds traversal depth, or contains more reachable groups than the implementation's bounded scratch capacity. Traversal must terminate and retain a safe fallback.
- A selected model has no recognized head attachment, has an invalid transform, is unloaded, or is removed while game mode is active. The camera must remain finite and expose a recoverable fallback.
- A physics update receives a large or negative delta, the character falls outside loaded terrain, encounters a steep slope, jumps into a ceiling, or has no collision source. The integrator must clamp or reject invalid time and avoid NaN/teleport behavior.
- The simple surface is opened before a map, WMO, audio source, or region table is available. Controls must remain usable and explain unavailable actions without forcing raw diagnostics on screen.
- Interactive mode is switched while a diagnostic action is running. The policy change must not strand the viewer or suppress errors needed to recover.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The implementation MUST record the 0.5.3 Ghidra evidence used to shape WMO portal visibility before changing the renderer, including the native portal transform, clip, recursive traversal, and portal-intersection anchors.
- **FR-002**: The renderer MUST maintain a WMO visibility decision that distinguishes exterior camera views from an interior group view and admits interior groups only when the portal/bounds contract proves them reachable.
- **FR-003**: WMO portal visibility MUST use the existing parsed portal vertices, portal records, portal references, group bounds, transforms, and frustum services; it MUST NOT duplicate WMO file readers or port original client code.
- **FR-004**: Portal traversal MUST be bounded by visited-state and depth/capacity limits, and MUST fall back conservatively when the graph or required geometry is incomplete.
- **FR-005**: The optimization MUST apply to opaque, transparent, doodad, and liquid submissions consistently, or explicitly document the pass-specific fallback when a pass cannot safely use the group result.
- **FR-006**: The viewer MUST provide a distinct simple interactive surface/profile separate from the information-rich data-explorer surface. The simple surface MUST expose map/asset loading, camera controls, game-mode toggle, audio toggle, and region overlay controls without requiring raw-data panels.
- **FR-007**: The information-rich surface and its raw inspection routes MUST remain available as an explicit advanced/forensic path; this feature MUST NOT remove existing data ownership or diagnostic capabilities.
- **FR-008**: Game mode MUST be opt-in and disabled by default. When disabled, the current editor/free camera behavior and stored pose MUST remain unchanged.
- **FR-009**: When game mode is enabled and a character model is available, the camera MUST follow a finite head anchor derived from the model's recognized attachment or documented height fallback. Missing or invalid model data MUST produce a visible recoverable fallback.
- **FR-010**: Game mode MUST provide configurable walking and running speeds, gravity, grounding, jump impulse, and bounded delta-time integration. It MUST support forward/back/strafe movement and jumping without requiring a networked player simulation.
- **FR-011**: Basic physics MUST use existing terrain/object collision seams where available, prevent movement through blocking collision in the supported fixture, and degrade to a clearly defined grounded/floating fallback when collision data is unavailable.
- **FR-012**: The interactive diagnostic profile MUST disable or throttle raw payload logging, per-frame verbose inspection, and expensive debug-route refreshes. It MUST retain warnings/errors, essential stage counters, and a user-invoked diagnostics path.
- **FR-013**: The forensic diagnostic profile MUST restore the existing detailed inspection behavior behind an explicit opt-in control and MUST NOT be the default for the simple surface.
- **FR-014**: Performance instrumentation MUST expose enough bounded stage timing and submission/cull counters to compare portal visibility and diagnostic profiles without requiring a raw data dump every frame.
- **FR-015**: Focused unit/integration tests MUST cover portal graph visibility, conservative fallback, game-mode state transitions, finite head anchoring, bounded physics, simple-surface defaults, and diagnostic-profile behavior.

### Key Entities *(include if feature involves data)*

- **WmoPortalVisibilityState**: Camera location/state, candidate interior group, reachable group set, traversal limits, confidence/fallback reason, and pass applicability.
- **GameModeState**: Disabled/editor or active/game mode, selected character/model identity, head anchor, movement intent, grounded state, and camera fallback reason.
- **PhysicsBodyState**: Position, velocity, grounded/contact data, movement parameters, jump state, and bounded integration result.
- **ViewerSurfaceProfile**: Simple interactive or advanced data-explorer surface, visible controls, overlay policy, and refresh policy.
- **DiagnosticProfile**: Interactive or forensic logging/inspection policy plus retained counters and error visibility.

## Success Criteria *(mandatory)*

- **SC-001**: On a controlled multi-group WMO fixture with valid portals, an exterior camera submits no interior groups that are not reachable through the validated portal/bounds decision, while an interior camera still submits its reachable portal-connected groups.
- **SC-002**: On malformed or incomplete portal fixtures, visibility remains conservative, traversal terminates within the configured bound, and no empty-frame regression is introduced.
- **SC-003**: The simple surface reaches a loaded scene and its core camera/audio/game-mode/overlay controls without rendering or refreshing raw inspection panels by default; the advanced surface still exposes those tools.
- **SC-004**: Game mode remains disabled on startup, follows a finite character-head anchor when enabled, and completes a deterministic movement/jump/fall simulation without NaN, runaway velocity, or unbounded delta-time behavior.
- **SC-005**: A controlled performance comparison demonstrates that the interactive diagnostic profile performs less raw logging/debug-route work than the forensic profile while retaining essential stage counters and errors. Any FPS or real-client claim remains pending user-owned runtime proof.
- **SC-006**: The implementation and its tests preserve existing editor-camera, data-reader, audio-toggle, region-overlay, and advanced diagnostic behavior outside the new opt-in paths.

## Assumptions

- The selected/spawned character model is the initial game-mode source; multiplayer, player identity, combat, AI, inventory, and network simulation are not required for this slice.
- Existing terrain, WMO, M2/MDX, collision, audio, and AreaName/region overlay owners remain authoritative and are extended only where their contracts lack the requested behavior.
- Native 0.5.3 behavior is used as clean-room evidence and design guidance only; no original executable code or client implementation is ported.
- Real-client visual, audible, GPU, and FPS validation against the configured client root remains user-owned and will be reported separately from build/unit-test evidence.

## Out of Scope

- Networked multiplayer or authoritative player simulation.
- Full navmesh/pathfinding, combat, AI, questing, inventory, or game rules.
- Replacing the existing data-explorer surface or deleting forensic/debug routes.
- Porting the original 0.5.3 renderer or using undocumented native code as a runtime dependency.
- Automatic audio playback policy changes already owned by the audio/AreaName work unless required to expose the simple surface's audio toggle.
