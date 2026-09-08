# Feature Specification: Renderer Marketing Capture Automation

**Feature Branch**: `233-marketing-capture-automation`

**Created**: 2026-09-08

**Status**: Draft

**Input**: User description: "Create automated marketing videos from the viewer's renderer: load a client and a camera path such as FlybyUndead.mdx/.m2, warm and run the path as a renderer benchmark, reveal selected UI features at scripted moments, capture the result directly from the renderer, and make the process automatable later through MCP and a ComfyUI authoring pipeline. The GitHub README ultimately needs real video and stills."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Record a scripted renderer feature tour (Priority: P1)

An operator who has loaded a supported client, world, and camera path selects a named feature-tour recipe. The viewer validates prerequisites, warms the path, then plays it and records renderer frames. The normal UI chrome remains hidden during the scene, except for the feature-specific UI state or callout scheduled by the recipe, so no manual option-bar clicking is necessary.

**Why this priority**: This is the operator's current manual promotional-video workflow and is also the renderer torture-test path.

**Independent Test**: With a known local client and `FlybyUndead.mdx` or `.m2` path loaded, run the recipe and inspect the recorded video plus its receipt. The result demonstrates scene playback and scheduled feature presentation without manual clicks after start.

**Acceptance Scenarios**:

1. **Given** a loaded client, world, and valid camera path, **When** the operator starts a feature-tour recipe, **Then** the viewer warms the path before recording and executes the recipe's ordered scene and UI/callout beats.
2. **Given** normal UI chrome is hidden for a running recipe, **When** a scheduled feature beat occurs, **Then** only the requested feature presentation is visible for that beat and the rest of the chrome remains hidden.
3. **Given** a recipe start lacks a required client, camera path, encoder, or writable output location, **When** validation runs, **Then** recording does not start and the viewer reports the missing prerequisite without producing a misleading success receipt.

---

### User Story 2 - Inspect a capture and renderer benchmark receipt (Priority: P2)

After each completed or failed tour, an operator can inspect a durable, self-describing receipt next to the capture. It identifies the recipe, world and camera-path inputs, renderer/application version, capture settings, ordered beat results, and measured frame-time/performance data, including observable hitches or failure state.

**Why this priority**: A polished-looking video alone cannot establish that a renderer path was healthy or reproducible; the known 15 FPS and 7 FPS dips must remain visible as measured evidence rather than be erased by marketing workflow claims.

**Independent Test**: Execute a deterministic recipe fixture and verify that its receipt serializes every required provenance and measurement field; simulate a failed prerequisite and verify a typed failed receipt.

**Acceptance Scenarios**:

1. **Given** a completed or failed tour attempt, **When** its receipt is read, **Then** it names the attempt outcome, input identities, capture output(s), settings, all scheduled beats, and frame-time/FPS statistics.
2. **Given** a renderer hitch or capture failure, **When** the attempt concludes, **Then** the receipt records it explicitly and never reports a clean benchmark solely because an encoded file exists.

---

### User Story 3 - Hand captures to external authoring automation (Priority: P3)

An external tool can discover and submit a validated, non-destructive authoring handoff containing the capture and receipt. The handoff is designed so a future MCP host and the operator's ComfyUI authoring setup can create titles, edits, thumbnails, and README-ready assets without reading undocumented viewer internals.

**Why this priority**: The requested production pipeline needs an automation boundary, but direct ComfyUI execution must wait for an available, versioned MCP capability and an operator-approved workflow.

**Independent Test**: Build a handoff from a completed fixture attempt and validate its schema, relative project-managed artifact paths, and refusal behaviour when its capture or receipt is absent.

**Acceptance Scenarios**:

1. **Given** a completed capture with a valid receipt, **When** an external client requests its authoring handoff, **Then** it receives a versioned descriptor with project-managed capture and still candidates, provenance, and no machine-local secrets or client-root paths.
2. **Given** ComfyUI/MCP transport is unavailable or rejects a request, **When** the handoff is requested, **Then** the viewer preserves the capture and reports a typed transport-unavailable result; it does not fabricate an authored video or publish anything.

## Edge Cases

- The camera path cannot load, ends early, or cannot be warmed: stop before recording or record a typed aborted outcome with the failing stage and no false completion.
- The renderer experiences low FPS, frame-time spikes, or a device/capture error: preserve measured values and mark the receipt degraded or failed according to the real attempt state.
- A recipe contains an unknown UI feature, conflicting beats, or an out-of-range timestamp: reject it before it changes the viewer state.
- The encoder is unavailable or output storage is not writable: use the existing capture validation path and do not emit a claimed video.
- A handoff points outside a project-managed output root, lacks a matching receipt, or targets an unavailable transport: refuse it safely and leave existing artifacts untouched.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST provide named, versioned feature-tour recipes whose inputs identify a camera path, warm/run behavior, capture settings, and an ordered timeline of scene and feature-presentation beats.
- **FR-002**: The viewer MUST validate the loaded client/world/path, encoder readiness, recipe validity, and project-managed output location before it starts a tour capture.
- **FR-003**: The viewer MUST use its renderer-frame video capture route, rather than desktop/screen capture, for a feature-tour recording.
- **FR-004**: The viewer MUST execute a warm-path stage before the recorded path run when the selected recipe requires warming.
- **FR-005**: The viewer MUST keep normal UI chrome hidden during a tour unless a scheduled beat explicitly presents a named feature control or a tour callout; presentation must not require an operator to click the options bar during the run.
- **FR-006**: The viewer MUST write a durable receipt inside the project-managed capture output for every attempted tour, including typed failure/aborted outcomes. A successful video file alone MUST NOT imply a successful benchmark.
- **FR-007**: A receipt MUST include recipe identity/version, world and camera-path identities, application/renderer version, capture settings, beat outcomes, output identities, elapsed time, and measured frame-time/FPS data sufficient to reveal hitches and low-FPS intervals.
- **FR-008**: The viewer MUST offer a versioned external authoring-handoff model that contains only project-managed artifact references and provenance needed by future MCP/ComfyUI automation.
- **FR-009**: The first release of this feature MUST keep external authoring transport opt-in and failure-explicit. It MUST NOT assume that ComfyUI at a local address is reachable, invoke an undocumented HTTP endpoint, upload artifacts, publish media, or expose client roots/secrets.
- **FR-010**: The project README showcase may link or embed only operator-reviewed, real capture and still artifacts with a matching receipt. Placeholder, synthetic, or unverified promotional assets MUST NOT be represented as viewer output.

### Key Entities

- **Feature-tour recipe**: Versioned, named timeline that describes prerequisites, warm/run behavior, capture settings, and visible feature-presentation beats.
- **Tour beat**: One validated action at a defined point in a recipe, such as starting a path, applying a temporary named UI presentation, displaying a callout, or ending the attempt.
- **Tour attempt receipt**: Durable record of one run, its provenance, measured renderer/capture data, beat outcomes, output identities, and typed terminal outcome.
- **Authoring handoff**: Versioned external descriptor derived from a valid receipt; carries approved artifact references and metadata for a future MCP/ComfyUI editing workflow.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a valid loaded `FlybyUndead` camera-path recipe, the operator can start a warm-and-record feature tour with no manual option-bar interaction after the start command.
- **SC-002**: Every tested valid, rejected, aborted, and failed attempt produces a receipt with all FR-007 fields and a terminal outcome matching the observed result.
- **SC-003**: Deterministic recipe/receipt and authoring-handoff validation tests pass, including unavailable-transport and out-of-root refusal cases.
- **SC-004**: A future external client can create and validate an authoring handoff without requiring a client-root path, desktop path, or undocumented viewer state.
- **SC-005**: README showcase links are added only after a real operator-run capture and at least one real still are reviewed with their receipt; this feature does not claim that gate before it occurs.

## Assumptions

- Existing camera-path playback and renderer-frame capture remain the authority; this feature layers orchestration and evidence on top of them rather than replacing format readers, rendering, or capture encoding.
- A licensed `ffmpeg.exe` and its notices are supplied and runtime-verified separately through the existing capture release-hardening gate.
- Real client playback, visual/UI timing, video quality, performance interpretation, ComfyUI workflow execution, README media review, and any Patreon setup remain operator-owned gates.
- The current ComfyUI/MCP server is an intended future consumer, not an assumed contract: no live transport integration is authorized until its concrete available MCP schema and workflow are selected.
- Patreon setup is explicitly deferred; this feature only prepares reproducible promotional artifacts and provenance.
