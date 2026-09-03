# Feature Specification: 3D Spatial UI Shell

**Feature Branch**: `212-spatial-ui-shell` (authored on `v0.5.3`; specs 193–211 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "3D spatial UI shell: render existing ImGui panels onto generated 3D geometry so the entire UI becomes objects composited over any loaded scene, with a context-morphing rig driven by the top bar"

## Context & Motivation

Today every panel in the viewer is a flat window that **subtracts** from the scene. `TryGetSceneViewportRect`
carves the sidebars, top chrome, bottom bar and status bar out of the window, and the 3D scene is
rendered into whatever rectangle is left. Opening the Navigator and Inspector shrinks the world.
The panels and the world do not share a space; they compete for one.

Spec 210 established the precedent this feature generalises. It put the *cursor* into the scene as
real geometry, authored as OpenSCAD source, compiled to `.off`, and committed into the repo as
bootstrapped data (`src/viewer/WoWViewer/Assets/OpenScad/`). `OffGeometry` (Core) and
`ProceduralMeshLoader` (Rendering) already read that data. The generation path is the OpenSCAD MCP
server; the *build* depends only on the committed artifacts.

This feature applies the same move to the shell itself: **panels stop being windows cut out of the
scene and become objects composited over it.** The scene reclaims the whole window; panels float in
front of it. Because the panels are objects, the set of them can be arranged, mounted and re-mounted
as a rig whose configuration follows the task the operator selected in the top bar — which is the
organising problem the current fixed sidebars do not solve.

There is a second, longer-range reason recorded here so later phases do not lose it. The engine is a
candidate spine for a separate private orchestration project that consumes the TensorStore/Zarr
datastores this program produces. A UI that is composed of addressable objects in a scene, rather
than of hard-coded screen rectangles, is the form that survives that transition. Spec 213 covers the
protocol half of that interoperation; this spec covers only the shell.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Panels become interactive surfaces in the scene (Priority: P1)

An operator turns on the spatial shell. A panel they already use — the Inspector, say — no longer
occupies a strip of the window edge. It appears as a flat surface positioned in front of the scene,
showing exactly the content it showed before. Every control on it still works: buttons press, sliders
drag, trees expand, text fields accept typing, lists scroll, tooltips appear.

**Why this priority**: Nothing else in this feature is worth anything if the panels stop being
usable. A panel that looks good and cannot be clicked is a regression. This story is the whole risk
of the feature concentrated into one testable slice, and it delivers value alone: even a single
panel floating over a full-window scene is more screen for the world than today.

**Independent Test**: Enable the shell for one panel. Without touching any other panel, exercise every
interactive control on it and confirm each behaves as it does in the 2D shell. Confirm the pointer
lands where the operator sees it land.

**Acceptance Scenarios**:

1. **Given** the spatial shell is enabled for a panel, **When** the operator moves the pointer across
   that panel's surface, **Then** the control under the visible pointer is the control that
   highlights, with no offset between where the pointer appears and what responds.
2. **Given** a slider on a spatial panel, **When** the operator presses on it and drags beyond the
   panel's edge and back, **Then** the slider tracks the drag continuously and does not release or
   jump, matching 2D behaviour.
3. **Given** a text field on a spatial panel, **When** the operator clicks it and types, **Then**
   keyboard focus is captured by that field and the scene does not also consume the keystrokes as
   camera input.
4. **Given** a spatial panel and a loaded scene, **When** the operator clicks a point on the panel
   that visually overlaps world geometry behind it, **Then** the panel consumes the click and no
   scene selection occurs.
5. **Given** a spatial panel, **When** the operator clicks past its edge onto the scene, **Then**
   scene picking behaves exactly as it does with the panel closed.

---

### User Story 2 - The scene reclaims the full window (Priority: P1)

With the spatial shell on, the 3D scene renders across the entire window. Panels occlude parts of it
rather than shrinking it. The operator can see more of the world with the same panels open.

**Why this priority**: This is the payoff the operator actually feels, and it is the difference
between "the UI is drawn in 3D" and "the UI stopped costing me the world". It is P1 alongside US1
because a spatial panel over a still-carved viewport would deliver none of the benefit.

**Independent Test**: Open the same set of panels in the 2D shell and the spatial shell, and compare
the rendered world area and the visible horizontal field of view in each.

**Acceptance Scenarios**:

1. **Given** any set of panels open in the spatial shell, **When** the scene renders, **Then** the
   scene occupies the full window area, excluding only the top chrome and status bar that remain
   deliberately 2D.
2. **Given** the operator opens or closes a spatial panel, **When** the scene renders, **Then** the
   camera's field of view and aspect ratio do not change, and the world does not shift or re-frame.
3. **Given** a spatial panel is open, **When** the operator looks at the world behind it, **Then**
   the panel's opacity is adjustable so world context behind the panel remains readable.

---

### User Story 3 - The rig changes with the task chosen in the top bar (Priority: P2)

The top bar selects what the operator is doing — terrain, objects, PM4 evidence, inspection,
publishing. Choosing a task re-arranges the shell: the panels that task needs are mounted and
positioned, the ones it does not are dismounted. The change is visible as a transition rather than a
jump, so the operator can see what moved where.

**Why this priority**: This is the organising win over the current sidebars, but it depends on US1
and US2 existing first. It is testable and shippable on its own once panels are surfaces.

**Independent Test**: Switch between two tasks in the top bar and confirm the mounted panel set,
positions, and ordering change to the profile defined for each, and change back on return.

**Acceptance Scenarios**:

1. **Given** the operator is in one workspace task, **When** they select a different task in the top
   bar, **Then** the mounted panel set changes to that task's profile within a single visible
   transition.
2. **Given** the operator has manually moved or resized a panel within a task, **When** they leave
   that task and return to it, **Then** their adjustment is preserved rather than reset to the
   profile default.
3. **Given** a panel is mounted in both the outgoing and incoming task profile, **When** the task
   changes, **Then** that panel moves to its new position rather than being destroyed and recreated,
   and its internal state (scroll position, expanded nodes, selection) survives.

---

### User Story 4 - Panels sit on authored shell geometry (Priority: P3)

Panels are no longer bare rectangles. They mount onto authored shells — bezels, angled or curved
backing plates, frames — whose source is OpenSCAD committed to the repo alongside the compiled mesh,
the same way the cursor and cluster pin assets already are.

**Why this priority**: Visual and ergonomic payoff, and the reason the OpenSCAD asset path exists.
It is last because curved and angled surfaces are the hard case for keeping the pointer accurate
(US1), and that must be proven on flat surfaces before geometry is allowed to complicate it.

**Independent Test**: Mount a panel on an angled shell and re-run the entire US1 interaction suite
against it. Pointer accuracy must hold at the shell's maximum designed angle.

**Acceptance Scenarios**:

1. **Given** a panel mounted on an angled or curved shell, **When** the operator interacts with any
   control on it, **Then** every US1 acceptance scenario still holds.
2. **Given** an authored shell asset, **When** the repository is built on a machine with no OpenSCAD
   installed and no MCP server reachable, **Then** the build succeeds and the shell renders from the
   committed compiled mesh.
3. **Given** panel text on an angled shell, **When** the operator reads it at the shell's maximum
   designed angle, **Then** the text remains legible at the default UI font scale.

---

### User Story 5 - The spatial shell is a mode, not a replacement (Priority: P2)

The operator can return any panel, or the whole shell, to classic 2D docking. Automated capture and
validation runs get a deterministic, unchanged UI.

**Why this priority**: The 2D shell is what currently works and what the capture and automation paths
depend on. Removing it would put unrelated workstreams at risk, and a per-panel escape hatch is what
makes the rest of this feature safe to land incrementally.

**Independent Test**: Toggle the shell off and confirm the 2D layout, panel contents, and viewport
rect are byte-for-byte the behaviour that shipped before this feature.

**Acceptance Scenarios**:

1. **Given** the spatial shell is disabled, **When** the viewer runs, **Then** panel layout and
   behaviour are identical to the pre-feature 2D shell.
2. **Given** a capture or validation automation run, **When** it executes, **Then** it produces the
   same UI framing it produced before this feature, regardless of the operator's shell preference.
3. **Given** the operator's chosen shell mode and per-panel placements, **When** the viewer is closed
   and reopened, **Then** the arrangement is restored.

---

### Edge Cases

- **Pointer misses every panel.** Ray hits no panel surface — the scene must receive the input, and
  the correct cursor must be shown.
- **Panel surface is edge-on or behind the camera.** A panel at a grazing angle presents almost no
  pointer target and its text is unreadable; a panel behind the near plane cannot be hit at all.
  Behaviour must be defined rather than emergent.
- **Two panels overlap along the pointer ray.** Exactly one must receive the input, deterministically,
  and it must be the one the operator sees in front.
- **Drag leaves the panel surface.** A slider or splitter drag that travels off the surface, or onto
  another panel, must continue to belong to the control that captured it.
- **Screen-space-by-nature ImGui constructs.** Modals, popups, drag-drop payloads, the main menu bar
  and its dropdowns, and the in-app path picker are anchored to screen coordinates. Their behaviour in
  a spatial shell must be specified, not left to chance.
- **Font and DPI scale.** `UiFontScale` and display DPI change the pixel size of panel content; the
  mapping from surface to content must stay correct across the full supported scale range.
- **Window resize and multi-monitor moves.** Panel placements must survive a resize without drifting
  off-window or inverting.
- **Extreme camera FOV.** The shell must not distort or de-anchor at the FOV extremes the viewer
  allows.
- **Scene geometry in front of a panel.** World geometry must never occlude a panel the operator is
  meant to be reading, and the panel must never be lost behind the scene the way the 3D cursor was
  lost behind the UI.
- **Capture taps.** Both the UI-inclusive and UI-exclusive capture paths must produce the intended
  content with the spatial shell active.

## Requirements *(mandatory)*

### Functional Requirements

**Surfaces and interaction**

- **FR-001**: The system MUST render the content of an existing panel onto a positioned surface in the
  scene without requiring that panel's content code to be rewritten.
- **FR-002**: The system MUST map a pointer position on a panel surface back to the corresponding
  point in that panel's content space, such that the control under the visible pointer is the control
  that receives the input.
- **FR-003**: The system MUST route pointer press, release, motion, drag, scroll and hover to the
  frontmost panel surface under the pointer, and to the scene only when no panel surface is under it.
- **FR-004**: The system MUST preserve input capture for the duration of a drag that begins on a panel
  control, including while the pointer is off that panel's surface.
- **FR-005**: The system MUST route keyboard input to a focused panel control when one exists, and to
  scene navigation otherwise, with no input reaching both.
- **FR-006**: The system MUST resolve overlapping panel surfaces deterministically, delivering input
  to the surface the operator sees in front.

**Compositing and the scene**

- **FR-007**: The system MUST render the scene across the full window when the spatial shell is
  active, unmodified by which panels are open.
- **FR-008**: The system MUST composite panel surfaces above all scene content, so that no world
  geometry occludes a panel.
- **FR-009**: The system MUST keep camera field of view and aspect ratio independent of panel
  visibility.
- **FR-010**: The system MUST provide an adjustable panel opacity so that world context behind a panel
  remains readable.

**The rig and context**

- **FR-011**: The system MUST support named shell profiles that define which panels are mounted, and
  where, for a given workspace task.
- **FR-012**: The system MUST apply the shell profile for the workspace task selected in the top bar
  when that selection changes.
- **FR-013**: The system MUST animate profile changes as a transition, so the operator can follow
  which panel moved where.
- **FR-014**: The system MUST preserve a panel's internal state across a profile change when that
  panel is mounted in both profiles.
- **FR-015**: Operators MUST be able to move and resize a mounted panel, and those adjustments MUST
  persist per task and across sessions.

**Authored geometry**

- **FR-016**: The system MUST render panel surfaces on authored shell geometry supplied as committed
  repository assets.
- **FR-017**: The system MUST build and run with no OpenSCAD installation and no MCP server
  reachable, using only committed compiled assets.
- **FR-018**: The repository MUST retain the editable source for every shell asset alongside its
  compiled form, so any shell can be regenerated.

**Mode, parity and persistence**

- **FR-019**: Operators MUST be able to disable the spatial shell and return to the existing 2D
  layout, per panel and globally.
- **FR-020**: The system MUST leave capture and validation automation output unchanged from its
  pre-feature behaviour regardless of the operator's shell preference.
- **FR-021**: The system MUST persist shell mode, per-task profiles, and per-panel adjustments across
  sessions.
- **FR-022**: The system MUST define and apply consistent behaviour for screen-anchored constructs
  (modals, popups, menu-bar dropdowns, drag-drop payloads, the path picker), which remain 2D.
- **FR-023**: The system MUST keep the pointer-to-content mapping correct across the full supported
  UI font scale and display DPI range.

### Key Entities

- **Panel Surface**: One existing panel presented as an interactive object in the scene. Has a
  position and orientation, a size in content pixels, an opacity, a mounted/dismounted state, and the
  identity of the panel whose content it shows.
- **Shell Rig**: The arrangement of panel surfaces currently presented. Owns mount points, ordering
  for overlap resolution, and the transition between one arrangement and the next.
- **Mount Point**: A named slot on the rig where a panel surface can sit, with a default placement.
- **Shell Profile**: The named mapping from a workspace task to the set of panels mounted and their
  placements — the thing the top bar selects.
- **Shell Geometry Asset**: An authored backing shape for a panel surface, held in the repository in
  both editable source and compiled form.
- **Surface Hit**: The result of testing the pointer against the rig — which surface, and the point in
  that panel's content space.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With a representative panel set open, the visible world area is at least 35% larger in
  the spatial shell than in the 2D shell at the same window size.
- **SC-002**: For every interactive control in a converted panel, the control that responds is the
  control under the visible pointer, in 100% of cases, at every supported UI font scale — measured as
  zero pointer-mapping defects across the interaction suite.
- **SC-003**: An operator can complete each of the primary panel tasks (select an object in the
  Navigator, edit a value in the Inspector, drive the terrain controls) in the spatial shell in no
  more time than the same task takes in the 2D shell.
- **SC-004**: Switching workspace tasks re-arranges the shell within a transition the operator can
  follow, and no panel loses its internal state across the switch.
- **SC-005**: The frame time cost of the spatial shell, with the full default panel set mounted, is
  under 2 ms — measured against the profiler's frame-time distribution, not a static-camera average.
- **SC-006**: Disabling the spatial shell reproduces the pre-feature 2D layout with zero observable
  difference.
- **SC-007**: A clean clone builds and runs the shell with no OpenSCAD installed and no MCP server
  reachable.
- **SC-008**: Capture and validation automation output is unchanged from its pre-feature baseline.

## Assumptions

- **Existing panel content is reused as-is.** The panels' content code is not rewritten; the shell
  changes where and how that content is presented and how input reaches it. Rebuilding widgets
  natively in 3D is explicitly out of scope.
- **Progressive geometry.** Flat surfaces come first and must fully satisfy US1 before curved or
  angled shells (US4) are attempted. This ordering was chosen deliberately because pointer accuracy
  on a curved surface is the feature's main technical risk.
- **The tab shell is the target.** `_useTabUi` is the default shell and the one this feature converts.
  The dockspace path is not a target; see the note in `memory-bank/activeContext.md` about
  `ShouldBypassDockspaceMouseCapture`.
- **Top chrome and status bar stay 2D.** The menu bar and status bar are not converted. They are the
  operator's fixed reference frame and the anchor for screen-space constructs.
- **Spec 210 supplies the pointer.** The scene cursor, its OpenSCAD asset path (`OffGeometry`,
  `ProceduralMeshLoader`), and the post-ImGui overlay pass established there are the foundation this
  feature builds on rather than re-derives.
- **OpenSCAD is an authoring dependency, never a build or runtime one.** The MCP server generates
  assets; the repository carries the results.
- **Testability follows the standing constraint.** No test project references the viewer, so any logic
  that must be tested — surface hit-testing, content-space mapping, profile resolution, transition
  state — belongs in `WowViewer.Core*` rather than beside its caller.
- **Spec 213 is separate.** The MCP tooling surface is a sibling spec. This feature has no dependency
  on it, and neither blocks the other.

## Out of Scope

- Rebuilding panel widgets as native 3D controls.
- Converting the menu bar, status bar, or screen-anchored modal constructs.
- Any change to the dockspace shell path.
- VR/stereo presentation.
- The MCP protocol surface (spec 213).
