# Feature Specification: 3D Spatial UI Shell

**Feature Branch**: `212-spatial-ui-shell` (authored on `v0.5.3`; specs 193–211 follow the same convention)

**Created**: 2026-09-02

**Status**: Phase 1 source-complete; Phase 2 REWRITTEN per operator correction 2026-09-06 — decorative reticle/compass/bezel scope struck as unrequested; re-aimed at ImGui panels composited onto camera-frame surfaces

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

### User Story 6 - Selection outlines follow the object, not a box (Priority: P1)

When the operator selects or hovers something in the world, the highlight traces the object's actual
silhouette. A lamp post is outlined as a lamp post; a scroll on a table is outlined as a scroll. No
wireframe cube, and nothing drawn outside the object's own shape.

**Why this priority**: A box is a proxy for the thing rather than the thing, and at small scales the
proxy is larger than what it describes — a third-of-a-yard scroll inside a wireframe cube tells the
operator nothing about what they just selected, and reads as a bug. This is the same thesis as the
rest of this feature (stop approximating; make the UI follow the real geometry) applied to selection,
which is why it lives here.

**This story has no dependency on US1–US5** and is separately shippable. It does not require panels
to be surfaces, the scene to be full-window, or any authored shell geometry. It is P1 because it is a
standing annoyance in daily use, and it must not be gated behind the shell work.

**Independent Test**: Select objects across the full size range — a building, a lamp post, a table
scroll — and confirm each highlight traces that object's silhouette and nothing beyond it.

**Acceptance Scenarios**:

1. **Given** any selectable object, **When** it is selected, **Then** the highlight traces the
   object's own silhouette and does not extend beyond its geometry.
2. **Given** a very small object, **When** it is selected, **Then** the outline matches its actual
   size — it is not inflated to a minimum extent that exceeds the object.
3. **Given** an object partly occluded by other geometry, **When** it is selected, **Then** the
   outline communicates the selection without the operator losing track of which object it is.
4. **Given** a hovered object and a separately selected object, **When** both are shown, **Then**
   they are visually distinguishable.
5. **Given** an object whose renderable geometry has not loaded, **When** it is selected, **Then**
   the fallback is presented as a position marker and is not mistaken for the object's extent.
6. **Given** many selectable objects on screen, **When** one is outlined, **Then** frame cost stays
   within the feature's declared budget.

---

### User Story 7 - A museum profile that dials everything back (Priority: P1)

The operator switches to a profile that removes the panels, docks, tabs and dense readouts and leaves
a small floating HUD locked to the camera. The world fills the screen. What remains is what is needed
to move through and look at the world, presented as 3D elements rather than as a data explorer.

**Why this priority**: The current shell has every capability and is, in the operator's words, "in
your face". A viewer that can be *inhabited* rather than operated is a different product, and it is
the reason the spatial shell is worth building at all. It is independently shippable — a minimal
camera-locked HUD over a full-window scene delivers the experience without US3's context rig or US4's
authored shells.

**Independent Test**: Switch to the museum profile and traverse a loaded world, confirming the scene
is unobstructed, the HUD stays legible and camera-locked, and every action the profile offers works.

**Acceptance Scenarios**:

1. **Given** the museum profile, **When** it is active, **Then** the docked panels, tab bars and
   dense readouts are hidden and the scene fills the window.
2. **Given** the museum profile, **When** the camera moves or turns, **Then** the HUD stays fixed
   relative to the camera and does not swim, lag or drift.
3. **Given** a HUD element, **When** the operator clicks it, **Then** the underlying action is
   invoked exactly as it is from the full shell — the same action, not a reimplementation of it.
4. **Given** the museum profile, **When** the operator needs a capability it does not surface,
   **Then** there is a discoverable way back to the full shell without losing scene state.
5. **Given** either profile, **When** the operator switches between them, **Then** camera position,
   loaded content and selection survive the switch.
6. **Given** the museum profile, **When** the scene renders, **Then** its HUD costs less than the
   full shell it replaces.

---

### User Story 8 - Tools are 3D controls shaped like what they do (Priority: P2)

Controls read as the thing they control. The time-of-day control is a clock face the operator can
turn to an hour, not a labelled bar from 0 to 1. Other tools follow the same principle as they are
converted.

**Why this priority**: This is what makes the museum profile navigable rather than merely emptier,
and the clock is the concrete case the operator named — reaching 3am to light a scene by torchlight
is a thing to *do*, and a linear 0..1 slider is a poor instrument for it. It is P2 because the
profile must exist before its controls matter, and because each control is independently valuable.

**Independent Test**: Set the time of day to a specific hour using the clock control and confirm the
scene lighting matches what the same value produces through the existing slider.

**Acceptance Scenarios**:

1. **Given** a time-of-day control, **When** the operator drags around its face, **Then** the time
   changes continuously and the position corresponds to the hour in an obvious way.
2. **Given** the clock control, **When** a time is set, **Then** it produces the same result as the
   same time set through the existing linear control.
3. **Given** the clock control, **When** the automatic day cycle is running, **Then** the control
   reflects the current time rather than fighting it, and the operator can take manual control.
4. **Given** a wrap-around drag through midnight, **When** it occurs, **Then** the value wraps
   without jumping or clamping.
5. **Given** any converted 3D control, **When** it is used, **Then** it drives the same underlying
   state as the control it replaces, with no second source of truth.
6. **Given** a converted control, **When** the operator prefers the original, **Then** the original
   remains reachable.

---

### Edge Cases

- **Pointer misses every panel.** Ray hits no panel surface — the scene must receive the input, and
  the correct cursor must be shown.
- **Outlining an object with no loaded geometry.** There is no silhouette to trace; whatever is shown
  must not read as the object's size.
- **Outlining instanced or batched geometry**, where one draw covers many objects but only one is
  selected.
- **Outlining terrain, liquid surfaces and PM4 overlay surfaces**, which are not discrete models and
  may have no meaningful silhouette.
- **Very large objects.** A building's outline at close range is mostly off-screen.
- **A museum-profile action with no HUD affordance.** The operator needs something the profile does
  not surface and must not be stranded.
- **HUD legibility across window sizes and aspect ratios**, including very wide and very small.
- **A circular control dragged through its wrap point**, or released outside its face.
- **A circular control while the value is being driven automatically** by the day cycle.
- **A converted control and its original both visible**, which must not become two sources of truth.
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

**Selection presentation**

- **FR-024**: The system MUST present selection and hover highlights as an outline of the selected
  object's own silhouette.
- **FR-025**: The system MUST NOT draw a highlight outside the selected object's geometry, and MUST
  NOT inflate a highlight to a minimum extent that exceeds the object.
- **FR-026**: The system MUST visually distinguish a hovered object from a selected one.
- **FR-027**: The system MUST present a fallback for an object whose geometry has not loaded in a
  form that cannot be mistaken for the object's extent.
- **FR-028**: The system MUST define and apply behaviour for surfaces that are not discrete models —
  terrain, liquid, and overlay surfaces.
- **FR-029**: Selection outlining MUST stay within a declared frame budget with many selectable
  objects on screen.

**Museum profile and 3D controls**

- **FR-030**: The system MUST provide a profile that hides docked panels, tab bars and dense readouts
  and presents a reduced HUD over a full-window scene.
- **FR-031**: The museum HUD MUST remain fixed relative to the camera without swim, lag or drift.
- **FR-032**: A HUD element MUST invoke the same underlying action as its full-shell equivalent, with
  no duplicated implementation and no second source of truth.
- **FR-033**: The system MUST provide a discoverable way back to the full shell, and camera position,
  loaded content and selection MUST survive a profile switch in both directions.
- **FR-034**: The museum profile MUST cost less to render than the full shell it replaces.
- **FR-035**: The system MUST provide a circular, directly-manipulated time-of-day control that wraps
  continuously through midnight without jumping or clamping.
- **FR-036**: A circular value control MUST reflect an externally driven value while it is being
  driven, and MUST allow the operator to take manual control.
- **FR-037**: Any converted 3D control MUST drive the same underlying state as the control it
  replaces, and that original MUST remain reachable.

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
- **SC-009**: Across a test set spanning the full object size range — building, lamp post, table
  scroll — every selection highlight traces the object's silhouette, with zero highlights extending
  beyond the object's own geometry.
- **SC-010**: For the smallest selectable object in the test set, the highlight's on-screen extent
  matches the object's rendered extent — no minimum-size inflation.
- **SC-011**: Selection outlining with the maximum expected number of selectable objects on screen
  stays within its declared frame budget, measured against the frame-time distribution rather than a
  static-camera average.
- **SC-012**: In the museum profile the scene occupies the full window and the HUD obscures no more
  than 10% of it.
- **SC-013**: Every action the museum profile offers invokes the same underlying operation as its
  full-shell equivalent — zero duplicated implementations.
- **SC-014**: Switching profiles in either direction preserves camera position, loaded content and
  selection exactly.
- **SC-015**: The museum profile's HUD costs less per frame than the full shell it replaces, measured
  against the frame-time distribution.
- **SC-016**: A time set on the clock control produces lighting identical to the same time set through
  the existing linear control, including across the midnight wrap.

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
- **Top chrome and status bar stay 2D in the full shell.** The menu bar and status bar are the
  operator's fixed reference frame and the anchor for screen-space constructs. The museum profile
  (US7) hides them; that is the point of it.
- **The museum profile is a presentation, not a fork.** It surfaces existing actions through fewer,
  simpler affordances. FR-032 forbids reimplementing an action behind a HUD element, because two
  implementations of one operation is how they drift apart.
- **ImGui has no built-in circular control.** A directly-manipulated clock face has to be drawn and
  hit-tested. The spec states the behaviour required; the drawing approach is a planning decision.
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
