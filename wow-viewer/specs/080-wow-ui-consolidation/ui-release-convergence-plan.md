# Viewer UI Release Convergence Plan

**Owner**: Spec 080 `080-wow-ui-consolidation`

**Status**: Proposed release plan

**Purpose**: Finish the active `WoWViewer` UI as one coherent, releasable
surface. This plan replaces competing layout decisions with an evidence-first
audit, restores lost working controls into one declared home, and prevents a
visible control from being shipped unless its route and runtime behavior are
proven.

## Ownership And Disposition

| Prior plan | UI release disposition |
|---|---|
| 049 viewer UI consolidation | Superseded. Retain every still-useful surface in the inventory; do not copy its unfinished layout plan. |
| 053 M2 animation pose farm | Include only viewer Model/Animations controls and proof. AnimFarm/export work is out of scope. |
| 056 GPU/LOD modernization | Include only viewer-facing LOD, runtime stats, and diagnostics facts. Renderer/library modernization remains separate. |
| 057 archive version selector | Include the user-facing client/build/source selector. Archive backend work remains separate. |
| 060 UI cleanup | Preserve completed de-duplication and capture-chrome behavior as constraints. |
| 069 UI overhaul | Historical implementation evidence only; its tab system is not the final information architecture. |
| 070 map workbench window | Retired layout proposal. Do not implement it beside the current workbench. |
| 071 sidebar split | Current behavior baseline, not release signoff. |
| 073b converter integration | Absorb as the Converters frame phase. |
| 090 viewer memory profiler | Reuse its Runtime Stats/cache counters as the CPU/process-memory evidence surface; retain its staged dense-map capture gates. |
| 093 render performance audit | Reuse its WMO/MDX/terrain/liquid counters as renderer evidence; do not select batching or liquid rewrites before its capture matrix. |
| 080 UI consolidation | Canonical owner and completion tracker. |

## Non-Negotiable Release Rules

1. Every visible menu item, button, checkbox, slider, hotkey, and launch route
   has exactly one runtime owner and one verification row.
2. A control may be restored, migrated, deliberately retired, or disabled with
   an explanatory tooltip. It may not be silently absent or appear functional
   without a reachable implementation.
3. Tabbed and legacy/dockspace modes must have explicit routes. A flag set by a
   menu route but not drawn in the active mode is a release-blocking defect.
4. Existing working file/map/client selection remains available until its
   replacement passes manual proof; no broad sidebar removal during the audit.
5. `gillijimproject_refactor` is read-only evidence only. All fixes land in
   `wow-viewer/src/viewer/WoWViewer`.

## Phase 0 — UI Surface Inventory And Release Gate

**Goal**: Produce one truth table before changing layout.

1. Inventory every `Draw*Window`, `Draw*Content`, menu item, toolbar action,
   bottom-bar control, hotkey, shell panel, and workbench route in the active
   viewer source.
2. Compare it against the legacy viewer only to identify previously exposed
   user-facing surfaces; record the active replacement or intentional
   retirement.
3. Record each entry in `docs/architecture/viewer-ui-surface-inventory.md`:
   surface ID, user label, source method, menu/bottom-bar/hotkey entry point,
   tabbed route, legacy route, required runtime state, owner frame, status,
   test scenario, and predecessor spec.
4. Add explicit rows for Settings, source/build selector, map discovery,
   minimap, Model Info/Animations/Actions/LOD, World Source/Placements/Tiles/
   Selection/LOD, PM4, Terrain, Archeology, Utilities, capture, WL/LIT
   investigation, and converter commands.
5. Classify every entry `working`, `misrouted`, `missing`, `placeholder`,
   `duplicate`, `retired`, or `disabled-with-reason`.

**Gate**: No feature migration starts until there are zero unclassified visible
controls. The known initial defect is Settings: tabbed mode sets
`_showSettingsWindow` but omits its draw dispatch.

## Phase 1 — Route Integrity And Dead-Control Repair

**Goal**: Make every existing promised route honest before adding surfaces.

1. Render global Settings in both tabbed and legacy modes; prove File, Tools,
   workspace, and bottom-bar launchers open the same window.
2. Audit every `_show*Window` flag against both draw-mode dispatches. Restore
   the window, route it into its declared frame, or remove its visible launcher.
3. Replace placeholder Model LOD and unavailable World/Tools controls with
   runtime facts or disabled explanatory states.
4. Preserve 060's one-home de-duplication: do not re-add copied controls to
   every sidebar.

**Gate**: A menu-route smoke matrix has no click that sets state without a
visible destination in the active UI mode.

## Performance Workstream — Measure UI, Minimap, And Overlay Cost

**Goal**: Keep the release UI from hiding a frame-time, loading, memory, or
draw-submission regression behind a visually functional control.

1. Extend the surface inventory with `performance_class`: `ui-only`,
`minimap-streaming`, `overlay-submission`, `asset-load-trigger`, or
`renderer-pass`.
2. Record a repeatable baseline on a staged dense map: total CPU frame time,
terrain/WMO/MDX/liquid/overlay/asset-load timings, WMO draw composition,
managed/process/cache memory, and minimap cache/load activity.
3. Capture the same camera state with: UI minimized, minimap closed, minimap
open, each high-pressure overlay off/on, WMO hidden, MDX hidden, and object
debug boxes/labels off/on. Change one switch at a time.
4. Instrument UI work only where the existing frame stats cannot attribute it:
major ImGui surface draw time, minimap texture cache hit/miss/decode/upload
counts, overlay build time, overlay draw time, and allocation count/bytes.
5. Treat minimap load scheduling as a hypothesis. The active viewer raises its
world loading budget while the minimap is visible; measure whether this helps
navigation or causes loading/frame spikes before changing it.
6. Make overlay work event-driven: rebuild CPU geometry only when camera,
selection, world data, or its setting changes; cull to the viewport; batch
compatible lines/boxes/labels; and do not allocate per-frame collections for
unchanged overlays.
7. Select one optimization only from the measured top cost. Candidates are
minimap loading/cache behavior, an overlay rebuild/draw path, WMO submission,
MDX submission, terrain, or asset loading. WMO liquid correctness remains a
separate visual correction after its measured pass is identified.

**Gate**: No rendering rewrite is allowed until the baseline matrix identifies
the top cost. A release candidate must show that opening minimap/overlays does
not exceed its documented frame-time and memory budget for the proof map.

## Phase 2 — Stable Information Architecture

**Goal**: Converge the current sidebar/workbench mess without another layout
rewrite.

1. Keep the left source/navigation sidebar during this release; normalize it
to client/build selection (057), file/map discovery, workspace, and navigator.
2. Classify and complete the Model surface: Info, Animations (053/071),
Actions, and factual LOD.
3. Classify and complete the World surface: Source, Placements, Tiles,
Selection Tools, and LOD.
4. Classify and complete tools into independent PM4, Terrain, Archeology,
Utilities, and Converters surfaces.
5. Restore WL-liquid and LIT investigation as explicitly named active tools if
the inventory finds no equivalent; do not leave legacy-only diagnostic routes.
6. Extract a named persistent frame only after the source content has one owner
and passes manual proof. Do not remove the current sidebar route first.

**Gate**: Every release surface has one discoverable home and a stable window
or sidebar destination; no content method has hidden duplicate dispatch.

## Phase 3 — Converter Surface (073b)

**Goal**: Restore the missing user-facing converter commands without rewriting
the converter tool.

1. Document each existing converter verb and its input/output contract.
2. Add one Converters destination under Tools with reusable command cards.
3. Launch the existing converter executable, capture status/output, and prevent
concurrent runs per card.
4. Cover Alpha/LK, M2/MDX, WMO versions, ADT utilities, and round-trip
validation only where the underlying command is available and proven.

**Gate**: Each shown card launches the existing command or is disabled with the
missing prerequisite stated. No conversion logic is duplicated in the viewer.

## Phase 4 — Release Proof And Simplification

**Goal**: Release only a provably complete UI.

1. Run the source/mode/build selector proof from 057 on staged client data.
2. Run standalone M2/WMO proof: Info, Animations, Actions, LOD, group overlays,
and settings persist.
3. Run world proof: source, placements, tiles, selection, LOD, terrain and
object wireframe, minimap, WL/LIT tools, PM4, terrain analysis, and capture.
4. Run every converter card against a safe bounded input or record an explicit
disabled prerequisite.
5. Repeat the matrix in tabbed and legacy modes until legacy mode is formally
retired by a separate approved slice.
6. Build `WowViewer.slnx`, capture the manual evidence, update Spec 080 tasks,
and write concise release notes listing restored, migrated, disabled, and
retired surfaces.
7. Attach the performance baseline and post-change comparison to the same
release package. A UI route is not signed off merely because it renders.

**Release gate**: Zero dead/misrouted visible controls; zero unclassified
legacy surfaces; build passes; the complete manual matrix has recorded proof.

## Explicitly Out Of Scope

- Renderer/library extraction from 056.
- AnimFarm, BVH, pose clip, and FBX work from 053.
- Archive backend implementation from 057.
- A competing dockspace/workbench redesign from 069 or 070.
- New viewer functionality unrelated to restoring or honestly classifying the
  release UI.
