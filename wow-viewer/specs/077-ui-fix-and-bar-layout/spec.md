# Spec 077: UI Fix — Animation Wiring, Toolbar, Layout Consolidation

**Branch**: `077-ui-fix-and-bar-layout`
**Status**: Draft
**Builds on**: 071 (left/right sidebar split), 073 (surface cleanup)
**Replaces**: 071 toolbar/viewport integration (partial)

## Context

071 landed left/right sidebars + Model viewer tabs. Post-merge testing revealed several regressions and design gaps:

1. **Animation tab shows no animations** for standalone M2/MDX models, but those same models animate correctly as WMO doodads. Models render perfectly (decode is fine), but the Animation tab UI is not wired to detect/display sequences. Root cause: `_renderer.Animator` returns null or empty sequences for certain renderer paths.
2. **Top toolbar has cutouts** on left/right to fit between sidebars, clipping option checkboxes. User cannot change settings on the fly.
3. **No bottom bar**: options scattered across UI with no dedicated bottom command bar.
4. **Minimap in World > Tiles** is confusing — terrain editing tools (Selection Map, Focused ADT chunk grid) mixed with tile navigation.
5. **Right sidebar duplicates** Source/World/Object/file select that already exists permanently on the left sidebar.
6. **"Chunk Manipulation"** (Selection Map, Focused ADT grid) needs its own tab, separate from tile navigation.
7. **General UI desync**: tooling exists in the program but the UI is not wired to it.

## User Stories

### US1 — Animation tab shows real sequences (P1)

**Given** user loads any M2 or MDX model that has animation,
**When** they switch to Model > Animations,
**Then** they see the sequence list, Play/Pause/Stop buttons, frame slider, speed controls, and loop checkbox.
**And** pressing Play advances the animation visually in the viewport.

**Why P1**: This is a regression from 071 — the animation UI existed before but broke during the sidebar rebuild.

**Acceptance**:
- `DrawModelAnimationControls` reaches `_renderer.Animator` and `animator.Sequences.Count > 0`
- Sequence names appear in the combo box
- Play button starts animation in viewport
- Frame slider updates as animation plays

### US2 — Top toolbar spans full window width (P1)

**Given** the viewer is open with sidebars visible,
**When** user looks at the top toolbar,
**Then** the toolbar spans the ENTIRE window width (overlapping sidebars visually).
**And** all option checkboxes (wireframe, grid, overlay toggles) are fully visible and clickable.
**And** the toolbar is rendered ON TOP of both sidebars (higher Z-order).

**Why P1**: User literally cannot toggle options. This is a functional regression from the sidebar implementation.

**Acceptance**:
- Toolbar X = 0, width = displayWidth (not viewportWidth)
- Toolbar drawn AFTER sidebars so it overlaps them
- No checkbox clipped at left or right edge

### US3 — Bottom bar above status bar (P1)

**Given** the viewer is open,
**When** user looks at the bottom of the screen,
**Then** they see a horizontal bar above the status bar containing:
- Scene state buttons (wireframe, lighting toggles)
- Quick settings (opacity, grid toggles)
- Status info (FPS, draw calls)

**Why P1**: User explicitly requested top+bottom bar layout instead of scattered controls.

**Acceptance**:
- `DrawBottomBar()` method renders horizontal strip above status bar
- Bottom bar height ~40px
- Bottom bar contains wireframe toggle, grid toggles (chunk/cell/tile), opacity slider
- Status bar below it remains unchanged

### US4 — Minimap moved out of World > Tiles (P2)

**Given** user is in the World tab,
**When** they click the Tiles sub-tab,
**Then** they see only tile navigation, chunk targeting, and terrain restore controls.
**And** the minimap is NOT shown in World > Tiles.
**And** the minimap is shown either as a standalone floating panel or in a dedicated location.

**Why P2**: Confusing UI placement, but minimap is accessible via M key fullscreen toggle.

**Acceptance**:
- `DrawTerrainWorkbenchSelectionContent` no longer called from `DrawWorldTilesSubTab`
- Minimap is accessible from a new location (Tools > Utilities sub-tab, or a dedicated panel)
- M key fullscreen minimap still works

### US5 — Right sidebar dedup: no Source/World/File select (P2)

**Given** user opens the right sidebar (workbench),
**When** they look at any tab,
**Then** they do NOT see "Open Game Folder", "Open File", "World Maps", or file browser content in the right sidebar.
**And** those controls exist ONLY in the left sidebar.

**Why P2**: Redundant UI, wastes right sidebar space.

**Acceptance**:
- Right sidebar World > Source sub-tab shows only world-loaded state info, no file browser or map discovery content
- `DrawWorkspaceBarsPanelContent` and `DrawMapDiscoveryContent` are not called from right sidebar tabs
- Left sidebar is the sole owner of source/file/map operations

### US6 — Chunk Manipulation tab (P3)

**Given** user loads a world,
**When** they look at the workbench tabs,
**Then** there is a "Chunks" or "Terrain Edit" top tab (or sub-tab under World).
**And** it contains: Selection Map grid (64x64 tile overview), Focused ADT chunk grid (8x8 per tile), chunk restore controls, alpha clipboard, terrain analysis.

**Why P3**: These are advanced editing tools, not primary navigation. Splitting them makes World > Tiles simpler.

**Acceptance**:
- Selection Map / Focused ADT chunk grid moved from World > Tiles to dedicated location
- Chunk clipboard + terrain analysis accessible from same location
- World > Tiles retains: chunk targeting (tile select), live restore tuning, heightmap export

### US7 — General UI wiring audit (P3)

**Given** user explores each workbench tab,
**When** they interact with controls,
**Then** every visible control does something useful (no dead buttons, no "not implemented" states).
**And** the UI accurately reflects the state of the underlying tooling.

**Why P3**: Builds trust that the UI is functional, not cosmetic.

**Acceptance**:
- Every sub-tab content method has been audited for missing wiring
- Dead or unimplemented controls are hidden or labeled "Coming soon"
- Key tools (PM4, archeology, terrain layers) all functional

## Functional Requirements

### FR-001: Animation wiring fix
- Debug `M2Renderer.Animator` and `MdxRenderer.Animator` for standalone model rendering
- Ensure `_runtimeAnimator` is created when `runtimeModel.Model.SequenceCount > 0` OR when `runtimeModel` has animation data that wasn't counted as sequences
- Ensure `MdxAnimator` is created for all MDX files with animation tracks (bones, geoset anims, etc.)
- Verify `DrawModelAnimationControls` reaches the animator for both M2 and MDX paths
- Add fallback: if `_runtimeAnimator` is null but the legacy renderer has an animator, use the legacy animator
- Log diagnostic info when animator is null for a loaded model

### FR-002: Full-width toolbar
- `DrawToolbar` uses full window width (x=0, width=displayWidth)
- `DrawToolbar` called AFTER sidebars in `DrawUI` execution order
- Toolbar respects `_leftSidebarWidth` and `_rightSidebarWidth` only for layout of its internal items, not for its own width
- All buttons and checkboxes fully visible at any sidebar width

### FR-003: Bottom bar
- New `DrawBottomBar()` method in `ViewerApp_Sidebars.cs` or `ViewerApp.cs`
- Position: y = displayHeight - statusBarHeight - bottomBarHeight
- Height: ~40px
- Content: wireframe toggle, grid overlay toggles (chunk/cell/tile), opacity slider, FPS counter, draw call count
- Bottom bar drawn BEFORE status bar so it sits above it

### FR-004: Minimap relocation
- Remove `DrawInteractiveMinimapSurface` call from `DrawWorldTilesSubTab` → `DrawTerrainWorkbenchSelectionContent`
- Add minimap as a Tools > Utilities sub-tab or as a standalone optional panel
- Keep M key fullscreen toggle working

### FR-005: Right sidebar dedup
- `WorldBottomTab.Source` sub-tab content: show only active world source path + status, no file browser or map discovery
- Remove `DrawWorkspaceBarsPanelContent` and `DrawFileBrowserContent` calls from any right sidebar sub-tab content
- Only `DrawLeftSidebar` calls source/file/map content

### FR-006: Chunk Manipulation tab location
- Either: split World into 5 sub-tabs (Source/Placements/Tiles/Overlays/ChunkEdit) where ChunkEdit gets the editing controls
- Or: add "Chunks" to the Tools bottom tab set
- Move from World > Tiles: Selection Map grid, Focused ADT chunk grid, chunk clipboard, terrain analysis
- World > Tiles retains: tile scope selector, tile export (alpha/heightmap), weak-signal restore

## Out of Scope

- New floating windows (all content stays in tabs or sidebars)
- Model viewer LOD improvements
- PM4 tool enhancements beyond existing wiring
- Archeology algorithm changes
- New minimap rendering features
- Theme/color changes

## Success Criteria

1. Animation tab shows sequences for every loaded MDX/M2 model that has them
2. Toolbar spans full window width, no clipped checkboxes
3. Bottom bar renders above status bar with grid toggles and FPS
4. Minimap accessible without going through World > Tiles
5. Right sidebar has no file browser or map discovery content
6. Chunk manipulation controls accessible from a dedicated location
7. Build: 0 errors

## Assumptions

- No architectural changes to the renderer or model parsers (animation fix is UI-wiring only)
- Toolbar Z-order fix is sufficient (no need for ImGui window flags changes)
- Bottom bar uses same ImGui draw context as existing toolbar
- Minimap can be accessed from Utilities > Minimap sub-tab (already exists in 071)
- Chunk manipulation controls already exist in `DrawTerrainWorkbenchSelectionContent` — need moving, not rewriting

## Files to Change

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Fix `DrawModelAnimationControls`, add `DrawBottomBar`, relocate minimap, dedup right sidebar |
| `src/viewer/WoWViewer/ViewerApp.cs` | Add `DrawBottomBar` call in `DrawUI`, fix toolbar position/z-order |
| `src/viewer/WoWViewer/Rendering/M2Renderer.cs` | Debug/improve `Animator` + `_runtimeAnimator` creation |
| `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` | Debug/improve `Animator` + `_animator` creation |
| `wow-viewer/memory-bank/activeContext.md` | Update after implementation |
| `wow-viewer/memory-bank/progress.md` | Update after implementation |
