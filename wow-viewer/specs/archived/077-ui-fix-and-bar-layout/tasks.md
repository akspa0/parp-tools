# Tasks: 077 UI Fix — Animation Wiring, Toolbar, Layout Consolidation

## Phase A — Fix animation wiring (US1)

**Goal**: Standalone M2/MDX models show sequences in Animation tab.

| ID | P | Story | Task |
|----|---|---|------|
| A1 | [P] | US1 | **`M2Renderer.cs`**: In the native static renderer constructor (line 98-108), after creating `_runtimeAnimator`, add a check: if `_runtimeAnimator == null` but the model has valid animation sequences in its Sequence/SD chunk data, create the animator with a best-effort sequence list. Also ensure `Animator` property falls back to `_legacyRenderer?.Animator` even when native path is active (line 163). |
| A2 | | US1 | **`ModelRenderer.cs`**: In `MdxRenderer` constructor (line 249-261), ensure `_enableM2Animation` is true for ALL standalone models (not just non-adapter models). Currently `_enableM2Animation = !_isM2AdapterModel || !disableM2Animation` — standalone MDX has `_isM2AdapterModel=false` so this should be true already. Debug/log if not. |
| A3 | | US1 | **`ViewerApp_Sidebars.cs`**: In `DrawModelAnimationControls` (line 1990), add diagnostic logging when animator is null to print model path, renderer type, and reason. Log sequences count when available. |
| A4 | | US1 | **`M2Renderer.cs`**: Add a `private bool CanProvideAnimation` flag that checks both `_runtimeAnimator != null` AND `_runtimeAnimator.HasAnimation` (and similarly for legacy). Use this flag to decide if animation is available. |

## Phase B — Fix top toolbar (US2)

**Goal**: Toolbar spans full width, no clipped controls.

| ID | P | Story | Task |
|----|---|---|------|
| B1 | | US2 | **`ViewerApp.cs`** `DrawToolbar()`: Change toolbar width from `viewportWidth` to `displayWidth`. Set toolbar position X to 0 (not `viewportX`). |
| B2 | | US2 | **`ViewerApp.cs`** `DrawUI()`: Move toolbar draw call to AFTER both sidebar draw calls so toolbar renders on top. |
| B3 | | US2 | Verify all toolbar checkboxes fully visible and clickable at default sidebar widths (360px left, 480px right). |

## Phase C — Add bottom bar (US3)

**Goal**: Bottom bar above status bar with grid/opacity/FPS controls.

| ID | P | Story | Task |
|----|---|---|------|
| C1 | | US3 | **`ViewerApp_Sidebars.cs`**: Add new `DrawBottomBar()` method. Renders a horizontal strip (~40px) at `y = displayHeight - statusBarHeight - 40`. Contains: wireframe toggle, chunk/cell/tile grid checkboxes, opacity slider, FPS counter. |
| C2 | | US3 | **`ViewerApp.cs`** `DrawUI()`: Call `DrawBottomBar()` before `DrawStatusBar()`. |
| C3 | | US3 | Remove grid toggles and opacity slider from the top toolbar (they move to bottom bar). Keep top toolbar for: frame model, auto-frame toggle, WMO doodad set selector, layer toggles (L0/L1/L2/L3). |

## Phase D — Minimap relocation + right sidebar dedup (US4, US5)

**Goal**: Minimap accessible from Utilities tab, right sidebar has no file browser.

| ID | P | Story | Task |
|----|---|---|------|
| D1 | | US4 | **`ViewerApp_Sidebars.cs`** `DrawWorldTilesSubTab()`: Remove `DrawTerrainWorkbenchSelectionContent()` call (this draws the minimap/selection-map/chunk-grid). Keep only tile scope selector + export + restore controls. |
| D2 | | US4 | Ensure minimap is accessible from Tools > Utilities > Minimap sub-tab (if not, add it). M key fullscreen must still work. |
| D3 | | US5 | **`ViewerApp_Sidebars.cs`** `DrawArcheologySubTabContent()` area: Ensure no `DrawWorkspaceBarsPanelContent` or file browser content in right sidebar. These belong to left sidebar only. |
| D4 | | US5 | **`ViewerApp_Sidebars.cs`** `DrawWorldSourceSubTab()`: Replace with lightweight world-loaded state info. Remove `DrawWorkspaceBarsPanelContent`, `DrawFileBrowserContent`, `DrawMapDiscoveryContent` calls. |

## Phase E — Chunk Manipulation tab + UI audit (US6, US7)

**Goal**: Terrain editing tools organized in dedicated location, no dead UI.

| ID | P | Story | Task |
|----|---|---|------|
| E1 | | US6 | **`WorkbenchNavigator.cs`**: Add `WorldBottomTab.ChunkEdit` enum value. Add "Chunks" label to the World tab's bottom tab list. |
| E2 | | US6 | **`ViewerApp_Sidebars.cs`**: Add `DrawWorldChunkEditSubTab()` — contains: Selection Map (64x64 tile overview), Focused ADT chunk grid (8x8 per tile), chunk clipboard, terrain analysis. Move these from `DrawTerrainWorkbenchSelectionContent`. |
| E3 | | US6 | `DrawWorldTilesSubTab()`: Now contains only tile scope selector, tile export (alpha/heightmap), weak-signal restore. |
| E4 | | US7 | **ViewerApp_Sidebars.cs**: Audit all sub-tab content methods for dead/unwired controls. Hide or label "Coming soon" for any non-functional buttons. Ensure all 3 top tabs + all sub-tabs have functional content. |

## Dependencies & Execution Order

- A → B → C → D → E (sequential — each builds on previous)
- B1, B2 can be done together (same file, related changes)
- C1, C2, C3 must be done in order
- D1-D4 can be done in parallel once C is done
- E1-E4 in order

## Parallel Opportunities

- E4 (UI audit) can run in parallel with D1-D4

## Checkpoints

- After Phase A: Load any M2/MDX model → Animation tab shows sequences
- After Phase B: Toolbar spans full width, no clipped controls
- After Phase C: Bottom bar visible with grid/FPS
- After Phase D: Minimap not in World > Tiles, right sidebar has no source/file browser
- After Phase E: Chunk editing tools accessible, no dead UI controls
