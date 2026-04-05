# MdxViewer UI Panel And Prefab Library Plan

Date: 2026-04-05
Target: `gillijimproject_refactor/src/MdxViewer`

## Objective

Stabilize and simplify the active viewer shell so it behaves like one coherent editor application instead of a viewer plus duplicated editor surfaces, then add the first real workflow for terrain alpha-mask brush or prefab harvesting.

This plan treats three problems as one program:

1. The current UI is too monolithic and duplicated.
2. The shell does not adapt well when the window starts or returns in a non-maximized size.
3. Terrain alpha-mask analysis needs a real panel workflow and a machine-assisted prefab or brush harvesting pipeline, not a one-off debug toggle.

## Product Direction

### 1. The app is one editor with workspaces, not a viewer/editor split product

- Stop treating `Viewer` and `Editor` as separate app identities.
- Treat the app as one editor-capable tool with different workspace surfaces.
- `Viewer` workspace means read-only or informational tools.
- `Editor` workspace means tools that manipulate the map, environment, placements, assignments, or terrain data.
- Do not duplicate the same controls in both workspaces unless the duplication is temporary and explicitly marked for removal.

### 2. Replace monolithic sidebars with modular panels

- Every major workflow should have one canonical panel home.
- Panels must be dockable into left, right, top, or bottom dock lanes.
- Multiple panels must be able to stack within each dock lane.
- Panels must be pop-out capable when useful.
- Drawers are acceptable as a compact presentation mode for constrained window sizes, but docking remains the primary organization model.
- UI profiles or shell presets are not the answer to the current confusion; the real problem is duplicated UI ownership.

### 3. Terrain alpha masks are not just layer textures

- Alpha-mask content must be treated as reusable authored terrain patterns.
- A brush is the reusable pattern unit.
- A prefab can be a larger grouped collection of related brush patterns.
- The harvested unit must include both:
  - the 2D alpha-mask sprite or mask pattern
  - the corresponding 3D terrain deformation pattern under that mask
- The library system must support rotation, mirroring, translation, and extrusion-aware matching instead of assuming exact 2D duplicates only.

## Current Problems

### Shell and layout problems

- Too much UI duplication across `ViewerApp_Sidebars.cs`, `ViewerApp_Investigation.cs`, utility windows, and workspace-specific surfaces.
- Existing left/right shell structure still hides ownership boundaries between navigation, inspection, runtime diagnostics, PM4 tools, terrain tools, and editor actions.
- Non-maximized startup or later window resize can leave panels clipped, hidden, or laid out in ways that feel broken.
- The current `Viewer` and `Editor` split adds cognitive load without clarifying actual tool ownership.

### Terrain-analysis problems

- The historical terrain plan still assumes one `Terrain Tools` panel in the old sidebar model.
- The older alpha visualization path regressed into a single `Alpha` button that no longer cooperates correctly with layer-specific views like `L1`, `L2`, and `L3`.
- The UI no longer exposes the alpha data in a way that supports systematic brush or prefab archaeology.
- Manual harvesting is not realistic at the scale of repeated, rotated, mirrored, or partially reused brush patterns.

## Required Shell Rules

### Stable layout rules

- The scene viewport must always recompute against the actual docked free space, not a stale maximized-only shell assumption.
- On non-maximized startup, the shell must clamp or rebalance dock sizes so panels stay visible instead of spawning outside the usable frame.
- Minimum panel sizes must be enforced centrally.
- Panels that do not fit must collapse into drawers, tabs, or overflow stacks instead of rendering partly off-screen.
- The shell should not depend on one exact saved `imgui.ini` geometry to remain usable.

### Panel ownership rules

- One workflow, one panel owner.
- The same setting or control should not exist in both a sidebar section and an investigation panel unless one is a temporary compatibility shim.
- Diagnostics that belong together should merge into one panel instead of spreading across many one-off windows.
- Utility windows should only survive as true transient tools, not as permanent alternate homes for core workflows.

### Workspace rules

- `Viewer` workspace:
  - inspection
  - runtime statistics
  - browsing
  - model, WMO, PM4, terrain, minimap, and lighting read-only investigation
- `Editor` workspace:
  - object manipulation
  - PM4-assisted assignment or placement workflows
  - terrain editing tools
  - prefab or brush harvesting, curation, and export workflows
- A workspace should hide irrelevant editing affordances, but should not duplicate the same underlying panel implementation.

## Proposed Panel Inventory

### Core panels

- `Navigator`
  - file browser
  - map browser
  - scene asset navigation
- `Inspector`
  - selected object summary
  - object details
  - world position and identity
- `Scene`
  - main viewport only
- `Minimap`
  - dockable and pop-out capable
- `Runtime Stats`
  - world timings
  - terrain, WDL, object, liquid, and asset-load counters

### Terrain and alpha panels

- `Terrain Layers`
  - base terrain toggles
  - layer visibility
  - restore working independent layer views for `L1`, `L2`, `L3`
  - fix the broken combined `Alpha` behavior so it no longer replaces layer-specific inspection
- `Alpha Brush Inspector`
  - show per-layer alpha masks
  - chunk-local alpha decomposition
  - mask previews
  - chunk brush occupancy summary
- `Brush Harvest`
  - detection controls
  - matching thresholds
  - harvest queue
  - provenance and export status
- `Brush Library`
  - accepted library entries
  - duplicate candidates
  - grouped prefab collections
  - 2D preview and 3D deformation preview

### Object and PM4 panels

- `World Objects`
- `PM4 Workbench`
- `Placement Tools`
- `Save And Dirty State`

### Environment panels

- `Lighting And Fog`
- `Liquids`
- `Overlays And Debug`

## Alpha And Brush Workflow Requirements

### Layer inspection recovery

- Restore separate alpha-layer inspection so `L1`, `L2`, and `L3` can be viewed independently again.
- Keep a combined alpha view only as an explicit derived visualization, not as the only alpha control.
- Make layer-to-alpha ownership visible at chunk scope.

### Brush or prefab harvesting workflow

- User selects map, tile set, or world region.
- System scans chunk alpha masks and paired terrain deformation.
- System proposes brush candidates.
- User reviews, merges, rejects, or promotes candidates to library entries.
- Library stores unique brushes and optionally grouped prefabs.

### Required captured data for each harvested entry

- source map and tile provenance
- layer index or indices
- 2D alpha-mask pattern
- normalized 3D terrain deformation patch
- transform metadata
  - rotation
  - mirror state
  - scale if applicable
  - offset window
- similarity or dedupe signature
- acceptance state
  - candidate
  - approved brush
  - grouped prefab member

## Detection And Deduplication Pipeline

### Candidate extraction

- Use connected-region and contour analysis on alpha-mask layers to find nontrivial painted patterns.
- Support chunk-border-aware region extraction so patterns that span chunk edges are not split into junk fragments.
- Extract paired terrain deformation windows from the same region.

### Computer vision pipeline

- Prefer `OpenCvSharp` or another practical OpenCV 2-compatible .NET binding for the first implementation pass.
- Use image normalization for:
  - rotation candidates
  - mirrored variants
  - scale-normalized comparisons where safe
- Use contour, template, keypoint, or shape-descriptor approaches only where they are measurable against real data.
- Do not over-claim exact semantic grouping until validated on real map data.

### Dedupe requirements

- Dedupe must not rely on the 2D alpha pattern alone.
- Dedupe must combine:
  - 2D alpha similarity
  - 3D terrain deformation similarity
  - transform-normalized comparison
- The system must preserve near-duplicates for manual review when confidence is low.

### Output forms

- 2D brush thumbnail for UI display
- raw alpha-mask crop or canonicalized representation
- 3D height or deformation patch export
- metadata manifest for provenance and signatures
- grouped library index for prefabs and brushes

## Data And Service Boundaries

### New likely service seams

- `UiShellLayoutService`
  - central dock or drawer sizing
  - panel registration
  - layout clamping on startup and resize
- `WorkspaceRouter`
  - workspace-specific panel visibility
  - same panels, different task exposure
- `TerrainAlphaLayerInspectionService`
  - layer-resolved alpha visualization and export helpers
- `TerrainBrushDetectionService`
  - candidate extraction from alpha and height data
- `TerrainBrushDedupeService`
  - uniqueness checks across 2D and 3D signatures
- `TerrainBrushLibraryService`
  - approved brush or prefab catalog ownership

### Data sources for brush harvesting

- alpha-mask payloads from active terrain decode path
- per-chunk terrain heights or a normalized local deformation patch
- optional MCCV or texture context later, but not required for slice 1

## Delivery Phases

### Phase 1 - Shell stabilization and panel extraction foundation

- centralize panel registration and ownership
- create dock-lane model for left, right, top, bottom
- add window-size-aware layout clamping for non-maximized startup
- keep old shell behavior only as a temporary compatibility layer

### Phase 2 - Workspace cleanup

- replace `Viewer` vs `Editor` shell duplication with workspace gating over shared panels
- move all editing actions behind workspace-aware tool exposure
- remove obsolete or duplicate shell toggles

### Phase 3 - Terrain layer and alpha recovery

- restore independent `L1`, `L2`, `L3` alpha inspection
- fix broken combined `Alpha` control
- move alpha diagnostics into dedicated terrain or alpha panels

### Phase 4 - Brush candidate extraction

- implement region extraction and paired height-patch capture
- add first harvest panel with preview queue
- export candidate artifacts for offline review

### Phase 5 - Deduped library pipeline

- add transform-aware dedupe
- add accepted library and grouped prefab model
- add 2D and 3D preview surfaces

### Phase 6 - Editor integration

- connect accepted brushes or prefabs back into terrain editing workflows
- support 2D preview plus 3D terrain deformation application semantics
- keep write paths cautious until validated on real data

## Validation Requirements

### Shell validation

- start the app non-maximized and confirm panels remain reachable
- resize down and up repeatedly while checking docked panel visibility and scene viewport correctness
- confirm docked, stacked, and popped-out panels preserve usable sizes
- confirm workspace switching changes tool exposure without duplicating panel state

### Terrain and brush validation

- use real data under `test_data/development/World/Maps/development`
- verify `L1`, `L2`, `L3`, and combined alpha views match decoded terrain layers
- confirm brush candidate extraction across repeated, mirrored, and rotated patterns on real ADTs
- confirm 2D and 3D dedupe does not collapse obviously distinct terrain deformations into one brush

## Proof Boundaries

- Do not claim the shell overhaul is complete based only on layout code compiling.
- Do not claim brush or prefab harvesting works based only on synthetic mask samples.
- Do not claim dedupe quality without real-data validation on repeated terrain motifs from actual maps.

## Immediate Next Slice

1. Extract a real panel registry and dock-lane shell contract from the current `ViewerApp` and sidebar code.
2. Replace the current `Viewer` vs `Editor` shell split with workspace gating over shared panels.
3. Restore separate terrain alpha-layer inspection before building the brush detector.
4. Add a first terrain brush candidate service that exports region crops plus paired height patches from real ADT data.