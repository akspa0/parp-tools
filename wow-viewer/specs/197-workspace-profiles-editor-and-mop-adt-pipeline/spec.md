# Spec 197: UI Workspace Profiles, Editor Mode Integration, PM4 Mouse Inspection, Multi-Client Map Staging & MoP 5.0.1 ADT Pipeline

## Executive Summary
Unifies and modernizes the WoWViewer interface, editing workflow, archaeology toolsets, and terrain rendering engine across five interconnected pillars:
1. **Top-Level Mode & Workspace Profile Switcher**: Resolves the missing Editor toggle by introducing a prominent top toolbar Mode Switcher (`[Viewer]`, `[Editor]`, `[Archaeology / Research]`), contextually filtering menus and right-sidebar tool tabs according to the active workflow.
2. **UI & Menu Audit / Legacy MK Dataset Deprecation**: Excises obsolete "MK Dataset" dead code and menu residue across the codebase, organizing offline ML and dataset tools into dedicated, clean sub-menus.
3. **PM4 Direct Viewport Mouse Raycast & Inspection**: Integrates PM4 spatial bounding-box and triangle raycasting into the unified click-selection system (`ViewerApp_ClickSelection.cs`), enabling point-and-click inspection of PM4 object parts, CK24 IDs, surface types, and collision flags directly in the 3D viewport.
4. **Multi-Client Map Staging & Restoration Library**: Establishes a multi-archive workspace ingestion pipeline capable of mounting multiple client sources simultaneously (e.g. 0.5.3 Alpha, 1.12.1 Vanilla, 3.3.5a WotLK, 4.0.1/4.3.4 Cata, 5.0.1 MoP) and copying terrain chunks, placements, and models into a staging/restoration artifact map.
5. **4.3.4 through 5.1 MoP ADT & Rendering Modernization Roadmap**: Expands `StandardTerrainAdapter` and `TerrainRenderer` to support split ADT secondary streams (`_obj0`/`_obj1`, `_tex0`/`_tex1`), height texture blending (`MHID`, `MDID`, `MCXH`), and WMO terrain vertex blending based on binary decompilation of `WoW.exe` 5.0.1.15464 via Ghidra.

---

## User Stories

### US1: Mode & Profile Switching (Viewer vs Editor vs Archaeology)
As an operator/designer, I want clear, dedicated mode buttons in the top toolbar to switch between **Viewer**, **Editor**, and **Archaeology/Research** modes, so that the right sidebar and menus present only relevant tooling for my current task without clutter.

### US2: Legacy "MK Dataset" Deprecation & Menu Cleanup
As a developer, I want all obsolete "MK Dataset" legacy references, menu entries, and defunct loaders removed from the application so that the interface reflects actual active systems.

### US3: Direct PM4 Viewport Mouse Click-Selection
As a collision/world geometry researcher, I want to hover and click directly on PM4 objects and terrain overlay carriers in the 3D viewport to inspect their attributes, part IDs, and CK24 groupings, just like WMO, MDX, and ADT chunks.

### US4: Multi-Client Map Staging & Restoration Library
As a world restorer, I want to mount multiple client MPQ/loose map sources simultaneously and copy terrain tiles, chunk parameters, or doodad placements across client eras into a single active restoration map.

### US5: 4.3.4–5.1 MoP ADT Chunks & Terrain/WMO Blending Support
As a researcher exploring late Cataclysm and MoP beta terrain, I want the renderer to parse and render 4.3.4 and 5.0.1 split ADT chunks (`MHID`, `MDID`, `MCXH`), height blending, and WMO-to-terrain seam blending informed by deep Ghidra reverse-engineering of `WoW.exe` 5.0.1.15464.

---

## Acceptance Criteria

- **AC-001 (Workspace Profile Switcher)**: Top bar displays explicit mode toggles (`Viewer`, `Editor`, `Archaeology`); switching modes reconfigures the right sidebar tabs, bottom page lists, and main menu actions dynamically.
- **AC-002 (Editor Tab Restoration)**: The Editor tab in the right sidebar is fully reachable and renders `DrawEditorContent()` with brush tools, overhead canvas, and placement manipulation.
- **AC-003 (MK Dataset Purge)**: Zero references to "MK Dataset" or `MkDatasetHarvester` remain in UI menus, toolbars, or dialogs.
- **AC-004 (PM4 Mouse Selection)**: Left-clicking a PM4 surface or bounding box in the 3D viewport selects the PM4 object, highlights its geometry, and opens its details in the Inspector.
- **AC-005 (Multi-Client Source Staging)**: Multi-source catalog manager can register 2+ client data sources and stage cross-client chunk/placement copies without archive collision.
- **AC-006 (4.3.4–5.1 MoP ADT Pipeline)**: `StandardTerrainAdapter` correctly parses 4.3.4 and 5.0.1 `_obj0`/`_obj1`, `_tex0`/`_tex1`, `MHID`, `MDID`, and `MCXH` chunks with shader height blending.
- **AC-007 (Ghidra 5.0.1.15464 Binary Knowledge Pack)**: Decompilation and string audit of `WoW.exe` 5.0.1.15464 yields authoritative C++ signatures, chunk layouts, lighting enables, and WMO blending shaders.
