# Spec 210: 3D Scene Cursor & In-World Spatial Selection

## Context & Motivation

In original World of Warcraft (from early Alpha 0.5.3 through modern retail), the mouse cursor is not a 2D OS hardware arrow. Blizzard tested the bring-up of the WoW engine using an authentic 3D cursor model rendered directly in the scene (`Interface\Cursor\Cursor.mdx` in Alpha 0.5.3, `Interface\Cursor\Point.m2` / `Cursor.m2` in 1.12+).

In `WoWViewer`, the mouse cursor has remained the desktop OS cursor (`ImGui.SetMouseCursor`), while object selection relies on a 2D screen-ray heuristic that casts rays into bounding boxes with generous brush radii, triggering intrusive 2D ImGui tooltip overlays (`##SceneHoverAssetOverlay`). This creates visual disconnect, imprecise picking, and frustration when navigating dense scenes.

This spec replaces the 2D ray-based picking and OS cursor with:
1. An authentic **3D in-scene rendered cursor** using `Cursor.mdx` / `Cursor.m2` loaded directly from client archives.
2. Direct **3D spatial intersection** based on the cursor's in-world location against terrain, WMOs, and M2/MDX models.
3. An **in-scene 3D disambiguation pop-up** (anchored in 3D world space, rather than a 2D menu) when multiple clustered objects occupy the same click neighborhood.

---

## What Is Known & Asset Inventory

1. **Alpha 0.5.3 Asset**: `Interface\Cursor\Cursor.mdx`
   - Contains 18 geosets and bones:
     - `Cursor` (geoset 0): Default gauntlet pointer.
     - `CursorCastGlow` (geoset 1): Casting glow.
     - `Attack`, `Ranged Attack`, `Interact`, `Speak`, `Shop`, `Taxi`, `Buy` (geosets 2–7, 14, 16).
     - Unable variants (geosets 8–13, 15, 17).
   - Textures:
     - `Interface\cursor\Cursor.blp`
     - `Interface\cursor\CursorsCastGlow2.blp`
     - `Interface\cursor\CursorIcons.blp`
     - `Interface\Buttons\GlowStar.blp`
2. **Vanilla 1.12+ Asset**: `Interface\Cursor\Point.m2` / `Cursor.m2`
   - Canonical M2 format cursor model.
3. **Existing Renderer Stack**:
   - `MdxRenderer` and `M2Renderer` already support loading and rendering these formats with material textures and animation.
   - `WorldScene` possesses the active camera, depth buffers, scene raycaster, and collision/hit detection.

---

## User Scenarios & Acceptance Criteria

### US1: Authentic 3D In-Scene Cursor Rendering
**As an operator navigating or editing maps in WoWViewer,**
**I want the mouse to render as an authentic in-world 3D cursor (`Cursor.mdx` / `Cursor.m2`),**
**So that the pointer lives inside the 3D space with accurate depth perception.**

- **AC-101**: When the mouse is inside the 3D viewport (and not hovering 2D ImGui windows), the OS system cursor is hidden, and `Cursor.mdx` (0.5.3) or `Point.m2` (1.12+) is rendered in the 3D scene.
- **AC-102**: The 3D cursor position corresponds to the world-space ray hit against the nearest visible scene surface (terrain, water, WMO, or M2 doodad).
- **AC-103**: If the cursor points into the sky (no scene geometry hit), the cursor floats at a configurable default focal plane distance along the ray.
- **AC-104**: The cursor maintains a consistent screen-relative scale across all depths via distance-proportional scaling:
  $$S = \text{BaseScale} \times \text{Distance} \times \tan(\text{FOV} / 2)$$
- **AC-105**: When moving the mouse over ImGui chrome, dock headers, or editor panels, the 3D cursor disappears and the appropriate standard OS cursor is restored smoothly.

### US2: 3D Spatial Selection & Dynamic Cursor States
**As an operator inspecting or selecting objects,**
**I want the cursor to change its 3D state and select objects based on true 3D spatial contact,**
**So that picking is immediate, natural, and free of 2D ray-cast inaccuracy.**

- **AC-201**: When the 3D cursor touches an M2 doodad, NPC, or WMO, the model is highlighted directly in 3D (bounding box or mesh outline).
- **AC-202**: The cursor's active geoset / bone adapts to the underlying target kind:
  - Empty space / terrain: `Cursor` (default pointing gauntlet).
  - NPCs / creatures / interactive doodads: `Interact` or `Speak`.
  - WMO portals / doors: `Interact`.
  - Hostile / combat targets: `Attack`.
  - Active editor placement / brush tool: `CursorCastGlow`.
- **AC-203**: Left-clicking selects the touched 3D entity directly into `EditorSession` / inspection without 2D ray ambiguity.

### US3: In-Scene 3D Disambiguation Pop-up for Clustered Objects
**As an operator clicking in a dense cluster of overlapping doodads or WMO geometry,**
**I want an in-scene 3D radial/pin selector rather than a 2D dropdown menu,**
**So that I can disambiguate which item I intended to select without breaking immersion or obscuring the view.**

- **AC-301**: When a click hits a location where $\ge 2$ distinct candidate objects lie within a compact 3D radius (e.g. within 1.5 yd of the hit point), an in-scene 3D disambiguation widget spawns centered at that world position.
- **AC-302**: The widget renders in 3D world space (e.g., subtle 3D selection rings or billboarded name tags orbiting the cluster datum).
- **AC-303**: Hovering a candidate tag in 3D previews that specific object; clicking it confirms selection and closes the 3D widget.
- **AC-304**: Clicking outside or moving the camera past a dismiss threshold dismisses the 3D pop-up cleanly.

### US4: Camera Culling Invariant & Configurable Cursor Style
**As an operator configuring viewer preferences,**
**I want the cursor to never be clipped or culled by the camera and to choose my preferred cursor object in Settings,**
**So that cursor tracking is 100% reliable and customizable.**

- **AC-401 (Camera Culling Invariant)**: The cursor mesh MUST NEVER be culled by the camera frustum, far clip plane, or near clipping plane. Distance along the camera ray is clamped to $Z \ge Z_{\text{near}} + \epsilon$.
- **AC-402 (Occlusion Invariant)**: The cursor rendering pass ensures the pointer remains clearly visible even when contacting or penetrating opaque world surfaces (via depth range bias or overlay shader).
- **AC-403 (Configurable Settings)**: Viewer Settings exposes a Cursor Selection dropdown:
  - `Authentic WoW Gauntlet` (`Cursor.mdx` / `Point.m2`)
  - `Procedural 3D Pointer` (OpenSCAD / procedural arrow)
  - `3D Reticle / Ring` (OpenSCAD target ring)
  - `Classic OS Cursor` (Standard hardware pointer)
- **AC-404 (Seamless ImGui Interop)**: Cursor state seamlessly transitions between the 3D scene and ImGui menus, toolbars, popups, and combo dropdowns without stutter, latency, or disappearing cursors.

### US5: OpenSCAD MCP Interface for Procedural 3D UI & Widget Geometry
**As an agent or developer extending the engine into procedural world building,**
**I want an OpenSCAD MCP interface and mesh loader,**
**So that simple, deterministic 3D procedural objects (menus, rings, pins, buttons, custom cursors) can be generated and loaded into the renderer.**

- **AC-501**: OpenSCAD CLI integration using `C:\Program Files\OpenSCAD\openscad.com`.
- **AC-502**: MCP tool definition providing `openscad_render(scad_code, format)` and procedural widget generators.
- **AC-503**: Lightweight `.off` / `.stl` procedural mesh loader in the viewer to ingest OpenSCAD output directly into GPU buffers.

---

## Architectural Constraints & Rules

1. **Respect AGENTS.md Boundaries**:
   - UI / rendering logic belongs in `src/viewer/WoWViewer/`.
   - Core data contracts and picking structures belong in `src/core/WowViewer.Core/` or `src/core/WowViewer.Core.Runtime/`.
   - Do NOT modify or break existing MPQ/ADT readers.
2. **Dual-Era Support & Extensibility**:
   - Alpha 0.5.3: `Interface\Cursor\Cursor.mdx`.
   - Retail / Standard Vanilla 1.12+: `Interface\Cursor\Point.m2` / `Cursor.m2`.
   - Procedural: OpenSCAD generated `.off` / `.stl` shapes.
   - Graceful fallback: If models are unavailable, falls back cleanly to the system OS cursor without throwing or crashing.
3. **Zero Performance Regression**:
   - The cursor is 1 lightweight draw call with static/cached geometry and lightweight vertex transform.
   - Spatial picking must not re-iterate the entire world model list; it queries the existing spatial acceleration structures (chunk bounds, octree, or visible object set).
