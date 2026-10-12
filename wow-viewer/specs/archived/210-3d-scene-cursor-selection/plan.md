# Technical Plan — Spec 210: 3D Scene Cursor & In-World Spatial Selection

## Architecture Overview

```mermaid
flowchart TD
    subgraph Input & Mouse Tracking
        Input[Window Mouse Input (X, Y)]
        ImGuiCheck{ImGui Consumes Mouse?}
        HideOS[Hide OS Cursor]
        RestoreOS[Restore OS Cursor]
    end

    subgraph Spatial Picking & Raycasting
        RayGen[Generate Viewport Ray]
        SceneHit[Raycast against Terrain, WMOs, M2 Doodads, Water]
        WorldPos[3D Contact Point in World Space]
        ClusterDetector{Clustered Objects Nearby?}
    end

    subgraph 3D Cursor Pipeline
        CursorLoader[Load Cursor.mdx (Alpha) / Point.m2 (Standard)]
        StateEval[Evaluate Cursor State: Pointer / Interact / Attack / Cast]
        CursorTransform[Compute Distance-Proportional 3D Transform]
        CursorDraw[Render Cursor Model in 3D Scene Overlay]
    end

    subgraph Selection & Disambiguation
        DirectSelect[Direct 3D Object Select]
        InScenePopup[Spawn 3D In-Scene Cluster Disambiguator]
        UserPick[Click Candidate Tag in 3D]
    end

    Input --> ImGuiCheck
    ImGuiCheck -- Yes --> RestoreOS
    ImGuiCheck -- No --> HideOS
    HideOS --> RayGen
    RayGen --> SceneHit
    SceneHit --> WorldPos
    WorldPos --> CursorTransform
    StateEval --> CursorDraw
    CursorTransform --> CursorDraw

    WorldPos --> ClusterDetector
    ClusterDetector -- 1 object --> DirectSelect
    ClusterDetector -- Multiple objects --> InScenePopup
    InScenePopup --> UserPick --> DirectSelect
```

---

## Component Design
### 1. `SceneCursorRenderer` (`src/viewer/WoWViewer/Rendering/SceneCursorRenderer.cs`)
- Responsible for:
  - Multi-asset cursor support:
    - Authentic WoW: `Interface\Cursor\Cursor.mdx` (0.5.3) / `Point.m2` / `Cursor.m2` (1.12+).
    - Procedural OpenSCAD: loads generated `.off` / `.stl` meshes (e.g. 3D Arrow, Reticle, Crystal).
    - Classic OS: bypasses 3D render pass, retaining hardware cursor.
  - Active state switching: controls active geosets/bones (`Default`, `Interact`, `Speak`, `Attack`, `Shop`, `Taxi`, `CastGlow`).
  - Distance-constant perspective scaling:
    $$\text{scale} = \text{BaseScale} \times \text{distance} \times \tan\left(\frac{\text{FOV}}{2}\right)$$
  - **Camera Culling Invariant**:
    - Unconditional rendering: Bypasses frustum culler.
    - Near-plane protection: Clamps minimum distance along camera ray to $Z \ge Z_{\text{near}} + \epsilon$, preventing near-clip vanishing.
    - Depth-range bias: Renders with an overlay depth bias (`glDepthRange(0.0, 0.05)` or always-visible depth test) so the cursor is never occluded or clipped inside world geometry.
  - **Seamless ImGui Interop**:
    - Queries `ImGui.GetIO().WantCaptureMouse` and open popup states.
    - When over ImGui chrome/modals: 3D cursor hides cleanly and OS cursor is restored (`CursorMode.Normal`).
    - When over 3D viewport: OS cursor hides (`CursorMode.Hidden`) and 3D cursor renders smoothly.

### 2. `ProceduralMeshLoader` (`src/viewer/WoWViewer/Rendering/ProceduralMeshLoader.cs`)
- Responsible for:
  - Ingesting OpenSCAD `.off` (Object File Format) and `.stl` files directly into OpenGL VBO/EBO vertex/index buffers.
  - Enables zero-friction procedural 3D UI objects, in-scene radial rings, candidate markers, and custom cursor geometry.

### 3. `SceneSpatialPicker` (`src/viewer/WoWViewer/Terrain/SceneSpatialPicker.cs`)
- Responsible for:
  - Accurate 3D ray-surface intersection:
    - Queries loaded terrain chunks (MCVT / height lattice).
    - Queries visible WMO groups and M2 doodads using bounding boxes and mesh triangles.
    - Queries water planes (MCLQ / WL / MH2O).
  - Returns the nearest 3D hit point $\mathbf{P}_{\text{hit}}$ and the intersected object reference.
  - Cluster Detection:
    - Finds all candidate objects whose bounds or origins lie within radius $R \approx 1.5\text{ yd}$ of $\mathbf{P}_{\text{hit}}$.
    - Returns single target or candidate cluster.

### 4. `SceneClusterSelector3D` (`src/viewer/WoWViewer/Rendering/SceneClusterSelector3D.cs`)
- Responsible for:
  - Rendering 3D in-scene widgets around dense clusters:
    - Orbital 3D ring / pedestal placed at $\mathbf{P}_{\text{hit}}$ in world space (procedurally generated or OpenSCAD mesh).
    - Lightweight billboard tags / pins floating above each candidate in 3D.
  - Hovering a 3D pin highlights the corresponding candidate in the scene.
  - Clicking a 3D pin commits the selection into `EditorSession` / `HoveredAssetInfo`.
  - Automatically dismisses when clicking away or moving the camera.

### 5. Integration into `ViewerApp` & `WorldScene`
- `WorldScene`:
  - Instantiates `SceneCursorRenderer`, `SceneSpatialPicker`, and `SceneClusterSelector3D`.
  - In `WorldScene.Render()`, renders `SceneClusterSelector3D` and `SceneCursorRenderer` as the final scene pass before ImGui.
- `ViewerApp`:
  - Manages cursor setting in Preferences/Settings UI:
    - Dropdown: `Authentic WoW Gauntlet`, `Procedural 3D Pointer`, `3D Reticle`, `Classic OS Arrow`.
    - Scale slider, depth mode toggle.
  - Controls OS cursor visibility without flicker or capture lock.

---

## File Changes

### [NEW] Files
- `src/viewer/WoWViewer/Rendering/SceneCursorRenderer.cs`
  - 3D cursor asset loader, animation/state manager, camera-invariant transform calculator, and OpenGL renderer.
- `src/viewer/WoWViewer/Rendering/ProceduralMeshLoader.cs`
  - Parser and GPU buffer uploader for OpenSCAD `.off` / `.stl` geometry.
- `src/viewer/WoWViewer/Rendering/SceneClusterSelector3D.cs`
  - In-scene 3D disambiguation pop-up renderer and interaction handler.
- `src/viewer/WoWViewer/Terrain/SceneSpatialPicker.cs`
  - Pure 3D spatial raycast and cluster detector.
- `tests/WowViewer.Core.Tests/SceneSpatialPickerTests.cs`
  - Unit tests verifying 3D ray-sphere / ray-box / cluster detection, distance scaling, and OFF parser logic.

### [MODIFY] Files
- `src/viewer/WoWViewer/Terrain/WorldScene.cs`
  - Integrate 3D spatial picker and 3D cursor rendering.
- `src/viewer/WoWViewer/ViewerApp.cs`
  - Coordinate system cursor hiding/restoring based on viewport focus; add cursor settings UI.
- `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs`
  - Route selection through 3D spatial picker instead of legacy 2D screen-ray heuristics.

---

## Phase Roadmap

- **Phase 1: OpenSCAD MCP Server & Procedural Mesh Loader**
  - Register and verify OpenSCAD MCP server via `uv` and `C:\Program Files\OpenSCAD\openscad.com`.
  - Implement `ProceduralMeshLoader` for `.off` / `.stl` in `WoWViewer`.
- **Phase 2: 3D Cursor Asset Loading & Viewport Rendering (US1, US4)**
  - Load `Cursor.mdx` / `Point.m2` / procedural pointer.
  - Implement camera culling invariant: never culled by frustum, near-clip clamped, depth range biased.
  - Seamless ImGui interop (hide in 3D viewport, restore over ImGui panels/modals).
  - Add Cursor Style selection in Viewer Settings.
- **Phase 3: Direct 3D Spatial Selection & Dynamic States (US2)**
  - Replace 2D screen-ray heuristics with direct 3D contact picking.
  - Dynamic state switching (`Default`, `Interact`, `Speak`, `Attack`, `CastGlow`).
  - Eliminate intrusive 2D screen hover overlays.
- **Phase 4: In-Scene 3D Cluster Disambiguator (US3, US5)**
  - Implement in-world 3D radial/pin widget for clustered objects.
  - 3D pin hover preview and selection commit.
- **Phase 5: Verification & Operator Hand-off**
  - Unit tests for spatial math, distance scaling, and OFF parser.
  - Live interactive verification in `WoWViewer` across Alpha 0.5.3 and 1.12+.
