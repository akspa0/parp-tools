# Technical Design Plan: 3D Spatial UI Shell & Camera-Anchored HUD (Spec 212)

**Created**: 2026-09-04
**Spec**: [spec.md](spec.md)
**Status**: Phase 1 source-complete; Phase 2 rewritten per operator correction 2026-09-06 (panels on surfaces, not decorative meshes); Phases 3–4 planned

## Architecture Overview

The 3D Spatial UI Shell transforms 2D screen-space panels into **camera-anchored 3D geometric surfaces composited directly over the full-window world scene**. Panels and HUD telemetry instruments no longer shrink or compete with the world viewport; instead, they live in front of the camera at a calibrated depth, rendered with procedural OpenSCAD bezels and hardware-accelerated shaders, and toggle seamlessly with the `Tab` shortcut (`_hideUiChrome`).

```mermaid
flowchart TD
    Camera[Camera View & Projection] --> Rig[Camera-Anchored 3D HUD Rig]
    TabKey["Tab Key Toggle (_hideUiChrome)"] -->|Visibility Gate| Rig
    
    subgraph OpenScadAssets [OpenSCAD Geometric Pipeline]
        CurvedBezel["camera_hud_curved_bezel.off"]
        TacticalReticle["camera_hud_reticle_tactical.off"]
        CompassTape["camera_hud_compass_tape.off"]
        AttitudeGimbal["camera_hud_gimbal.off"]
    end
    
    OpenScadAssets --> ProceduralLoader[ProceduralMeshLoader / OffGeometry]
    ProceduralLoader --> Rig
    
    subgraph SpatialSurfaces [Spatial Interactive Surfaces]
        ImGuiCard[ImGui Sub-surface / Offscreen FBO]
        RayHit[Spatial Ray-Cast Hit Tester (Core)]
    end
    
    Rig --> SceneComposite[Full-Window Viewport Compositor]
    RayHit --> ImGuiCard
```

---

## Key Decisions

### D1 — Camera-Space Attachment Transform
- All 3D HUD elements exist in camera-local space ($Z_{\text{forward}} = -1.0\text{ yd}$ to $-2.0\text{ yd}$ in front of the lens).
- The HUD model transform is computed as:
  $$M_{\text{hud}} = M_{\text{camera\_world}} \times T(x, y, z) \times R(\text{pitch}, \text{yaw}, \text{roll}) \times S(s)$$
- HUD geometry bypasses world fog, ensuring crisp holographic readouts regardless of weather, time-of-day, or terrain distance.

### D2 — Tab Key Visibility Synchronization (`_hideUiChrome`)
- In WoWViewer, pressing `Tab` flips `_hideUiChrome`.
- When `_hideUiChrome == true`, the 3D HUD rig and all floating bezels are instantly cleared/hidden, leaving a 100% pristine, unimpeded world viewport for cinematic inspection and photography.
- Toggling `Tab` back immediately restores the HUD rig at its exact spatial configuration.

### D3 — OpenSCAD MCP Geometric Asset Backbone
- Primitives are designed and generated via the OpenSCAD MCP server into high-precision `.off` geometry files located in `src/viewer/WoWViewer/Assets/OpenScad/`.
- Committed `.off` files are loaded via `ProceduralMeshLoader.LoadFromOff(...)` backed by `OffGeometry` in `WowViewer.Core.Geometry`.
- Zero runtime or build dependency on external OpenSCAD installations.

### D4 — Core Ray-Casting & Surface Hit Testing
- Because no test project references the OpenGL viewer, hit-testing math (unprojecting screen mouse ray $\vec{r}$, intersecting planar quad or curved cylinder surface, computing normalized $(u, v)$ coordinates) lives in `WowViewer.Core` (`SpatialUiHitTestService`).
- Unit tests in `WowViewer.Core.Tests` validate millimeter precision and boundary clamping across FOV and aspect ratio changes.

---

## Phased Implementation Roadmap

### Phase 1 — Camera-Anchored 3D HUD Rig & Tab Visibility Toggle
- Implement `CameraHudRig` in `WoWViewer.Rendering`.
- Wire `OnRender` pass immediately after world terrain/model rendering and before 2D ImGui.
- Bind HUD visibility to `!_hideUiChrome` (`Tab` key).
- Add HUD toggle checkbox under `Settings > Interface` and `Quick > Camera`.

### Phase 2 — ImGui Panels on Camera-Frame Surfaces (OPERATOR CORRECTION 2026-09-06)

The original Phase 2 (decorative reticle/compass/bezel meshes) was **rejected by the operator as
unrequested scope** and struck; those meshes are unused and `CameraHudRig.Enabled` defaults OFF.
Phase 2 is now the core of the actual assignment:

- Render ImGui panel content into an offscreen framebuffer texture sized to the panel.
- Map the texture onto a camera-frame quad using `CameraHudTransform` / `CameraSpaceProjection`.
- Composite after world rendering, before fullscreen chrome; `Tab` hides it with the chrome.
- Phase 3 hit testing consumes pointer events on that quad before world picking.

### Phase 3 — Spatial Ray-Casting & Surface Hit Testing
- Implement `SpatialUiSurface` and `SpatialUiHitTestService` in `WowViewer.Core`.
- Convert screen cursor coordinates into surface-local $(u, v)$ coordinates.
- Route mouse clicks and hovers to spatial surfaces before falling through to world ray picking.

### Phase 4 — Museum HUD Profile & Circular 3D Controls
- Implement Museum profile (camera-locked floating HUD, hiding heavy sidebars and chrome).
- Render circular 3D time-of-day clock widget manipulating `_todHours`.
- Ensure all museum actions invoke canonical backend services without duplication.
