# Tasks: 3D Spatial UI Shell & Camera-Anchored HUD (Spec 212)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)  
Checklist order = execution order. Each phase ends at a verification gate before the next begins.

## Phase 1 — Camera-Anchored 3D HUD Rig & Tab Visibility Toggle

- [x] 212-T101: Define camera-space transform contracts and projection structures in `WowViewer.Core.Runtime` (`CameraHudTransform`, `CameraSpaceProjection`).
- [x] 212-T102: Implement `CameraHudRig` in `WoWViewer.Rendering` (`src/viewer/WoWViewer/Rendering/CameraHudRig.cs`) with depth-clamped local projection and fog-bypass shader.
- [x] 212-T103: Wire `CameraHudRig.Render(...)` into `ViewerApp.OnRender()` immediately following world scene rendering and before ImGui chrome.
- [x] 212-T104: Synchronize HUD visibility directly with `!_hideUiChrome` (toggled by the `Tab` key) and add runtime enable toggle in `Settings > Interface`.
- [ ] **Gate 1**: Camera moves/rotates in 3D scene while HUD anchor stays perfectly locked to viewport orientation; pressing `Tab` hides both 2D chrome and 3D HUD instantly; 0 build errors. Source/test/build evidence passes; live visual/input verification is operator-owned.

---

## Phase 2 — OpenSCAD HUD Geometry Integration

- [x] 212-T201: Author OpenSCAD HUD primitives and generate `.off` files via OpenSCAD MCP tool:
  - `camera_hud_curved_bezel.scad` / `.off` (curved floating viewport bezel with chevrons)
  - `camera_hud_reticle_tactical.scad` / `.off` (multi-ring segmented reticle with elevation notches)
  - `camera_hud_compass_tape.scad` / `.off` (cylindrical graduated heading tape)
  - `camera_hud_gimbal.scad` / `.off` (attitude & heading spherical gimbal)
- [x] 212-T202: Wire `ProceduralMeshLoader.LoadFromOff` for all four HUD assets in `CameraHudRig` initialization with caching.
- [x] 212-T203: Render tactical crosshair reticle at center of viewport with pitch ladder response.
- [x] 212-T204: Render compass heading tape at top-center of HUD, dynamically rotated by `_camera.Yaw`.
- [x] 212-T205: Render curved visor bezel framing the viewport perimeter.
- [ ] **Gate 2**: All four OpenSCAD HUD assets render crisp in viewport with authentic shader accents; heading tape rotates with camera yaw; reticle stays centered; build green. Source/build evidence passes; real-client visual validation is operator-owned.

---

## Phase 3 — Spatial Ray-Casting & Surface Hit Testing

- [ ] 212-T301: Implement `SpatialUiHitTestService` in `WowViewer.Core.Geometry` (camera ray unprojection, quad/cylinder plane intersection, local normalized $(u, v)$ calculation).
- [ ] 212-T302: Author unit tests in `WowViewer.Core.Tests` (`SpatialUiHitTestTests.cs`) testing ray hit accuracy, edge cases, and boundary clamping.
- [ ] 212-T303: Wire mouse hit testing into `ViewerApp_ClickSelection.cs` so spatial surfaces consume pointer events before world picking.
- [ ] **Gate 3**: Core unit tests 100% green; clicking spatial surface triggers correct $(u, v)$ coordinates and consumes click without selecting underlying world objects.

---

## Phase 4 — Museum HUD Profile & 3D Time-of-Day Clock

- [ ] 212-T401: Implement Museum Profile top-level view mode: floating minimal HUD, hiding sidebars and legacy windows (under 10% screen occlusion).
- [ ] 212-T402: Implement circular 3D Time-of-Day clock widget on the HUD rig manipulating `_todHours` continuously across midnight.
- [ ] 212-T403: Verify all Museum HUD actions invoke canonical backend services without duplication.
- [ ] **Gate 4**: Museum profile renders full-screen world with floating HUD; clock widget smoothly adjusts sun/fog lighting; profile switching preserves camera and scene state; build 0 errors.
