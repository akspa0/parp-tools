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

## Phase 2 — ImGui Panels on Camera-Frame Surfaces (OPERATOR CORRECTION 2026-09-06)

**Operator correction 2026-09-06:** the original Phase 2 — rendering a tactical reticle, compass
tape, and curved visor bezel as decorative meshes — was **rejected as unrequested scope** and is
struck. Those meshes remain committed assets but are unused, and `CameraHudRig.Enabled` now
defaults to `false`. The actual assignment is: **put the ImGui sidebars onto 3D objects that sit on
the camera frame** — render ImGui panel content to an offscreen texture, map that texture onto a
quad mounted in camera space, and interact with it in place (per Spec 212 US1/US2).

- [x] 212-T201: OpenSCAD HUD primitive assets authored and committed (`.off` + `.scad`). *(kept;
      assets are unused until a panel-surface design consumes them)*
- [x] 212-T202: Camera-space transform contracts (`CameraHudTransform`, `CameraSpaceProjection`)
      in Core.Runtime with focused tests. *(absorbed from struck Phase 1/2 geometry work — this is
      the part of the delivered foundation that serves panels-on-surfaces)*
- [ ] 212-T203: Render ImGui panel content to an offscreen framebuffer texture sized to the panel.
- [ ] 212-T204: Map that texture onto a camera-frame quad via the camera-space contracts.
- [ ] 212-T205: Composite the textured quad after world rendering and before fullscreen chrome;
      `Tab` hides it with the rest of the chrome.
- [ ] **Gate 2**: A real ImGui panel renders legibly on a camera-frame surface with pointer
      accuracy matching Phase 3 hit testing; the 2D shell is unchanged when the mode is off; build
      0 errors. Operator visual check owed.

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
