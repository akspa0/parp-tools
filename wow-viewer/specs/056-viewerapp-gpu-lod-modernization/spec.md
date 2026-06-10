# Feature Specification: 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

**Feature Branch**: `056-viewerapp-gpu-lod-modernization`
**Created**: 2026-06-10
**Status**: Draft
**Input**: User description — "use speckit. we need to refactor `ViewerApp.cs` for WoWViewer, so it is a bit smaller in terms of lines of code. It should be split up into a Viewer and Renderer library for this data — but we also need to refactor our renderer to use gpu acceleration and Level of Detail optimizations. We have some old plans talking about this, but we haven't committed to it in a single focused plan yet. Let's audit what we've got, and figure out the best route forwards. We also have a plan for building a bridge to Unreal Engine, but that's for beyond v0.5.0-dev work."

## Context

`wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` is 621,301 bytes, already organized as a C# partial class, with renderer code duplicated between the viewer-app `Rendering/` namespace (28+ files) and the thin shared `WowViewer.Core.Renderer` library (single-tile, no LOD, no instancing, no compute). There is no single owner plan that ties together the ViewerApp reduction, the renderer library promotion, GPU acceleration, and LOD. This spec is the result of an audit (`docs/architecture/spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md`) and consolidates overlapping partial plans (`specs/020`, `030`, `031`, `032`, `036`, `gv-14`-`gv-17`, `wow-viewer-library-completeness-plan-2026-05-06.md` Section 2.3 + Phase F, `game-viewer-host-plan-2026-05-13.md` slices 3-6) under one focused owner plan.

**Supersedes**: `specs/036-renderer-improvements` (archived by this spec).

**Parents**:
- `docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (top-level engine program)
- `docs/architecture/game-viewer-host-plan-2026-05-13.md` (app-host sub-plan; slices 3-6 covered by this spec)
- `docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md` (Phase F: Renderer Architecture)
- `docs/architecture/spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md` (this spec's audit)

## Locked Decisions

| # | Decision |
|---|---|
| D1 | Build a real shared renderer in `WowViewer.Core.Renderer` by **improving and moving the current `wow-viewer/src/viewer/WoWViewer/Rendering/*` code** (28+ files) into the shared library. The current WoWViewer renderer is the **source of truth for current behavior**. The legacy `gillijimproject_refactor/src/MdxViewer/Rendering/*` is read-only reference (RULE 1) and is **not** the source of truth. The `wow-viewer/src/viewer/WowViewer.App.Defunct/*` directory is **forbidden** as a source for this spec — it is not to be read, ported, or referenced. |
| D2 | GPU acceleration for v0.5.0-dev means: multi-tile batching, retained-mode VBO/IBO, UBO, instanced rendering. Compute shaders and async streaming are out of scope. |
| D3 | LOD coverage: terrain mesh LOD + object LOD (M2/WMO distance + draw-distance culling) + water LOD (`waterLOD`) + light LOD (`mapObjLightLOD`, `MaxLights`) + WDL far-horizon (1/16th LOD for the far distance) + BLP mipmap selection. |
| D4 | Backend: OpenGL modernized via Silk.NET. Vulkan primary is a follow-on spec. |
| D5 | LoC target: no numeric target. The user said "a bit smaller" / "substantially smaller." |
| D6 | Supersede `specs/036-renderer-improvements`. |
| D7 | v0.5.0-dev scope: full recommended core. |
| D8 | **Correctness oracle**: the Ghidra-confirmed 3.3.5 renderer research at `docs/architecture/wmo-render-pass-architecture-2026-05-30.md` is a **correctness oracle** for the new WMO renderer — it tells us *what the renderer must do* (interior/exterior dispatch, per-batch MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility). It is **not** a code source. The new renderer conforms to it because the native client does; we do not "port" the Ghidra-disassembly patterns. |

## Out of Scope (Hard)

- **Unreal Engine bridge** (post-v0.5.0-dev, per user; tracked by `specs/055-unreal-engine-bridge`).
- **Vulkan primary backend** (deferred; engine plan Phase E2 becomes a follow-on spec).
- **Compute shaders** for procedural terrain or GPU culling.
- **Async / triple-buffered resource streaming.**
- **Format readers and writers** (RULE 3 — complete; do not rewrite).
- **ML training pipeline** (RULE 7 — separate lane).
- **Audio engine** (separate plan).
- **VLM / PM4 workbench.**
- **M2 parity recovery** (tracked in 037/038).
- **Browser/embed delivery** (engine plan Pillar C, long-range).
- **`wow-viewer/src/viewer/WowViewer.App.Defunct/*`** (poisoned source — do not read, do not port, do not reference).
- **Porting from `gillijimproject_refactor/src/MdxViewer/Rendering/*`** (read-only reference per RULE 1; the Ghidra doc is the correctness oracle, not the legacy MdxViewer code).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Render Multiple Tiles From The Shared Renderer (Priority: P1)

As a viewer user, I need the viewer to load and render a real `3_3_5_12340` map with multiple adjacent terrain tiles visible at once, with terrain and objects appearing correctly across tile seams, so the world looks like a real outdoor space instead of a one-tile preview.

**Why this priority**: The current shared renderer is single-tile. Every downstream story (LOD, host split, performance) depends on multi-tile rendering existing first.

**Independent Test**: Load a staged `3_3_5_12340` outdoor map through `WowViewer.App` against the new shared renderer, place the camera at a tile boundary, and confirm the terrain mesh and the seam between adjacent tiles are both visible and visually correct.

**Acceptance Scenarios**:
1. **Given** a staged `3_3_5_12340` outdoor map is loaded, **When** the new shared renderer is invoked with an AOI of 3×3 tiles, **Then** all 9 tiles render and the seams between adjacent tiles are visually continuous.
2. **Given** a 5×5 AOI is requested, **When** the renderer processes the request, **Then** it uses instanced rendering and a single bind per material instead of 25 independent draw setups.
3. **Given** the renderer is rendering multiple tiles, **When** frustum culling is applied, **Then** tiles fully outside the camera frustum are skipped (verified via draw-call counter and per-tile diagnostic log).
4. **Given** a map switch occurs, **When** the previous map's GPU resources are released, **Then** no GL errors are reported and no out-of-memory symptoms appear.

---

### User Story 2 — Terrain Mesh LOD With WDL Far Horizon (Priority: P1)

As a viewer user, I need distant terrain to render at lower mesh resolution than near terrain, with the WDL (low-res world image) handling the far horizon the way the WoW engine intended (1/16th far-distance LOD), so the renderer can keep up with normal outdoor exploration instead of choking on a dense mesh at the horizon.

**Why this priority**: Terrain mesh LOD is the single largest frame-time win for outdoor maps and is the user's explicit ask.

**Independent Test**: Run a staged `3_3_5_12340` outdoor traversal route, capture the visible scene at three camera distances (near / mid / far), and confirm the mesh resolution drops monotonically with distance and the WDL is used beyond a configurable distance threshold.

**Acceptance Scenarios**:
1. **Given** the camera is near a tile, **When** the terrain mesh is built, **Then** it uses full 257×257 vertex density.
2. **Given** the camera is mid-distance from a tile, **When** the terrain mesh is built, **Then** it uses a reduced vertex density (configurable, default 33×33 or 17×17) without visible popping.
3. **Given** the camera is far from a tile, **When** the terrain LOD selector picks the far-distance bucket, **Then** the WDL is used as the visible representation and no per-chunk ADT mesh is built for that tile.
4. **Given** a tile crosses the LOD threshold during camera movement, **When** the LOD transition happens, **Then** the change is gradual (no sudden popping) and verified by a deterministic capture at the threshold distance.

---

### User Story 3 — Object LOD (M2/WMO) And Draw-Distance Culling (Priority: P1)

As a viewer user, I need M2 models and WMO world objects to use a lower detail level at distance, and to be culled past a configurable draw distance, so dense cities and instances stay responsive.

**Why this priority**: Object LOD is what makes indoor WMO scenes and crowded outdoor areas (Stormwind, Orgrimmar) usable. Without it, even a fast terrain renderer chokes on draw calls.

**Independent Test**: Run a staged `3_3_5_12340` route through a city tile, capture the frame, and confirm objects beyond the draw distance are not submitted and objects within the draw distance use the expected detail level by distance.

**Acceptance Scenarios**:
1. **Given** an M2 model with multiple detail levels (or a default near/far switch), **When** it is rendered at far distance, **Then** the far detail level is used (or the model is culled past `maxDrawDistance`).
2. **Given** a WMO instance, **When** it is beyond the configurable WMO draw distance, **Then** it is not submitted to the renderer (verified by zero draw-call entries).
3. **Given** an object crosses the detail threshold during camera movement, **When** the LOD switches, **Then** the transition is a single-frame swap and the swap is logged in the per-frame diagnostic stream.
4. **Given** a small object is occluded by a WMO wall, **When** occlusion culling is enabled, **Then** it is not submitted (verified by a deterministic scene with known occlusion).
5. **Given** a WMO group with `flags & 0x48 == 0` (interior mode per the Ghidra correctness oracle), **When** the camera is inside the WMO, **Then** the renderer dispatches `WmoBatchRenderPass.Group_Int / GroupColorTex_Int / GroupLightmapTex_Int` as appropriate, with MOCV vertex color and **no** dynamic lighting; the per-frame diagnostic stream records the per-batch pass chosen.
6. **Given** the same WMO group with `flags & 0x48 != 0` (exterior mode), **When** the camera is outside, **Then** the renderer dispatches `Group_Ext / GroupColorTex_Ext / GroupLightmapTex_Ext`, with dynamic lighting and no MOCV; the per-batch dispatch is logged.
7. **Given** a WMO group with `flags & 0x1000` (has liquid), **When** the group is rendered, **Then** the liquid is dispatched by `WmoLiquidType` (interior/exterior water vs magma) per the Ghidra doc's section 6.
8. **Given** a WMO group with `flags & 0x10000` (always visible / skybox shell), **When** the renderer dispatches, **Then** it uses the `RenderAlways` path, not the portal-walk path.
9. **Given** a WMO group with `flags & 0x88` (no render + no collide), **When** the renderer dispatches, **Then** the group is skipped entirely, with no minimap contribution.
10. **Given** an interior render with `intFog` enabled, **When** the WMO being rendered is the one the camera is inside, **Then** the interior fog (start, end, color from `DayNightGetInfo`) is applied to both the WMO surface and the interior water surface.

---

### User Story 4 — Water LOD (`waterLOD`) And Light LOD (`mapObjLightLOD`, `MaxLights`) (Priority: P2)

As a viewer user, I need water surfaces and dynamic lights to fall back to cheaper paths at distance, so the world remains responsive when many liquids and many lights are visible at once.

**Why this priority**: The WoW engine's `waterLOD` and `mapObjLightLOD` controls are explicitly named in the user's request and in `specs/036-renderer-improvements/spec.md` User Story 2. They are the difference between a usable city and a slideshow.

**Independent Test**: Load a staged `3_3_5_12340` map with a lake, river, and a city with many lights; capture frames at near / mid / far distances; confirm water mesh and light count drop with distance.

**Acceptance Scenarios**:
1. **Given** a liquid surface, **When** the camera is at far distance, **Then** the `waterLOD` reduced mesh (or shader) is used and the per-frame liquid draw-call count drops.
2. **Given** a scene with more than `MaxLights` active light sources, **When** the per-object light selection runs, **Then** only the closest `MaxLights` are bound to each object, and the rest are skipped.
3. **Given** `mapObjLightLOD` is enabled, **When** an M2 is far from the camera, **Then** the per-vertex dynamic light count is reduced to a flat ambient + key term.

---

### User Story 5 — BLP Mipmap Selection (Priority: P2)

As a renderer engineer, I need texture sampling to use the correct BLP mip level by distance so the GPU doesn't waste memory bandwidth sampling high-resolution mips for distant terrain and objects.

**Why this priority**: Mipmaps are an explicit part of the user's LOD request and are essentially free once a retained-mode texture cache exists. Not doing them is wasted GPU bandwidth.

**Independent Test**: Compare per-frame GPU texture bandwidth (or a deterministic frame capture) with mip selection enabled vs disabled; mip selection must not regress visual quality at typical viewing distances.

**Acceptance Scenarios**:
1. **Given** a terrain texture in BLP with mips, **When** it is sampled at far distance, **Then** the lowest appropriate mip is bound and the texture is not re-uploaded.
2. **Given** the texture cache is populated, **When** the same texture is requested by a second tile, **Then** the cache returns the existing GPU texture (verified by a single upload per texture in the diagnostic log).

---

### User Story 6 — ViewerApp Becomes A Thin Host Over The Shared Renderer (Priority: P1)

As a maintainer, I need `WowViewer` (the viewer app) to stop owning renderer implementation code so its `ViewerApp.cs` is substantially smaller and the renderer becomes a library that can be reused by other hosts (e.g. `WowViewer.Tool.ValidationCapture`).

**Why this priority**: The user asked for this explicitly. It is also the precondition for any future host to use the engine (post-v0.5.0-dev editor host, Unreal bridge, etc.).

**Independent Test**: After cutover, the viewer-app project references `WowViewer.Core.Renderer` and the legacy `wow-viewer/src/viewer/WoWViewer/Rendering/*` namespace is empty; `WowViewer.Tool.ValidationCapture` and the viewer app both invoke the same shared renderer entry point.

**Acceptance Scenarios**:
1. **Given** the cutover is complete, **When** the viewer app is built, **Then** `wow-viewer/src/viewer/WoWViewer/Rendering/*` is empty or absent, and `WowViewer.App` no longer depends on viewer-local renderer classes for the standard render path.
2. **Given** a host (viewer app, validation capture, or future editor) wants to render, **When** it calls the shared renderer entry point, **Then** the same renderer code path is used (no per-host renderer forks).
3. **Given** the cutover, **When** `ViewerApp.cs` is compared to its pre-spec size, **Then** it is "substantially smaller" per the user's stated criterion (no numeric target; reviewed by the maintainer).
4. **Given** the cutover, **When** an ImGui panel needs to inspect a renderer resource, **Then** the inspector calls into the shared library, not into a viewer-local type.

---

### User Story 7 — Real-Data Validation Suite Survives The Refactor (Priority: P1)

As a maintainer, I need the existing real-data validation surfaces (terrain alpha, WMO placement, M2 animation, headless capture) to keep working after the refactor, and the new LOD surfaces to be validation-testable, so the refactor does not silently regress parity.

**Why this priority**: The user's stated validation language rule: "library compile + tests in `wow-viewer` are primary proof. Real-data captures on staged `output/tmp/wowarchive-clients/0_5_3_3368` and `3_3_5_12340` are required for any change touching terrain, liquid, WMO, or M2 rendering."

**Independent Test**: Run the staged-client capture workflow on `0_5_3_3368` and `3_3_5_12340` before and after each cutover step; output captures must match within tolerance.

**Acceptance Scenarios**:
1. **Given** a deterministic capture of `Azeroth_30_48` on `3_3_5_12340` before the cutover, **When** the same capture is run after the cutover, **Then** the rendered output is visually equivalent (no missing terrain, no missing objects, no missing liquids).
2. **Given** the Alpha `0_5_3_3368` capture baseline, **When** the same capture is run against the new shared renderer, **Then** the Alpha MCAL alpha-mask parity (RULE: terrain alpha risk area) is preserved.
3. **Given** the headless capture path (`WowViewer.Tool.ValidationCapture`), **When** it is invoked against the new shared renderer, **Then** it produces the same `object_visibility_mask` it produced against the legacy renderer for the same tile and build.
4. **Given** a regression in any of the above, **When** the validation suite runs in CI / on demand, **Then** the failure is localized to a specific phase and a specific validation command (per the engine plan and AGENTS.md).

---

### Edge Cases

- What happens when a tile's mesh build fails (corrupt ADT, OOM)? The renderer must skip the tile, log a diagnostic, and continue rendering the rest of the AOI.
- What happens when the camera is exactly at a tile corner or on a WMO portal boundary? Frustum culling must not flicker between culled and visible.
- What happens when the GPU device is lost (driver crash, sleep/wake)? The renderer must attempt to recreate resources and surface a clear error to the host.
- What happens when BLP mip 0 is corrupt or truncated? The texture cache must fall back to a higher mip and log the issue instead of failing the tile.
- What happens when a WMO has more lights than `MaxLights`? The renderer must select the closest `MaxLights` deterministically, not arbitrarily.
- What happens when the user switches maps mid-frame? All GPU resources for the previous map must be released before the new map's first frame.
- What happens when the WDL is missing for the current map (some custom maps)? The far-horizon LOD must fall back to a flat low-res mesh and continue.
- What happens when a frame's draw-call count exceeds a budget? The renderer must log a per-frame budget diagnostic (the validation language rule).
- What happens during viewer-app startup before any map is loaded? The renderer must be in a no-op or default-scene state, not crash.
- What happens when the ImGui dockspace shell (spec 044) is docked and re-docked mid-frame? The renderer must not lose its viewport binding.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `WowViewer.Core.Renderer` MUST expose a public, host-agnostic multi-tile render entry point that accepts a backend-agnostic scene description (camera, AOI tiles, WMO/M2 visibility set, sky state, fog state, render variant flags).
- **FR-002**: The shared renderer MUST use retained-mode VBO/IBO plus UBO for all per-tile and per-frame state. Immediate-mode GL usage is forbidden in steady-state draw paths.
- **FR-003**: The shared renderer MUST support instanced rendering for repeated geometry (terrain chunks, WMO doodad instances, M2 instance groups).
- **FR-004**: The shared renderer MUST support frustum culling per tile, per WMO, and per M2 instance, and MUST emit a per-frame draw-call and instance-count diagnostic.
- **FR-005**: The shared renderer MUST consume `WorldTerrainLodSelector` and `WorldObjectVisibilityCollector` from `WowViewer.Core.Runtime.World` and MUST NOT reimplement LOD or visibility selection logic in the renderer.
- **FR-006**: The shared renderer MUST support terrain mesh LOD with at least three buckets (near full / mid reduced / far WDL) and a configurable per-bucket distance threshold.
- **FR-007**: The shared renderer MUST support WDL-based far-horizon rendering at the configured far distance, with the WDL treated as a 1/16th-scale LOD representation.
- **FR-008**: The shared renderer MUST support object LOD: M2 and WMO instances MUST switch to a reduced detail level at distance and MUST be culled past a configurable draw distance.
- **FR-009**: The shared renderer MUST support water LOD via the `waterLOD` control and MUST support light LOD via `mapObjLightLOD` and `MaxLights`.
- **FR-010**: The shared renderer MUST select the appropriate BLP mip level based on sampling distance, using the existing `WowViewer.Core.Renderer.Texture.TextureCache` (extended if needed).
- **FR-011**: The shared renderer MUST be usable headless via the existing `WowViewer.Core.Renderer.Headless/*` path and MUST continue to support `WowViewer.Tool.ValidationCapture`.
- **FR-012**: The shared renderer MUST be backend-agnostic at the contract level: all public APIs accept backend-neutral types. The OpenGL implementation lives in `WowViewer.Core.Renderer.OpenGL/*`; future Vulkan lives in `WowViewer.Core.Renderer.Vulkan/*` and conforms to the same contracts.
- **FR-013**: The viewer-app `wow-viewer/src/viewer/WoWViewer/Rendering/*` namespace MUST be retired once the shared renderer is feature-complete for the standard render path. During the cutover it is allowed to exist as a parallel path.
- **FR-014**: `WowViewer.App` (the viewer host) MUST NOT contain renderer implementation code; it may only wire inputs (camera, AOI, render variant flags) to the shared renderer.
- **FR-015**: `ViewerApp.cs` MUST be "substantially smaller" after the cutover (no numeric target per the user's D5; reviewed by the maintainer).
- **FR-016**: The shared renderer MUST validate against staged `output/tmp/wowarchive-clients/0_5_3_3368` (Alpha) and `3_3_5_12340` (LK) on every phase boundary. The validation command set is documented in the plan's `quickstart.md`.
- **FR-017**: The shared renderer MUST NOT regress the Alpha MCAL alpha-mask parity (terrain alpha risk area) or the LK 3.3.5 parity. Both must be verified by deterministic capture before and after each cutover step.
- **FR-018**: The shared renderer MUST NOT regress `specs/020-renderer-culling-and-tile-capture` (the P1 frustum culling fix is a hard prerequisite for tile capture correctness).
- **FR-019**: The shared renderer MUST be repo-independent (no source file references a path outside `wow-viewer/`).
- **FR-020**: The shared renderer MUST be cross-platform (Windows and Linux; macOS best-effort) and MUST NOT introduce a CUDA-only assumption (per `wow-viewer` AGENTS.md "open backend seams" rule).

### Key Entities

- **RenderScene**: backend-agnostic scene description. Inputs: camera (`SceneCamera`), AOI tile set, visible WMO/M2 set, sky state, fog state, render variant flags, water LOD setting, light LOD setting, terrain LOD distance thresholds.
- **RenderBackend**: the GPU backend interface. Implementations: `OpenGLRenderBackend` (v0.5.0-dev), `VulkanRenderBackend` (post-v0.5.0).
- **RenderResources**: per-tile VBO/IBO, per-frame UBO, per-material shader, per-texture handle. Lifetime is bound to the tile or to the frame, not to a host UI control.
- **LodSelector**: thin adapter over `WowViewer.Core.Runtime.World.WorldTerrainLodSelector` and `WorldObjectVisibilityCollector`. Same input, backend-neutral output.
- **TextureCache**: the existing `WowViewer.Core.Renderer.Texture.TextureCache`, extended to track per-texture mip selection and per-texture residency.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A staged `3_3_5_12340` outdoor map renders a 5×5 AOI of terrain tiles through the shared renderer with no missing terrain at tile seams, verified by deterministic capture.
- **SC-002**: A 3×3 AOI render uses one material bind per unique material, not 9 (verified by draw-call counter in the per-frame diagnostic stream).
- **SC-003**: Terrain mesh LOD produces three visibly distinct vertex densities at near / mid / far distances on `3_3_5_12340`; the WDL is the only representation beyond the configured far distance (verified by per-tile diagnostic log).
- **SC-004**: Object LOD culls all M2 / WMO instances beyond the configured draw distance, verified by a zero-draw-call count past the threshold.
- **SC-005**: `waterLOD`, `mapObjLightLOD`, and `MaxLights` are wired and observable via runtime diagnostic output (the per-frame log exposes the active control values).
- **SC-006**: BLP mip selection reduces per-frame texture bandwidth by a measurable amount on `3_3_5_12340` (validated against a pre-spec baseline; tolerance is documented in the plan).
- **SC-007**: `ViewerApp.cs` is "substantially smaller" after the cutover (D5: no numeric target; maintainer review).
- **SC-008**: `WowViewer.Tool.ValidationCapture` continues to produce the same `object_visibility_mask` for `Azeroth_30_48` on `3_3_5_12340` after the cutover (within documented tolerance).
- **SC-009**: A staged `0_5_3_3368` (Alpha) capture still passes the Alpha MCAL alpha-mask parity check after every cutover step (terrain alpha risk area).
- **SC-010**: `wow-viewer/src/viewer/WoWViewer/Rendering/*` is empty or absent after the cutover; the viewer-app project no longer compiles any renderer implementation code outside the shared library.

---

## Assumptions

- The user wants the refactor to be a one-time, focused effort inside v0.5.0-dev, not an open-ended modernization. The "one phase at a time" rule (RULE 8 + engine plan) applies.
- The user is willing to retire the legacy viewer-app `Rendering/*` namespace once the shared renderer is feature-complete, even though that means a temporary dual-path during the cutover.
- The existing `WowViewer.Core.Runtime.World` LOD/visibility/pass-routing surface is the canonical owner of LOD selection logic; the renderer is a consumer, not a parallel owner.
- The existing `WowViewer.Core.Renderer.Headless/*` capture path is the canonical way to validate renderer parity, and `WowViewer.Tool.ValidationCapture` is the canonical CLI tool that exercises it.
- The existing `specs/020-renderer-culling-and-tile-capture` P1 culling fix is correct and MUST NOT be regressed by any change in this spec.
- Staged clients `0_5_3_3368` and `3_3_5_12340` are available in `output/tmp/wowarchive-clients/` (RULE 9).
- `gillijimproject_refactor/src/MdxViewer` is read-only reference input; we do not modify it (RULE 1).
- Compute shaders, async streaming, and Vulkan primary backend are follow-on specs, not this one.
- The user is comfortable with "substantially smaller" being a code-review judgement, not a numeric metric (D5).
- The library completeness plan's Phase F is now this spec.
- The game-viewer host plan's slices 3-6 are now covered by this spec.

---

## Related Specs and Plans

- **Supersedes**: `specs/036-renderer-improvements` (archived by this spec).
- **Replaces intent of**: `wow-viewer-library-completeness-plan-2026-05-06.md` Section 2.3 + Phase F (now this spec).
- **Replaces intent of**: `game-viewer-host-plan-2026-05-13.md` slices 3-6 (now this spec).
- **Reuses**: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` Pillar C (Backend Separation), the `gv-14`-`gv-17` micro-plans in `docs/architecture/game-viewer-plan-pack-2026-05-14/`.
- **Must not regress**: `specs/020-renderer-culling-and-tile-capture` (P1 culling fix is prerequisite).
- **Adjacent, not in this spec**: `specs/030-wmo-render-pass-architecture`, `specs/031-terrain-cell-awareness`, `specs/032-native-renderer-parity` (WMO / terrain / native-renderer sub-concerns; the new shared renderer MUST consume their contracts, not redefine them).
- **Adjacent, not in this spec**: `specs/044-viewer-shell-usability`, `specs/045-scene-graph-workbench`, `specs/049-viewer-ui-consolidation` (viewer-app UX; not a renderer-library concern).
- **Adjacent, not in this spec**: `specs/055-unreal-engine-bridge` (post-v0.5.0-dev per user).
