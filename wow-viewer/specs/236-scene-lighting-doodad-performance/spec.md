# Feature Specification: Unified Scene Lighting, Doodad Performance & Client-Constrained World Pipeline (Spec 236)

**Feature Branch**: `v0.5.4-dev`

**Created**: 2026-09-15

**Status**: Draft — authored verbatim from operator directive; ready for execution

**Input**: Operator directive 2026-09-15:
> *"all MDX files now are tinted dark for no apparent reason. MDX objects randomly do and don't emit light, lighting across the board is horrendous when it comes to WMO and MDX/M2. We are missing an entire transform and lighting system that the game renderer has and utilizes for performance reasons, that we ignore, it seems... but we need that to make it look the part. Let's branch to a v0.5.4-dev branch, and figure out where the tooling stands as of right now - do a complete audit, what is implemented, what is missing, against the speckit plans, and then let's reconcile the work that has not been implemented, that has truly not been simply superceeded and replaced with better implementation work, and then write a new v0.5.4-dev speckit plan (epic) for improving lighting and renderer performance. I believe we are probably thinking about this all the wrong ways on many levels. I do have video from the earliest Alpha game footage, that shows we're not handling lights in WMO surfaces at all. None of the objects that emit light, cast light onto other object surfaces. We need to figure out a way to improve the doodad rendering, as it is STILL the main constraint that kills performance. Remember that WMO's are effectively demo scenes the art team used to show off all new doodad art assets when they were finished, so optimization was expected to exist in the engine, and not lay heavy on the art team who had no idea about the underlying lighting functionality, they just placed the objects and expected lit objects to emit light on other surfaces in the scene... as I expect but do not see. It's one thing for the objects to emit light, but for the light to not cast light onto other surfaces areound the area, that's a problem. Also, the custom map generator still uses random fucking shit from random expansions, even when generating new maps with 0.5.3 data, which is WRONG. it needs to be constrained to the listfile of the currently loaded client, no if's ands or buts. Also, none of hte map merge saving shit works, nothing. no GLB exports either. no files in the output folder. folder is created, but no glb is, ever."*

---

## 1. Background & Context

The viewer currently suffers from six critical issues that degrade visual fidelity, frame rate, and world construction:

1. **Dark MDX Shading**: Standalone models and mirrored/CW-wound MDX geometry invert normals due to an unconditional `gl_FrontFacing` normal negation in the fragment shader, dropping direct diffuse light to zero and leaving models shaded only by 35% ambient light.
2. **Missing Half-Lambert Diffuse**: MDX models use raw Lambert (`max(nDotL, 0.0)`), causing sharp black cutoffs on curved surfaces instead of the smooth Half-Lambert wrap used by WMOs and authentic WoW rendering.
3. **Isolated Lighting Islands**: MDX `LITE` chunks only illuminate the model instance that contains them. Torches, lanterns, and braziers do not cast any light onto surrounding terrain, WMO geometry, or nearby doodads.
4. **WMO Surfaces Ignore Local Lights**: WMO shaders only evaluate global outdoor directional light and ambient light. Embedded WMO `MOLT` point lights and placed doodad lights do not cast light onto WMO walls, floors, or ceilings.
5. **Doodad Performance Bottleneck**: Doodads remain the primary frame-time constraint. Lack of multi-placement batching and unified GPU instancing results in thousands of unbatched draw calls in dense interior and exterior scenes.
6. **Expansion-Polluted Map Generator & Broken Save/Export**: The procedural terrain generator hardcodes Wrath (`EXPANSION02`) snow textures and `.m2` models even when 0.5.3 Alpha data is active. GLB exports write to `bin/Debug` instead of workspace root `output/export`, and map merge saving (Spec 234) has no backend implementation.

---

## 2. User Scenarios & Requirements

### US1 — Authentic MDX/M2 Diffuse Shading & Normal Fidelity (Priority: P0)
Models must render with authentic diffuse illumination without dark tinting or flipped normals.
- **FR-001**: Remove unconditional `!gl_FrontFacing` normal inversion that corrupts mirrored and CW-wound geometry.
- **FR-002**: Apply Half-Lambert diffuse wrapping (`diff = ((N·L) * 0.5 + 0.5)^2`) to MDX and M2 models to match WMO and client visual softness.
- **FR-003**: Ensure models with `SphereEnvMap` evaluate specular Blinn-Phong highlights accurately without washing out base textures.

### US2 — Multi-Surface Light Casting (Doodad & WMO Lights) (Priority: P0)
Light-emitting objects (torches, braziers, lamps, crystals) and WMO `MOLT` lights must illuminate surrounding surfaces.
- **FR-004**: Implement a spatial `SceneLightManager` that aggregates active light sources (position, color, intensity, radius).
- **FR-005**: WMO shaders must evaluate up to 8 active local lights affecting each group/cluster.
- **FR-006**: Terrain shaders must evaluate nearby local lights affecting each chunk.
- **FR-007**: Doodad shaders must evaluate nearby scene lights affecting their bounding volume.

### US3 — Doodad Rendering Performance & GPU Instancing Overhaul (Priority: P1)
Doodads must render at high frame rates through aggressive batching and culling.
- **FR-008**: Unify WMO doodad rendering into GPU instanced batches across all visible WMO instances.
- **FR-009**: Implement bounding-sphere culling for doodad clusters prior to draw queuing.
- **FR-010**: Support distance fade and alpha transitions without breaking batching into single draws.

### US4 — Loaded-Client Constrained Map Generation (Priority: P1)
The New Map Creator must only use assets available in the currently loaded client.
- **FR-011**: Query `IDataSource` listfile before assigning biome textures or doodad placements.
- **FR-012**: If Alpha 0.5.3 is active, use only `.mdx` models and Alpha tilesets; reject `.m2` and expansion paths.
- **FR-013**: Provide fallback asset resolution or clear UI warnings if a requested theme asset is missing from the client.

### US5 — Map Merge Save Pipeline & Reliable GLB Export (Priority: P1)
Composed maps and 3D scenes must save reliably to disk.
- **FR-014**: Route all exports to project workspace root `output/export/`.
- **FR-015**: Implement `MapSaveService` to serialize composed cartography layers to Alpha 0.5.3 WDT/ADT and LK v18 ADT.
- **FR-016**: Enable GLB viewport scene and collision export when terrain/maps are active in the viewer.
