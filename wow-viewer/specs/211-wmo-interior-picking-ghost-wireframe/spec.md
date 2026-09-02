# Spec 211: WMO Interior Ray Picking, Doodad Selection & Ghost Transparent Wireframes

## Overview & User Intent

In complex 3D environments (cities, taverns, castles, dungeons), World Map Objects (WMOs) have large spatial bounding boxes that encompass entire building envelopes, roofs, and large interior spaces. Previously:
1. **Container Lockout**: Because a WMO's outer bounding box is so large, any ray aimed toward an interior table, chair, NPC, or item intercepted the WMO's outer AABB first ($d_{\text{wmo}} < d_{\text{interior}}$). The viewer selected the WMO and occluded every interior object along the ray, making it impossible to select objects inside buildings.
2. **WMO Doodad Inaccessibility**: WMOs spawn internal doodads (chairs, tables, torches, chests, chandeliers via MODD/MODS), but these instances were not queryable or selectable via 3D raycast.
3. **Opaque Wireframe Occlusion**: Toggling wireframe mode on M2/MDX, WMO, or terrain rendered solid textured polygons in line mode or opaque fills with black lines. The user requested a "ghosted" transparent wireframe mode where textured polygons are rendered with ~33% opacity (`alpha = 0.33f`), allowing the wireframe to provide primary structural representation while retaining ~33% visual texturing for context, with terrain preserving texturing visibility under distinct wireframe lines.

---

## User Stories

### US-1: WMO Bounding Box Click Fall-Through & Interior Picking
- **As a** 3D world investigator and editor,
- **I want** clicking inside a building's volume to fall through to interior MDX/M2 models, nested WMOs, or WMO doodads,
- **So that** I am never locked into selecting the outer WMO shell when clicking objects situated inside its bounding box.

### US-2: WMO Doodad Ray Picking & In-Scene Selection
- **As a** world viewer user,
- **I want** to click directly on WMO-spawned doodads (furniture, torches, decorations),
- **So that** they can be inspected, highlighted, and framed in the 3D scene.

### US-3: Ghost Transparent Wireframe Rendering (33% Alpha Blend)
- **As a** 3D modeler and terrain investigator,
- **I want** toggled wireframe modes to render textured polygons at ~33% opacity with prominent wireframe overlays on top,
- **So that** wireframe lines dominate visual representation without losing spatial context, both on objects (M2/MDX/WMO) and on terrain.

---

## Acceptance Criteria

### Criteria 1: Interior Object Priority Over Enclosing WMO (AC-001)
- When a camera ray intersects both an outer WMO AABB and an interior MDX/M2 object (or nested WMO) whose center or bounds lie within the WMO's bounding box:
  - The interior object takes precedence over the enclosing container WMO.
  - The container WMO is NOT chosen merely because its outer bounding box intercepted the ray at an exterior roof/wall.

### Criteria 2: WMO Doodad Raycast Picking (AC-002)
- Raycasts through WMO doodads (`_doodadInstances` in `WmoRenderer`) detect intersections against doodad bounding boxes transformed into world space.
- A hit WMO doodad produces a selectable candidate labeled `WMO Doodad [index] {modelName}`.
- Selecting it highlights the doodad in the 3D scene and links to the WMO Doodad Inspector.

### Criteria 3: Ghosted Transparent Wireframe on Objects (AC-003)
- When wireframe mode is enabled on an M2/MDX model or WMO:
  - Textured polygons are drawn with ~33% opacity (`fadeAlpha = 0.33f`) with alpha blending enabled (`GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA`).
  - Wireframe lines are drawn prominently on top with polygon offset (`glPolygonOffset(-1.0f, -1.0f)`) to eliminate z-fighting.

### Criteria 4: Dual-Pass Wireframe on Terrain (AC-004)
- When wireframe mode is enabled on terrain:
  - The textured terrain remains clearly visible.
  - Wireframe lines are overlaid crisply on top using `PolygonMode.Line` with polygon offset, preventing the terrain surface from turning into an empty grid.

---

## Technical Constraints & Invariants

1. **Format Readers Frozen**: Do not touch raw format readers or decoders (`WmoReader`, `MdxFile`, `M2Model`).
2. **Performance Budget**: WMO doodad picking must be bounding-box early-rejected against the parent WMO's bounding box so raycasting remains fast even on dense maps.
3. **Clean Blend State Restoration**: All OpenGL blend modes, polygon offsets, and polygon modes must be restored cleanly to `GL_FILL` after rendering wireframe passes.
