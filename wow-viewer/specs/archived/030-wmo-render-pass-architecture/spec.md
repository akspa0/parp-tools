# Feature Specification: WMO Render Pass Architecture (Ghidra-Confirmed)

**Feature Branch**: `030-wmo-render-pass-architecture`

**Created**: 2026-05-30

**Status**: Research complete — consumed by spec 056

**Input**: Ghidra RE of wowclient.exe build 3368 reveals the complete WMO group rendering pipeline, including the render mode dispatch, batch structure, per-batch material flags, interior/exterior lighting split, liquid rendering, lightmap UV handling, and portal-walk visibility. The current `WmoRenderer.cs` in MdxViewer implements a simplified 4-pass system that does not match the native client's dispatch logic, producing incorrect lighting, missing render passes, and broken interior fog. This spec captures the Ghidra-confirmed architecture as documentation and prescribes the fixes needed for `WowViewer.Core.Runtime` to implement WMO rendering correctly.

## Problem Statement

The existing `WmoRenderer.cs` renders WMO groups through a simplified 4-pass approach that diverges from the native client in several ways:

1. **Render mode dispatch is wrong**: The native client uses function pointers at `DAT_00ec1b98` (interior) and `DAT_00ec1ca0` (exterior) set during `PrepareUpdate`, which select between distinct render paths based on group flags. MdxViewer hardcodes a different pass order.
2. **Per-batch material flags are ignored**: Each MOMT batch has flags (bit0=lighting, bit1=fog, bit2=culling, bit3=texture addr, bit4=wrap/clamp, bit0x10=emissive, bit0x20=window-lit exterior sun override). MdxViewer does not test these per-batch.
3. **Interior fog is missing**: Applied when `DayNightGetInfo()->intFog != 0` and the WMO is the camera's map object. MdxViewer has no interior fog path.
4. **Lightmap UV handling differs**: The native client has `RenderGroupLightmap` (lightmap UV + tex, no separate color) and `RenderGroupLightmapTex` (tex + lightmap on tex1), with different interior vs exterior behavior. MdxViewer conflates these.
5. **Liquid rendering is incomplete**: The native client dispatches `RenderInteriorWater`, `RenderExteriorWater`, or `RenderMagma` based on liquid type flags. MdxViewer's liquid path is simplified.
6. **Group flags affect render path**: `0x88` = skip (no render, no collide, no minimap), `0x48` = special render path (exterior), `0x1000` = has liquid. These flag checks are missing or incorrect.

## Ghidra RE Findings (Build 3368)

### Render Group Dispatch (`CMapObj::RenderGroup` at 0x0069bd50)

```
For each visible frustum:
  if (group.flags & 0x48) == 0:
    call DAT_00ec1b98()        // Interior render callback
  else:
    call DAT_00ec1ca0(group, frustumIdx)  // Exterior render callback

After all frusta:
  if (group.flags & 0x1000):
    RenderLiquid(group)
  if (CWorld::enables & 0x40000000):
    RenderGroupNormals(group)    // Debug normals visualization
  if (CWorld::enables & 0x1000):
    RenderPortals(group)         // Debug portal visualization
```

### Render Pass Functions

| Function | Address | Behavior |
|----------|---------|----------|
| `RenderGroup_Int` | 0x0069db70 | Interior base pass — MOCV vertex color, no texture |
| `RenderGroup_Ext` | 0x0069da70 | Exterior base pass — no MOCV, vertex lit |
| `RenderGroupColorTex_Int` | 0x0069d210 | Interior textured + MOCV, lighting OFF |
| `RenderGroupColorTex_Ext` | 0x0069d450 | Exterior textured + MOCV, lighting ON |
| `RenderGroupColorTex` | 0x0069d6f0 | Dispatches Int/Ext based on interior flag |
| `RenderGroupLightTex` | 0x0069ca10 | Textured + per-batch material flags, most common pass |
| `RenderGroupLightmap` | 0x0069d770 | Lightmap UV channel + tex, no separate vertex color |
| `RenderGroupLightmapTex_Int` | 0x0069cc90 | Interior: tex + lightmap, lighting OFF, lightmap on tex1 |
| `RenderGroupLightmapTex_Ext` | 0x0069cf00 | Exterior: tex + lightmap, lighting ON, no lightmap on tex1 |
| `RenderGroupLightmapTex` | 0x0069d190 | Dispatches Int/Ext based on interior flag |
| `RenderGroupTex` | 0x0069d8c0 | Texture only, white vertex color, no lighting |
| `RenderGroupBsp` | 0x0069df60 | BSP polygon renderer for collision/debug (3 sub-passes by poly flags 0x20, 0x04, 0x08) |

### Batch Structure

Each WMO group has 4 interior sub-batches and 4 exterior sub-batches:
- `intBatch[4]` / `extBatch[4]`
- Each batch: `{ vertStart, gxbuf, batchStartIndex, batchCount }`
- Batch iteration follows material index order

### Per-Batch Material Flags (from MOMT)

| Bit | Meaning |
|-----|---------|
| 0 | Lighting enabled |
| 1 | Fog enabled |
| 2 | Backface culling |
| 3 | Texture address mode |
| 4 | Wrap vs clamp |
| 0x10 | Emissive (self-illum) |
| 0x20 | Window-lit (exterior sun override for interior windows) |

### Liquid Rendering (`RenderLiquid` at 0x0069e4b0)

```
Determine liquid type from first tile: liquid & 0xF
  type 0,4,8: water → interior or exterior based on group flags & 0x48
  type 2,3,6,7: magma → RenderMagma()
Select GxVS_PassThru, get liquid texture from CMap::GetLiquidTexture(type)
Push render state, disable culling
Set texture blend mode 3 for water
Interior water: vertex color from material diffColor, interior fog if DayNightGetInfo()->intFog && this == camMapObj
Exterior water: vertex+normal+color+tc0, DayNight lighting for water color
Pop render state
```

### Interior/Exterior Render Dispatch (`IntRender` / `ExtRender`)

**IntRender**: Portal-walk traversal from player's group list. For each group:
- If `flags & 0x10000`: skip (always-render group, handled separately via `RenderAlways`)
- If `flags & 8`: frustum cull, then `RRenderThruPortals(groupIdx, 0xFFFF, screenRect, 0)`

**ExtRender**: For each group:
- If `flags & 0x10000`: frustum cull, `RenderAlways(groupIdx)`
- If `flags & 8`: frustum cull, `RRenderThruPortals(groupIdx, 0xFFFF, screenRect, 0)`

**RenderAlways**: Used for groups flagged `0x10000` (always visible regardless of portal walk).

### Portal Walk (`RRenderThruPortals` at 0x0069bf60)

Recursive portal traversal:
- Start from a group index
- Walk all portal references to adjacent groups
- Check portal is visible (screen-space rect intersection)
- Clip portal rect to parent rect
- Recurse into adjacent group with depth limit `maxRLevel`
- Track visited portals via `gRenderCount` stamp
- For interior rendering, build `_extViewList` for exterior-visible groups

### Lightmap System (`CreateLightmaps` at 0x006adba0)

- Each group can have `lightmapTexCount` lightmap textures
- Lightmaps are 256x256, format `LIGHTMAP_FORMAT` (DXT if supported, else RGB565)
- Created on first demand with `TextureCreate("Lightmap", ...)` 
- User-data callback `UpdateLightmapTex` for on-demand streaming
- Flush timeout: 30 seconds (`lightmapTexFlushTime = 30.0`)

## Scope

### In Scope

- Documenting the complete WMO render pass architecture from Ghidra findings in `wow-viewer/docs/architecture/`
- Updating `WmoRenderer.cs` in `gillijimproject_refactor` (READ-ONLY reference) is NOT in scope — the fixes go in `WowViewer.Core.Runtime`
- Implementing the correct render pass dispatch in `WowViewer.Core.Runtime` based on the Ghidra-confirmed architecture
- Per-batch material flag handling (lighting, fog, culling, emissive, window-lit)
- Interior fog support when `intFog != 0` and camera is inside the WMO
- Correct lightmap UV pass selection (Lightmap vs LightmapTex, interior vs exterior)
- Liquid type dispatch (interior water, exterior water, magma)

### Out of Scope

- GPU shader implementation (this spec covers the dispatch logic and state setup, not shader code)
- Terrain chunk rendering (separate spec 031)
- Minimap BLP harvesting (separate spec 029)
- M2 doodad rendering within WMO groups
- BSP collision rendering (debug-only in native client)

## User Scenarios & Testing

### User Story 1 — WMO render pass architecture is documented (Priority: P1)

An engine developer can read a single architecture doc that describes every WMO render pass, the dispatch logic, batch structure, material flags, liquid types, lightmap system, and interior/exterior split as confirmed by Ghidra, without needing to run Ghidra themselves.

**Why this priority**: The documentation is the foundation. Without it, all implementation work is guessing. This captures the Ghidra evidence before context is lost.

**Independent Test**: The doc exists at `wow-viewer/docs/architecture/wmo-render-pass-architecture-2026-05-30.md` and covers every function listed in the findings table above.

**Acceptance Scenarios**:

1. **Given** the architecture doc, **When** a developer reads it, **Then** they can identify which render pass handles a WMO group with `flags & 0x48 == 0` and MOCV data (answer: `RenderGroupColorTex_Int`).
2. **Given** the architecture doc, **When** a developer reads the liquid section, **Then** they know which liquid type values map to water vs magma and how interior/exterior water dispatch differs.
3. **Given** the architecture doc, **When** a developer reads the material flags section, **Then** they can determine per-batch whether lighting, fog, culling, emissive, or window-lit should be applied.
4. **Given** the architecture doc, **When** a developer reads the portal walk section, **Then** they understand the recursive traversal, depth limiting, and screen-rect clipping.

---

### User Story 2 — WMO render dispatch matches native client (Priority: P2)

A viewer user sees WMO groups rendered with the correct pass selection, per-batch material flags, and interior/exterior split, matching the native client's visual output for the same WMO in the same viewing conditions.

**Why this priority**: This is the core fix — making the viewer produce correct WMO renders. P2 because the documentation (P1) must exist first.

**Independent Test**: Render a known WMO (e.g., Deadmines interior) and compare group-by-group output against a reference screenshot from the native client. Verify lighting, fog, and liquid behavior match.

**Acceptance Scenarios**:

1. **Given** a WMO group with `flags & 0x48 == 0` (interior), **When** it is rendered, **Then** the interior render callback path is selected (not the exterior path).
2. **Given** a WMO batch with material flags `bit0=0`, **When** it is rendered, **Then** lighting is disabled for that batch only.
3. **Given** a WMO group with `flags & 0x1000` and liquid type 0, **When** it is rendered, **Then** `RenderInteriorWater` or `RenderExteriorWater` is called based on `flags & 0x48`.
4. **Given** a WMO group with lightmap data, **When** it is rendered in interior mode, **Then** `RenderGroupLightmapTex_Int` path is used (lighting OFF, lightmap on tex1).
5. **Given** a WMO group with lightmap data, **When** it is rendered in exterior mode, **Then** `RenderGroupLightmapTex_Ext` path is used (lighting ON, no lightmap on tex1).

---

### User Story 3 — Interior fog works inside WMO groups (Priority: P3)

A viewer user inside a dungeon WMO sees interior fog that matches the native client's fog density, color, and start/end distances, which change based on the day/night cycle's `intFog` settings.

**Why this priority**: Interior fog is a visible quality issue but is secondary to getting the basic render dispatch correct (P2).

**Independent Test**: Enter a known dungeon WMO and verify fog is visible and matches native client density/color.

**Acceptance Scenarios**:

1. **Given** the camera is inside a WMO and `DayNightGetInfo()->intFog != 0`, **When** the WMO is the camera's current map object, **Then** interior fog is applied with the correct start, end, and color values.
2. **Given** the camera is inside a WMO but `intFog == 0`, **When** the WMO is rendered, **Then** no interior fog is applied.
3. **Given** interior fog is active, **When** water is rendered inside the same WMO, **Then** interior fog also applies to the water surface.

---

### Edge Cases

- WMO groups with `flags & 0x88` (no-render, no-collide) should not be rendered at all — currently MdxViewer may attempt to render some of these.
- WMO groups with `flags & 0x10000` (always visible) require `RenderAlways` dispatch, not portal-walk — these are typically skyboxes or exterior shells visible from all interior groups.
- The `window-lit` flag (0x20) is an exterior sun override for interior window polygons — this is a special blending mode that doesn't exist in MdxViewer.
- Some WMO groups may have 0 batches in one or more sub-pass arrays — empty passes should be skipped.
- Lightmap textures may not exist for all groups — the `lightmapTexCount` field determines availability.

## Requirements

### Functional Requirements

- **FR-001**: The WMO render pass architecture MUST be documented in `wow-viewer/docs/architecture/wmo-render-pass-architecture-2026-05-30.md` covering all functions, dispatch logic, batch structure, material flags, liquid types, lightmap system, and portal walk.
- **FR-002**: `WowViewer.Core.Runtime` WMO renderer MUST implement the correct render mode dispatch: interior callback for `flags & 0x48 == 0`, exterior callback for `flags & 0x48 != 0`.
- **FR-003**: The WMO renderer MUST test per-batch material flags from MOMT: lighting (bit0), fog (bit1), culling (bit2), emissive (bit0x10), window-lit (bit0x20).
- **FR-004**: The WMO renderer MUST select the correct lightmap pass based on interior/exterior mode: `LightmapTex_Int` (lighting off, lightmap on tex1) vs `LightmapTex_Ext` (lighting on, no lightmap on tex1).
- **FR-005**: The WMO renderer MUST dispatch liquid rendering based on liquid type: types 0/4/8 → water (interior or exterior based on group flags), types 2/3/6/7 → magma.
- **FR-006**: The WMO renderer MUST apply interior fog when `intFog != 0` and the rendered WMO is the camera's current map object.
- **FR-007**: The WMO renderer MUST skip groups with `flags & 0x88` entirely (no render, no collide).
- **FR-008**: The WMO renderer MUST handle `flags & 0x10000` groups via an always-render path, not portal-walk.
- **FR-009**: All code MUST live under `wow-viewer/`.
- **FR-010**: The architecture doc MUST include the Ghidra function addresses as references for future RE work.

### Key Entities

- **WMO Render Pass**: A specific rendering function selected by group flags and batch properties. Each pass configures GPU state (lighting, textures, blending, fog) differently.
- **WMO Batch**: A sub-range of indices within a group's index buffer, associated with a material (MOMT entry) that carries per-batch flags.
- **Interior/Exterior Split**: WMO groups are rendered differently when the camera is inside the WMO (interior: MOCV vertex color, no dynamic lighting, interior fog) vs outside (exterior: dynamic lighting, sun, no interior fog).
- **Portal Walk**: Recursive visibility traversal through WMO group portals, clipping screen-space rects and limiting recursion depth.
- **Lightmap**: Pre-baked 256x256 texture containing static lighting for WMO group surfaces. Applied differently in interior vs exterior mode.
- **Liquid Type**: A 4-bit value per liquid tile that determines water (types 0/4/8) vs magma (types 2/3/6/7) rendering behavior.

## Success Criteria

- **SC-001**: The architecture doc covers all 11 render pass functions listed in the findings table with addresses and behavior descriptions.
- **SC-002**: The architecture doc describes the per-batch material flag system with all 7 flag bits.
- **SC-003**: A WMO with known group flags renders with the correct pass selection (verifiable by comparing render output group-by-group against native client screenshots).
- **SC-004**: Interior fog is visible when inside a dungeon WMO with `intFog != 0`.
- **SC-005**: Liquid rendering dispatches correctly for both water and magma types.

## Assumptions

- The Ghidra findings from build 3368 (0.5.3) apply to later builds (3.3.5) with minor additions (e.g., more MOMT flag bits). The core dispatch architecture is the same.
- The `WowViewer.Core.Runtime` WMO renderer will use Silk.NET.OpenGL as the rendering backend (per constitution).
- The function pointer dispatch (`DAT_00ec1b98` / `DAT_00ec1ca0`) is set during scene preparation and depends on the current day/night state and camera position. The `wow-viewer` runtime will need an equivalent state mechanism.
- M2 doodad rendering within WMO groups is out of scope for this spec and will be addressed separately.

## Relationship to Other Specs

- **Informs**: `029-wmo-minimap-signal` — correct WMO rendering is needed for the GPU-rendered top-down capture path that complements the BLP harvest.
- **Extends**: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (viewer-first + UE bridge; WMO rendering serves the viewer and the UE bridge equally)
- **Complements**: `031-terrain-cell-awareness` — terrain and WMO are the two primary world rendering subsystems.
- **Replaces**: The ad-hoc WMO rendering knowledge previously scattered across chat sessions and `WmoRenderer.cs` comments.
