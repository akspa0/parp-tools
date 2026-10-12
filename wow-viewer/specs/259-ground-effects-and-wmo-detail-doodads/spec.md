# Spec 259: Ground Effects & WMO Detail Doodad Engine

**Owner**: Epic 249 (Renderer Performance, Lighting & Correctness) & Epic 248 (Formats & Readers)  
**Origin**: Operator prompt (2026-10-02) based on recent wowdev.wiki updates (2026-09-28/30) & 1.60 client support  
**Status**: Proposed / Authoring  

---

## 1. Executive Summary

Ground effects (detail doodads such as grass, flowers, shrubs, and pebbles) bring world surfaces alive across both outdoor terrain and WMO architectural surfaces. Based on the 2026-09-28 and 2026-09-30 wowdev.wiki disclosures and client reverse-engineering (3.3.5 / 1.60.1), ground effects are lightweight, single-batch M2 models placed dynamically based on DBC/DB2 texture mappings and WMO root `MDDL` chunks.

This spec introduces comprehensive support for ground effects and detail doodads across:
1. **Terrain ADT Chunks**: Driven by `GroundEffectTexture.dbc`/`db2` and `GroundEffectDoodad.dbc`/`db2`, respecting slope limits ($Z_{\text{normal}} \ge 0.4$), normal alignment, MCCV vertex lighting, and MCSH shadow maps.
2. **WMO Geometry**: Driven by the `MDDL` (Map Detail Doodad Layer) root chunk and `MOC2` vertex weights, placing foliage and debris directly on WMO structures.
3. **1.60 Client Data**: Support for up to 8 terrain layers per chunk, FileDataID resolution in CASC, and modern DB2 structures.
4. **Viewer Controls & Performance**: High-performance GPU instancing, distance culling (`groundEffectDist`), density control (`groundEffectDensity`), and UI workbench toggles.

---

## 2. Requirements & Acceptance Criteria

### User Stories

- **US1: Authentic Terrain Ground Effects**: As an explorer in the viewer, when viewing terrain with grass/foliage textures, I see authentic detail doodads rendered at high FPS, properly leaning on slopes and shaded by terrain lighting.
- **US2: WMO Detail Doodads (`MDDL`)**: When viewing WMOs with rooftop grass, overgrown ruins, or floor debris (e.g. Arathor Monastery, Dalaran), the viewer parses `MDDL` layers and renders the detail doodads placed upon the WMO batches.
- **US3: 1.60 Client Data Conformance**: Ground effect and WMO detail doodad rendering functions seamlessly with modern 1.60.1 CASC data, supporting up to 8 layers per ADT chunk and CASC FileDataID models.
- **US4: Runtime Controls & Quality Settings**: In the Render Quality and Settings panels, I can toggle ground effects on/off, adjust density ($0.0 - 2.0$), and adjust draw distance ($0 - 150\text{ m}$).

### Acceptance Criteria

- **AC-001**: `GroundEffectLookup` parses both legacy DBC and modern DB2 tables for `GroundEffectDoodad` and `GroundEffectTexture`, extracting flags (`0x1` AlignToNormal, `0x2` IgnoreMCCV), density, models/FDIDs, and weights.
- **AC-002**: Slope limit rule is strictly enforced: no ground effect is placed on triangles where the surface normal's $Z < 0.4$ (slopes steeper than $\approx 66^\circ$).
- **AC-003**: Flag `0x1` (AlignToNormal) aligns the doodad's up-vector to the terrain triangle normal with random yaw; unflagged doodads remain upright along world $Z$.
- **AC-004**: Flag `0x2` (IgnoreMCCV) renders plain white (`0xFFFFFFFF`), whereas unflagged doodads interpolate MCCV vertex colors across the supporting triangle.
- **AC-005**: MCSH shadow map marks alpha to $0$ on shadowed spots, or darkens to $70\%$ in non-shader mode.
- **AC-006**: `WmoMddlReader` parses the WMO root `MDDL` chunk (`Layer detailDoodadLayers[]` and per-group RLE placement data), exposing detail doodad placements in `WmoRenderDocument`.
- **AC-007**: GPU rendering leverages instanced drawing so that thousands of detail doodads incur minimal draw-call overhead.
- **AC-008**: Code respects god-class freeze (AGENTS.md §10), isolated in dedicated services (`GroundEffectPlacementService`, `WmoMddlReader`, `GroundEffectSceneService`).
