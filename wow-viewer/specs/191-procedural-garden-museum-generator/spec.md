# Feature Specification: Procedural Garden Museum Map Generator & Dense Calibration Corpus

**Feature Branch**: `191-procedural-garden-museum-generator`  
**Spec ID**: `191`  
**Status**: Draft  
**Created**: 2026-08-29  

---

## 1. Overview & Problem Statement

The initial calibration map generator (`rosetta-generate`) established functional multi-map splitting, authentic DBC patching, TRS/BLP minimap generation, and $+20\text{Z}$ model placement. However, the resulting maps exhibit several critical visual, geometric, and ergonomic limitations:

1. **Low Object Density**: Large 4-chunk (133.3m) and 2-chunk (66.6m) grid cells place far too few objects per tile, creating vast expanses of empty terrain. Small items, weapons, and doodads are marooned in oversize zones.
2. **Tiny Model Visuals in In-Game Perspective**: In third-person and first-person character views, standard-scale M2 weapons, items, small critters, and doodads (often 0.2m–1.5m tall) are nearly invisible when running around the world.
3. **Rigid, Unplayable Terrain Mesh**: The terrain mesh is flat with abrupt, jarring square raised quads next to objects. Characters get trapped on steep stepped cliff edges, making navigation frustrating.
4. **Boring, Rigid Texture Alpha Splatting**: Texture splatting is blocky and square (rigid 1-chunk blocks), lacking organic garden landscape blending, walkable paths, and visual finesse. Pedestals have noisy texture patterns directly beneath exhibits rather than clean exhibit floors.
5. **Lack of a Generic Procedural Map Generation Engine in Editor**: There is no generic, modular map generation surface in the Editor/Core layers that can be configured by users, run procedurally with diverse thematic styles (e.g., Garden Museum, Marble Pavilion, Botanical Arboretum), or converted across all WoW eras (Alpha 0.5.3, Classic 1.12.1, WotLK 3.3.5).

---

## 2. User Stories & Acceptance Criteria

### User Story 1 (US1): Dense Multi-Tier Layout Packing & Adaptive Grid Sizing
*As a map generator user and ML researcher, I want high-density exhibit packing with adaptive cell sizes so that hundreds of items and props can be placed efficiently per tile without wasting space.*

- **AC1.1**: Support multi-tier cell subdivisions: Micro ($16.66\text{m} = \frac{1}{8}\text{ chunk}$, $33.33\text{m} = \frac{1}{4}\text{ chunk}$), Standard ($66.66\text{m} = \frac{1}{2}\text{ chunk}$, $133.33\text{m}$), and Grand ($266.66\text{m}$, $533.33\text{m}$).
- **AC1.2**: Automatically categorize and pack assets by bounding radius ($r \le 2\text{m}$, $2\text{m} < r \le 8\text{m}$, $8\text{m} < r \le 25\text{m}$, $r > 25\text{m}$) into optimal density courtyards.
- **AC1.3**: Support `--density` presets (`compact`, `balanced`, `spacious`) yielding 4x–16x more exhibits per tile for items/doodads.

### User Story 2 (US2): Configurable M2 Model Scaling (2x–5x Detail Magnification)
*As an in-game explorer and computer vision dataset consumer, I want M2 models scaled up by 2x–5x so that fine details, engravings, facial features, and weapon textures can be clearly inspected at character eye-level.*

- **AC2.1**: Support `--m2-scale <float>` (default $2.5\times$ for weapons/items, $2.0\times$ for creatures/characters, $1.0\times$ for giant WMOs/titans).
- **AC2.2**: Placed M2 scale must be written into `MDDF.Scale` correctly across LK (1024 = 1.0x), Classic, and Alpha converters.
- **AC2.3**: Pedestal dimensions, clearance radii, and $+Z$ elevation offsets must dynamically scale proportionally with the scaled bounding box, ensuring models never clip into terrain or float excessively high.

### User Story 3 (US3): Procedural "Garden Museum" Terrain Mesh (Smooth, Playable Landscapes)
*As a player walking through the map in-game or in WoWViewer, I want continuous, smooth, walkable terrain with gentle garden slopes, terraced plazas, and zero jarring step-quad cliffs.*

- **AC3.1**: Sculpt terrain heightmaps (`MCVT`) using harmonic Perlin/Simplex noise with smooth gradient limits (maximum slope $\le 25^\circ$ for 100% walkability).
- **AC3.2**: Exhibit pedestals sculpted as smooth low-angle stepped terraces or raised circular/octagonal podiums with natural radial falloff ramps (`SmoothStep` blending into surrounding terrain).
- **AC3.3**: Continuous arterial pedestrian promenades and garden paths connecting all courtyards, ensuring seamless navigation without collision barriers.

### User Story 4 (US4): Organic MCAL / MCLY Texture Painting & Garden Aesthetic
*As a visual observer, I want rich, artistic multi-layer texture painting featuring garden grass, cobblestone paths, decorative checkerboard borders, and clean exhibit floors.*

- **AC4.1**: Multi-layer texture blending:
  - Layer 0 (Base): Lush garden lawn/grass.
  - Layer 1 (Walkways): Paved cobblestone / gravel garden pathways.
  - Layer 2 (Pedestal Perimeter): Stylized checkerboard / boundary guide ring around each exhibit.
  - Layer 3 (Exhibit Center): Clean, low-contrast neutral marble/ground devoid of distracting texture noise.
- **AC4.2**: Smooth 64x64 alpha splatting (`MCAL`) using radial falloffs, anti-aliased edge blends, and distance-to-path splatting.
- **AC4.3**: Texture Palette Matcher: Automatically resolve appropriate thematic textures (e.g. grass, stone, marble, pavers) from the client's archive library based on client version.

### User Story 5 (US5): Generic Map Generator Surface & WoWViewer Editor Integration
*As a world builder and modder, I want a modular map generation framework in `WowViewer.Core.Editor` and a UI surface in WoWViewer to generate procedural custom maps for any targeted WoW version.*

- **AC5.1**: Define clean engine interfaces (`IGenerativeMapSurface`, `ProceduralTerrainGenerator`, `ProceduralTexturePainter`, `ProceduralPlacementEngine`) in `WowViewer.Core.Editor` / `WowViewer.Core.IO.Maps`.
- **AC5.2**: Full cross-era compilation and translation: generate native LK 3.3.5 ADTs, Classic 1.12.1 ADTs, or Alpha 0.5.3 WDT/ADT/WDL files via existing core format converters.
- **AC5.3**: Add a "Procedural Map Generator" UI panel in the WoWViewer Editor allowing interactive parameter tuning (density, theme, scale, noise, client version) and immediate preview.

### User Story 6 (US6): Semantic Folder & Filename Asset Categorizer with Contextual Dungeon Disambiguation
*As a world generator, I want deep semantic parsing of asset directory paths and filenames so that objects are intelligently categorized by archetype (weapons, armor, small doodads, large monuments, critters, characters) with contextual disambiguation (such as Scarlet Monastery `sm_` vs size prefix `_sm`).*

- **AC6.1**: Parse directory paths and filename tokens:
  - Items / Weapons / Armor: `item/objectcomponents/...` (helm, shoulder, shield, 1h, 2h, bow, staff, wand, buckle, boots) -> micro cells, $3.5\times$–$5.0\times$ scaling.
  - Spells & Particles: `spells/...` -> micro cells, $4.0\times$ scaling.
  - Doodads & Props: `world/generic/doodads/...`, `doodads/...`.
- **AC6.2**: Context-aware size token extraction:
  - Discriminate `_sm`, `_small`, `_tiny`, `_micro` vs `_med`, `_medium` vs `_lg`, `_large`, `_huge`, `_giant`.
  - Disambiguate dungeon namespace prefixes: prefix `sm_` under dungeon paths (e.g. `world/wmo/dungeon/sm_...`, `doodads/sm_...`) resolved as *Scarlet Monastery* set rather than small size indicator.
- **AC6.3**: Assign semantic archetype metadata to every placed record to drive cell density, scale factor, and thematic pavilion grouping (e.g., Armory, Menagerie, Sculpture Garden, Grand Pavilion).

---

## 3. Architecture & Technical Design

### 3.1 Component Layering
```text
[WoWViewer Editor UI]  <-->  [Procedural Map Generator Panel]
         |                                |
         v                                v
[WowViewer.Core.Editor]  -->  [IGenerativeMapSurface / ProceduralPipeline]
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
[ProceduralTerrainGenerator]   [ProceduralTexturePainter]      [ProceduralPlacementEngine]
(Harmonic Noise, Terracing,    (Multi-layer Alpha Splatting,   (Adaptive Density Packing,
 Ramps, Walkable Gradients)     Garden Paths, Clean Plazas)     M2 2x-5x Scale, Bounding)
        |                                 |                                 |
        +---------------------------------+---------------------------------+
                                          |
                                          v
                              [Format Writers & Converters]
                 (LkAdtWriter, AlphaWdtWriter, WdlWriter, RosettaDbcGenerator)
```

### 3.2 Key Constants & Mathematics
- **Tile Dimensions**: $533.33333\text{m} \times 533.33333\text{m}$, 16 chunks ($33.33333\text{m}$ each).
- **Height Grid per Chunk**: $9 \times 9$ outer $+ 8 \times 8$ inner vertices ($145$ vertices/chunk, $2080$ vertices/tile).
- **Alpha Splatting Resolution**: $64 \times 64$ per chunk ($1024 \times 1024$ virtual resolution per tile).
- **Scale Encoding**:
  - LK / Classic `MDDF.Scale`: uint16, where $1024 = 1.0\times$, $2560 = 2.5\times$, $5120 = 5.0\times$.
  - Alpha `MDDF.Scale`: uint32 or float equivalent depending on client profile.
