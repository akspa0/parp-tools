# Technical Design & Architecture Plan: Procedural Garden Museum Map Generator

**Spec ID**: `191`  
**Feature**: Procedural Garden Museum Map Generator & Dense Calibration Corpus  
**Status**: Draft  
**Created**: 2026-08-29  

---

## 1. Architectural Strategy

We will build a high-performance, modular procedural world generation pipeline in `WowViewer.Core.Editor.Procedural` and `WowViewer.Core.IO.Maps`.

### Key Design Tenets:
1. **Core Library First**: All mathematical terrain sculpting, alpha splatting, layout packing, and model scaling logic lives in `WowViewer.Core.Editor` and `WowViewer.Core.IO`, completely independent of any UI framework.
2. **Deterministic & Seeded**: Every map generation run with identical inputs and seed produces bit-exact identical terrain heights, textures, and placements.
3. **Playable Geometry (Navmesh Grade)**: Heightmap generation applies a maximum gradient constraint ($|\nabla z| \le \tan(25^\circ) \approx 0.466$) and `SmoothStep` blending around pedestals so the player character can smoothly walk anywhere without collision catching or jumping required.
4. **Multi-Era Format Agnostic**: Generates canonical intermediate structures (`ProceduralTileData`, `ProceduralMapPlan`) that feed cleanly into WotLK `LkAdtWriter`, Classic `AdtWriter`, or Alpha `AlphaWdtWriter` / `RosettaDbcGenerator`.

---

## 2. Component Design & New Types

### 2.1 `SemanticAssetClassifier` (`WowViewer.Core.Editor.Procedural`)
- **Semantic Path & Filename Parser**:
  - Parses full relative path and filename to extract structural tokens and archetypes (`Weapon`, `Armor`, `Item`, `SpellEffect`, `SmallCritter`, `Humanoid`, `Mount`, `SmallDoodad`, `MediumDoodad`, `LargeDoodad`, `Monument`, `Structure`).
  - Distinguishes size tokens (`_sm`, `_small`, `_tiny`, `_micro` vs `_med`, `_medium` vs `_lg`, `_large`, `_huge`, `_giant`).
  - Disambiguates contextual dungeon set prefixes (e.g. `sm_` under dungeon folders indicates *Scarlet Monastery*, not a small prop).
- **Archetype-Driven Scale & Density Rules**:
  - Assigns fine-tuned default scale factors: $4.0\times$ for daggers/wands/rings, $3.0\times$ for swords/shields/helmets, $2.5\times$ for small doodads/props, $2.0\times$ for humanoids/creatures, $1.0\times$ for giant bosses/WMOs.
  - Groups related archetypes into thematic pavilions (e.g. Armory Pavilion, Menagerie, Grand Hall, Sculpture Court).

### 2.2 `ProceduralPlacementEngine` (`WowViewer.Core.Editor.Procedural`)
- **Adaptive Density Bucketer**:
  - `Micro` ($16.66\text{m} \times 16.66\text{m}$ / 64 cells per chunk): Items, weapons, jewelry, armor, spell effects ($r \le 1.5\text{m}$).
  - `Small` ($33.33\text{m} \times 33.33\text{m}$ / 1 cell per chunk): Small props, critters, doodads ($1.5\text{m} < r \le 4.0\text{m}$).
  - `Medium` ($66.66\text{m} \times 66.66\text{m}$ / 4 chunks per cell): Standard creatures, mounts, humanoids, trees ($4.0\text{m} < r \le 15.0\text{m}$).
  - `Large` ($133.33\text{m} \times 133.33\text{m}$ / 16 chunks per cell): Large monsters, dragons, small buildings ($15.0\text{m} < r \le 35.0\text{m}$).
  - `Grand` ($266.66\text{m}$ / 64 chunks per cell): Massive WMO dungeons, keeps, capital city assets ($r > 35.0\text{m}$).
- **M2 Scale Resolution**:
  - Automatically calculates category scale multiplier and writes scale into placement metadata and format-specific `MDDF` structures.

### 2.2 `ProceduralTerrainSculptor` (`WowViewer.Core.Editor.Procedural`)
- **Harmonic Landscape Generator**:
  - Evaluates multi-octave Simplex/Perlin noise $H(x, y) = \sum_{k=0}^{N-1} A_k \text{Noise}(f_k x, f_k y)$ over the $9 \times 9 + 8 \times 8$ vertex grid ($145$ vertices per chunk, $2080$ vertices per tile).
  - Smooth slope clamping ensures no vertical cliffs.
- **Pedestal Podium Integration**:
  - For each placed exhibit, defines a circular/octagonal podium with radius $R_{\text{podium}}$ and height $Z_{\text{podium}}$.
  - Applies smooth blending function:
    $$Z(x, y) = \text{Lerp}(Z_{\text{landscape}}(x, y), Z_{\text{podium}}, \text{SmoothStep}(R_{\text{outer}}, R_{\text{inner}}, \text{dist}(x, y)))$$
  - Center of podium ($d \le R_{\text{inner}}$) is perfectly flat ($Z = Z_{\text{podium}}$).
  - Transition zone ($R_{\text{inner}} < d \le R_{\text{outer}}$) forms a gentle, natural walkable ramp.

### 2.3 `ProceduralTexturePainter` (`WowViewer.Core.Editor.Procedural`)
- **Layer Stack (4 Layers per Chunk)**:
  - **Layer 0 (`Ground`)**: Lush Elwynn / Westfall turf base.
  - **Layer 1 (`Walkways`)**: Cobblestone / flagstone path connecting exhibit corridors.
  - **Layer 2 (`Border`)**: Stylized decorative checkers / trim ring along $R_{\text{inner}} \le d \le R_{\text{outer}}$.
  - **Layer 3 (`Plaza`)**: Clean neutral marble / smoothed stone in the center exhibit zone ($d < R_{\text{inner}}$).
- **Anti-Aliased 64x64 Alpha Splatting**:
  - Generates 64x64 byte alpha maps for each chunk using sub-pixel analytical distance sampling.
  - Devoid of distracting noise in the center exhibit zone, ensuring the placed model's silhouette and details are clear.

### 2.4 `IGenerativeMapSurface` & CLI / Editor Integration
- Standardized generative map engine pipeline.
- CLI flags in `rosetta-generate`:
  - `--density <compact|balanced|spacious>`
  - `--m2-scale <float>`
  - `--theme <garden|marble|desert|autumn>`
  - `--noise-roughness <float>`
  - `--pedestal-style <terrace|podium|sunken>`
- WoWViewer Editor integration:
  - Procedural Map Generation Dialog in Editor sidebar / menu.

---

## 3. Implementation Phases

- **Phase 1 (US1 & US2)**: Adaptive Multi-Tier Density Packing & M2 Model Scaling ($2\times$–$5\times$).
- **Phase 2 (US3)**: Procedural Garden Terrain Sculptor (Harmonic Noise, Walkable Ramps, Smooth Podiums).
- **Phase 3 (US4)**: Multi-Layer Organic Texture Painter (Garden Turf, Cobblestone Paths, Checker Ring, Clean Center Plaza).
- **Phase 4 (US5)**: Generic Map Generator Surface & CLI / Editor Tooling Integration.
- **Phase 5**: Automated Unit Testing & Multi-Era Validation.
