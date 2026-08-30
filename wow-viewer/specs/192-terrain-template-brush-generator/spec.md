# Spec 192: Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator

## 1. Overview & Vision

Existing WoW map editors (Noggit, Taliis, Machinima Studio, standard tools) force world builders to start from a blank, flat, untextured canvas, requiring tedious manual vertex pulling and layer-by-layer alpha painting. However, Blizzard's original world building heavily relied on reusable **fractal brush patterns, terrain pastes, and template motifs** (as evidenced by archaeological analysis of alpha masks and terrain height signatures across 0.5.3–3.3.5 ADTs).

Spec 192 delivers a first-of-its-kind **Terrain Template Brush & Paste Library and Procedural Map Generator**, integrated directly into the `WowViewer` Editor plugin platform. It extracts recurring terrain patterns (height contours + multi-layer MCLY/MCAL textures) from real game data and provides:
1. **A Reusable Terrain Paste & Brush Catalog**: Categorized library of terrain stamps (roads, hills, plateaus, courtyards, clearings, ridges, riverbanks).
2. **Interactive In-Viewer Brush Stamping**: An ImGui editor plugin to preview, rotate, scale, and stamp terrain pastes directly onto loaded ADT chunks with seamless boundary blending.
3. **Templated Procedural Map Generator**: A high-level generation engine that constructs cohesive, fully textured, walkable maps (such as lush garden museums, temperate valleys, or rocky plateaus) using composed pastes and connected path networks.
4. **Guaranteed Walkability & Legibility**: Zero collision-trapping quad bevels, maximum slope constraints ($\le 25^\circ$), and clean multi-tileset alpha splats.

---

## 2. User Stories

- **US1 (Terrain Paste Data Model & Catalog)**: As a world builder or tool developer, I want a structured, serializable data model (`TerrainBrushPaste`) representing multi-layer terrain motifs (height deltas, MCAL/MCLY texture splats, surface normals, tags, dimensions) so that terrain features can be saved, shared, and manipulated independently of full ADT files.
- **US2 (Authentic ADT Paste Extraction & Built-In Library)**: As a researcher or developer, I want an extractor that harvests reusable terrain pastes from authentic game ADTs (MCVT heights + MCLY/MCAL splats) and packages a rich built-in catalog of standard WoW terrain building blocks (paths, hills, plazas, ridges).
- **US3 (Interactive Editor Brush Stamping Plugin)**: As an editor user in `WowViewer`, I want a dedicated "Terrain Brush & Pastes" editor plugin panel where I can browse pastes, adjust stamp parameters (radius, height scale, rotation, blend feathering), and stamp features onto the live 3D terrain canvas with full Undo/Redo support.
- **US4 (Templated Procedural Map Synthesis Engine)**: As a map creator or calibration engineer, I want a procedural map generator that synthesizes complete multi-tile maps from high-level templates (Biome, Path Network, Exhibit Courtyards, Terrain Relief) using the paste library, producing authentic 3–4 layer ADTs.
- **US5 (Seamless Boundary Blending & Texture Budget Enforcement)**: As a terrain artist, I want stamp operations to blend height and alpha splats smoothly with existing terrain without exceeding the engine's 4-layer-per-chunk texture limit.

---

## 3. Requirements & Acceptance Criteria

### Functional Requirements (FR)
- **FR-001**: Define `TerrainBrushPaste` containing 2D relative height array ($N \times N$), multi-layer alpha splat masks ($N \times N \times L$), texture identifiers, canonical bounding dimensions, classification tags, and metadata.
- **FR-002**: Implement `TerrainBrushLibrary` supporting serialization (JSON/binary), tag indexing (`Road`, `Hill`, `Plaza`, `Ridge`, `Depression`), search, and filtering.
- **FR-003**: Provide a built-in curated library of at least 20 foundational terrain motifs across multiple biomes (Garden, Temperate, Forest, Cobblestone City, Arid).
- **FR-004**: Implement `AdtPasteExtractor` to extract bounded paste regions from loaded `LkMcnkData` / `AlphaMcnk` structures.
- **FR-005**: Implement `TerrainStampOperation` supporting:
  - Height delta application with smooth cosine/SmoothStep edge feathering.
  - Multi-layer MCLY texture layer allocation (re-mapping layer indices, merging shared textures, pruning negligible weights, capping at 4 layers).
  - Normal recalculation across modified chunk vertices.
- **FR-006**: Create `TerrainTemplateEditorPlugin` implementing `IEditorPlugin` with:
  - Visual paste browser with thumbnail previews and metadata.
  - Interactive stamping parameters (Scale, Rotation, Height Multiplier, Feather Radius).
  - In-viewer "New Map from Template" wizard.
  - Integration with `EditorSession` for non-destructive Undo/Redo.
- **FR-007**: Provide CLI command `terrain-generate-templated` in `WowViewer.Tool.Inspect` with options `--template`, `--client-root`, `--output`, `--theme`, `--rows`, `--cols`, and `--format`.

### Non-Functional & Safety Requirements (NFR)
- **NFR-001 (Walkability)**: Procedural walkways and exhibit plazas must maintain flat or smoothly graded slopes ($\le 25^\circ$) with zero sharp quad bevels that could trap player collision capsules.
- **NFR-002 (Texture Constraint)**: Every generated or edited chunk must strictly obey the $\le 4$ texture layer hardware limit.
- **NFR-003 (Read-Only Client Safety)**: Client source directories (`H:\CLIENTS`) are strictly read-only; all outputs write to specified target output paths.
- **NFR-004 (Core Separation)**: Core algorithms reside in `WowViewer.Core.IO` and `WowViewer.Core.Editor`; UI rendering is isolated to `WoWViewer` (viewer shell).
