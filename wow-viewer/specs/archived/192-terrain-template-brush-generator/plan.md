# Technical Plan: Spec 192 — Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator

## 1. Architecture Overview

Spec 192 introduces a complete terrain authoring and procedural synthesis pipeline layered cleanly across Core, Editor, and Viewer subsystems:

```text
[ Authentic ADT / Data-Harvester Mining ]
                  │
                  ▼
   [ AdtPasteExtractor / Curated Defaults ]
                  │
                  ▼
     [ TerrainBrushLibrary (JSON/Zarr) ]
        ├── TerrainBrushPaste (Heights, Alphas, MCLY Layers, Normals, Tags)
        └── Tagged Categories (Road, Hill, Plaza, Ridge, Depression, Garden)
                  │
                  ├──────────────────────────────────────────────┐
                  ▼                                              ▼
    [ TerrainStampOperation ]                     [ TemplatedTerrainGenerator ]
    - Multi-Layer 4-Texture Allocator             - Procedural Road / Path Splines
    - SmoothStep Edge Feathering                  - Biome Palette Synthesizer
    - Height Delta Blending                       - Walkability & Slope Enforcer
    - Normal Recalculation                        - ADT (Alpha / LK) Emission
                  │                                              │
                  ▼                                              ▼
   [ TerrainTemplateEditorPlugin ]               [ CLI Tool: terrain-generate-templated ]
   - Interactive ImGui Brush Stamp Tool          - Batch generation from templates
   - Paste Browser & Previews                     - Multi-era output support
   - In-Viewer New Map Wizard
   - Undo/Redo via EditorSession
```

---

## 2. File Organization & Boundaries

### Core & IO Layer (`WowViewer.Core.IO/Terrain/`):
- `TerrainBrushPaste.cs`: Data model representing a self-contained terrain motif with 2D relative heights, multi-layer alpha splats, textures, normals, and tags.
- `TerrainBrushLibrary.cs`: In-memory and serializable catalog with tagging, searching, and JSON serialization.
- `AdtPasteExtractor.cs`: Extracts sub-regions from loaded ADTs (Alpha / LK MCNK chunks) into `TerrainBrushPaste` objects.
- `CuratedTerrainBrushLibrary.cs`: Built-in stock catalog of foundational terrain motifs (cobblestone roads, grass knolls, stone plazas, riverbeds, mountain ridges).
- `TerrainLayerAllocator.cs`: Merges and allocates texture layers across overlapping stamps while enforcing the strict 4-layer-per-chunk engine limit.
- `TemplatedTerrainGenerator.cs`: Procedural generator creating entire maps from high-level templates using the paste catalog.
- `TerrainMapTemplate.cs`: High-level configuration defining biomes, road network topologies, courtyard distributions, and height variations.

### Editor Layer (`WowViewer.Core.Editor/`):
- `Operations/TerrainStampOperation.cs`: `EditorOperation` implementation applying stamps to terrain with full undo/redo state capture.
- `Plugins/TerrainTemplateEditorPlugin.cs`: Implements `IEditorPlugin` providing the interactive UI surface.

### Viewer Shell (`WoWViewer/`):
- `ViewerApp_Editor.cs`: Hosts the `TerrainTemplateEditorPlugin` UI in the Editor dock panel, including brush selection, viewport stamping cursor, and template generation wizard.

### Tool CLI (`WowViewer.Tool.Inspect/`):
- `Program.cs`: Adds `terrain-generate-templated` command.

### Tests (`WowViewer.Core.Tests/`):
- `TerrainBrushPasteTests.cs`
- `TerrainLayerAllocatorTests.cs`
- `AdtPasteExtractorTests.cs`
- `TemplatedTerrainGeneratorTests.cs`
- `TerrainStampOperationTests.cs`

---

## 3. Detailed Component Designs

### 3.1 TerrainBrushPaste & Multi-Layer Alpha Representation
A `TerrainBrushPaste` is dimensioned by vertex resolution (e.g. $17 \times 17$ for a single chunk or $33 \times 33$ for $2 \times 2$ chunks):
```csharp
public sealed class TerrainBrushPaste
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public string Category { get; init; } = "General"; // Road, Hill, Plaza, Ridge, Depression, Garden
    public string[] Tags { get; init; } = [];
    public float WidthMeters { get; init; }
    public float LengthMeters { get; init; }
    public int ResolutionX { get; init; }
    public int ResolutionY { get; init; }
    public float[] HeightDeltas { get; init; } = []; // Relative elevation deltas
    public List<TerrainPasteLayer> Layers { get; init; } = []; // Up to 4 texture layers
    public float MaxSlopeDegrees { get; init; }
}

public sealed class TerrainPasteLayer
{
    public string TexturePath { get; init; } = string.Empty;
    public byte[] AlphaMask { get; init; } = []; // 64x64 or 256x256 normalized alpha
    public uint EffectId { get; init; }
}
```

### 3.2 4-Layer Texture Budget Allocator (`TerrainLayerAllocator`)
When stamping a paste onto an existing chunk:
1. Identify existing chunk texture layers ($E_0..E_3$) and stamping layers ($S_0..S_k$).
2. Union unique texture paths.
3. If union count $\le 4$, assign directly.
4. If union count $> 4$, prioritize by total coverage area / max alpha energy; least significant layers ($< 5\%$ contribution) are blended into the base layer or dropped.
5. Reconstruct chunk `MCLY` and `MCAL` splats with normalized $0..255$ weights.

### 3.3 Templated Procedural Map Generator (`TemplatedTerrainGenerator`)
Synthesizes a cohesive playable zone:
1. **Base Biome Layer**: Fills all chunks with primary biome ground (e.g., lush `elwynngrass.blp`).
2. **Road / Path Network**: Places connected spline/corridor paths with cobblestone or dirt pastes (`stormwindcobble.blp`).
3. **Plaza & Exhibit Courtyards**: Places flat, polished stone/marble exhibit pads (`whitemarble.blp`) with zero-bevel transitions.
4. **Natural Relief Features**: Scatters gentle rolling hills, perimeter ridges, and clearing depressions using matching biome pastes.
5. **Walkability Gate**: Validates all generated paths and courtyards against the $\le 25^\circ$ slope limit.

---

## 4. Phase Roadmap

- **Phase 1**: Core Data Models, Layer Allocator, and Curated Paste Library (`US1`, `US2`)
- **Phase 2**: ADT Extraction & Serialization Engine (`US2`)
- **Phase 3**: Terrain Stamp Operation with Seamless Feathering & Normal Computation (`US3`, `US5`)
- **Phase 4**: Templated Procedural Map Generator (`US4`)
- **Phase 5**: Editor Plugin UI in `ViewerApp_Editor.cs` (`US3`)
- **Phase 6**: CLI Command & Comprehensive Unit Test Suite (`US1`–`US5`)
