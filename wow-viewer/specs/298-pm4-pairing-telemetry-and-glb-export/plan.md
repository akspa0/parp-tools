# Plan 298 — PM4 Authentic Pairing Telemetry, Markdown Elimination of CK24 & Multi-Object GLB Export

## Architecture & Technical Design

### Track 1: Authentic Pairing Telemetry (FR-001)
1. **Model Enrichment**:
   - In `Pm4DoodadDuplicationService.cs`, update `TaughtPairing` record definition to include:
     ```csharp
     internal sealed record TaughtPairing(
         uint Ck24,
         byte DominantGroupKey,
         string ModelName,
         string ModelPath,
         string AssetKind,
         float SolvedScale,
         float YawDegrees,
         float Length,
         float Width,
         float Height,
         DateTime TaughtUtc,
         Vector3 LocalBoundsMin = default,
         Vector3 LocalBoundsMax = default,
         float YawOffset = 0f,
         float[]? VerticalSliceRadii = null,
         Vector3 InertiaTensor = default,
         Vector3 NormalDistribution = default,
         // Spec 298 Authentic Telemetry
         Vector3 PlacementPosition = default,
         Vector3 PlacementRotation = default,
         float PlacementScale = 1.0f,
         uint UniqueId = 0,
         int TileX = 0,
         int TileY = 0,
         int ObjectPartId = 0,
         uint LinkGroupObjectId = 0,
         Vector3 HullCentroid = default,
         Vector3 HullBoundsMin = default,
         Vector3 HullBoundsMax = default);
     ```
   - In `TeachPairing`: Populate all new telemetry fields from `inst` (`PlacementPosition`, `PlacementRotation`, `PlacementScale`, `UniqueId`) and `pm4Obj` (`TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId`, `HullCentroid`, `BoundsMin`, `BoundsMax`).
   - In `Pm4PairingProjectStore`: Ensure serialization and deserialization seamlessly round-trip all new fields while preserving default fallbacks for older files.

### Track 2: Purge CK24 from Markdown Summaries (FR-002)
1. **Batch Analysis Markdown Summary**:
   - In `Pm4ShapeMatchBatchService.cs`:
     - Update `FormatMarkdownSummary`:
       Replace table header:
       `| Tile | Object ID | Position (X, Y, Z) | Dimensions (LxWxH) | Matched Model | Unique ID | Classification | Confidence | Provenance | Conflict |`
       Format each row with authentic physical metrics instead of `CK24`.
2. **Overlay and LLM Bundle Markdown Reports**:
   - In `Pm4WorkbenchService.Collections.cs` (`BuildPm4LlmBundleMarkdown`):
     - Replace `ck24=0x...` with `Placement Z ~= {z:F1}yd` and `Obj #{partId}`.
   - In `Pm4WorkbenchService.SceneGraph.cs` (`ExportPm4OverlayReport`):
     - Replace `CK24` column with `Placement Z (yd)`.

### Track 3: PM4 GLB Exporter Engine (FR-003, FR-004)
1. **New Exporter Class `Pm4GlbExporter`** in `wow-viewer/src/viewer/WoWViewer/Export/Pm4GlbExporter.cs`:
   - Uses `SharpGLTF.Geometry`, `SharpGLTF.Materials`, `SharpGLTF.Scenes`.
   - Method `BuildPm4ObjectMesh(Pm4OverlayObject obj, string nodeName)`:
     - Extracts all triangles from `obj.Triangles`, transforms via `obj.BaseTransform`, and converts Z-up to Y-up (`v_yup = (x, z, -y)`).
     - Assigns a color-coded PBR collision material based on `obj.DominantGroupKey`:
       - Doodad (0x03): Cyan (`#00E5FF`, opacity 0.85)
       - Lowest floor / interior floor (0x10, 0x11, 0x12): Emerald/Green (`#00E676`)
       - Upper floor / portals (0x13): Purple/Violet (`#D500F9`)
       - Angled surface / ramp (0x14): Orange (`#FF9100`)
       - Flat base (0x15): Amber (`#FFD600`)
       - Default: Sky blue (`#2979FF`)
   - Method `ExportTilePm4Glb(TerrainManager terrain, IDataSource? dataSource, Md5TranslateIndex? md5Index, int tileX, int tileY, IReadOnlyList<Pm4OverlayObject> objects, string outputPath, bool includeTerrain = true)`:
     - If `includeTerrain` and chunks exist, invokes `MapGlbExporter.BuildTerrainTileMesh` to add the textured/MCCV terrain base.
     - Adds each PM4 object as its own named node in the scene builder: `node.Name = $"[{tileX}_{tileY}]_Obj_{obj.ObjectPartId}_{(obj.LinkGroupObjectId != 0 ? $"Lnk0x{obj.LinkGroupObjectId:X}" : "Hull")}"`.
     - Converts to GLB and writes to disk.
   - Method `ExportAllLoadedPm4Glb(...)`:
     - Exports all loaded PM4 tiles and objects across the active map with combined terrain.
2. **UI Integration**:
   - In `ExportWorkbenchService.cs`:
     - Add `[ Export Active Tile PM4 to GLB... ]`
     - Add `[ Export All Loaded PM4 to GLB... ]`
   - In `Pm4WorkbenchService.Exports.cs`:
     - Connect GLB export actions to file dialog / export runners.
   - Register new controls in `ui-controls.json` and regenerate `UI-REFERENCE.md`.

### Track 4: Verification & Governance (FR-005)
1. Unit tests for `TaughtPairing` telemetry serialization and `FormatMarkdownSummary` output.
2. Full solution build with 0 errors.
3. Verification of `Verify-UiReference.ps1`.
4. Record receipt in `evidence/receipt-spec298.md`.
