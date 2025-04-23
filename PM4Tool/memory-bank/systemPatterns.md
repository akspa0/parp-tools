# System Patterns

## Implemented Patterns

1. Legacy Support Pattern
   ```csharp
   public interface ILegacyChunk : IIFFChunk
   {
       int Version { get; }
       bool CanConvertToModern();
       IIFFChunk ConvertToModern();
   }
   ```

2. Version Conversion Pattern
   ```csharp
   public interface IVersionConverter<T> where T : IIFFChunk
   {
       bool CanConvert(int fromVersion, int toVersion);
       T Convert(ILegacyChunk source);
   }
   ```

3. Validation Pattern
   ```csharp
   public class ChunkValidator
   {
       public IEnumerable<ValidationError> ValidateChunk(IIFFChunk chunk);
   }
   ```

4. Chunked File Loading Pattern
   - Specific file format classes (e.g., `PM4File`, `PD4File`, `Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero`, `Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain`) inherit from `Warcraft.NET.Files.ChunkedFile` (directly or via intermediate base classes).
   - These specific classes represent individual physical files (like `.pm4`, `.pd4`, `.adt`, `_obj0.adt`).
   - Derived classes define properties for expected chunks within that *specific file* (e.g., `TerrainObjectZero` has `ModelPlacementInfo` for `MDDF`).
   - The base `ChunkedFile` constructor uses reflection to find these properties, reads the chunk header (signature, size), and attempts to load corresponding chunks by passing a `BinaryReader` (positioned *after* the header) to the chunk's `Load` method.
   - Chunk properties can be marked `[ChunkOptional]` if their presence is not guaranteed.
   - Chunk definitions (`IIFFChunk` implementations) handle their own internal data parsing within their `Load(BinaryReader br)` method.
   - **Current Issue:** The `MDSFChunk.Load` implementation currently assumes the passed `BinaryReader`'s `BaseStream.Length` represents the chunk size, which is incorrect when used by the base `ChunkedFile` loader. This causes parsing failures after the first file. The loader expects `Load` to read the size defined in the chunk header it already processed.
   - Handling composite file formats like ADT (which consists of multiple physical files) requires loading each relevant split file into its corresponding `Warcraft.NET` class (e.g., `BfA.Terrain` for base, `TerrainObjectZero` for _obj0) and then combining the data logically in services like `AdtService`.
   - Analysis logic (e.g., in `AnalysisTool`) is PM4-centric, iterating through PM4s and looking for corresponding `_obj0.adt` files.

## Architecture

1. Core Framework
   - Legacy chunk interface system
   - Version conversion infrastructure
   - Extension method support
   - Validation framework

2. Documentation System
   - Markdown parsing
   - Specification modeling
   - Validation rules
   - Relationship tracking

3. Validation System
   - Size validation
   - Version validation
   - Field validation framework
   - Relationship validation framework

## Design Patterns

1. Interface Segregation
   - ILegacyChunk for legacy support
   - IVersionConverter for conversions
   - IIFFChunk base compatibility

2. Template Method
   - ChunkConverterBase for conversion logic
   - Abstract conversion implementation
   - Version-specific handling

3. Strategy Pattern
   - Validation rule application
   - Format detection
   - Version conversion

4. Factory Pattern
   - Chunk creation
   - Converter instantiation
   - Validator creation

## Code Organization

1. Core Library (WoWToolbox.Core)
   ```
   /Legacy
     /Interfaces
       - ILegacyChunk.cs
       - IVersionConverter.cs
     /Converters
       - ChunkConverterBase.cs
   /Extensions
     - ChunkedFileExtensions.cs
   /Navigation
     /PM4
       - PM4File.cs
       /Chunks
         - MVERChunk.cs
         - MSHDChunk.cs
         - // ... (other PM4 chunk classes)
     /PD4
       - PD4File.cs
       /Chunks
         - MCRCChunk.cs
     /ADT
       - Placement.cs
       - AdtService.cs
   ```

2. Validation Library (WoWToolbox.Validation)
   ```
   /Chunkvault
     /Models
       - ChunkSpecification.cs
     /Parsers
       - MarkdownSpecParser.cs
     /Validators
       - ChunkValidator.cs
   ```

## Implementation Guidelines

1. Legacy Support
   - Extend ILegacyChunk for each format
   - Implement version-specific converters
   - Use ChunkConverterBase template

2. Validation Rules
   - Define in chunkvault markdown
   - Parse using MarkdownSpecParser
   - Apply using ChunkValidator

3. Extension Methods
   - Chunk loading helpers
   - Conversion utilities
   - Validation helpers

## Testing Strategy

1. Unit Tests
   - Interface implementations
   - Converter logic
   - Validation rules

2. Integration Tests
   - Format conversion
   - Documentation parsing
   - Validation pipeline

3. Documentation Tests
   - Specification compliance
   - Markdown parsing
   - Relationship validation

4. **Resource Management in Test Automation (NEW, 2024-07-21)**
   - All file and stream resources in tests must be properly disposed to prevent process hangs after test completion.
   - Recent experience: A test for mesh+MSCN boundary output wrote all files successfully but the process hung, requiring manual cancellation. Emphasizes the need for robust cleanup and test method completion.

## Emerging Patterns / Discoveries
*   **MSLK Doodad Placement (Confirmed):** **NEW/Significant:** Visualization and analysis confirm that `MSLK` entries (specifically those with `MspiFirstIndex == -1`, previously termed "nodes") represent **Doodad placements** (M2/MDX models). The `Unknown_*` fields (`Unk00`, `Unk01`, `Unk04`, `Unk12`) likely encode the specific model ID (potentially linking to `MDBH`), rotation, scale, and other properties. `Unknown_0x10` provides the vertex index for the placement anchor point via `MSVI`->`MSVT`.
*   **MSCN Chunk (Clarified):**
    *   The MSCN chunk is now confirmed to be an array of Vector3 (float) values, not int32/C3Vectori.
    *   All MSCN vectors have been exported as OBJ points for visualization.
    *   The semantic purpose of MSCN is now confirmed by visual inspection: it represents the exterior (boundary) vertices for each object in PM4 files.
    *   Plan: Update the PM4 exporter to annotate/export MSCN points as "exterior vertices" for further validation and analysis.
    *   Previous hypotheses about MSCN being normals or int32 vectors are incorrect for the current data.
*   **MSLK Hierarchical Structure (PM4 Confirmed, PD4 Different - Tied to File Scope):**
    *   **Context:** PM4 files represent multi-object map tiles, while the tested PD4 files represent single WMO objects.
    *   **PM4 (Multi-Object):** Analysis using `WoWToolbox.AnalysisTool` on PM4 log data confirmed `Unknown_0x04` acts as a group/object identifier. It creates **"Mixed Groups"** linking Doodad placement entries (`MspiFirstIndex == -1`) directly to their corresponding geometry path entries (`MspiFirstIndex >= 0`) for a specific object within the collection.
    *   **PD4 (Single Object):** Analysis of tested PD4 files (`6or_garrison...`) showed `Unknown_0x04` still acts as a group ID, but *not* to link Doodad placements and geometry directly. It creates separate **"Node Only"** (Doodad placements) and **"Geometry Only"** groups.
*   **ADT/PM4 Correlation:** PM4 data (`m_destructible_building_index` in `MDOS` via `MDSF`) can be linked to ADT object placements via **Unique IDs** (`UniqueID` field in `MDDFEntry`/`MODFEntry`). This is key for understanding PM4 context, including potentially linking `MSLK` Doodad groups (`Unk04`) to ADT placements.
*   **Surface Definition (Hypothesis -> Confirmed Links & Handling Unlinked):** `MSUR` defines surface geometry using indices from `MSVI` (which point to `MSVT` vertices). `MDSF` acts as a mapping layer, linking `MSUR` surfaces (`msur_index`) to `MDOS` destructible object states (`mdos_index`). **Logic now also includes `MSUR` faces without an `MDSF` link, assuming they represent the default state (0).**

## Chunk Correlations (Investigated & Updated)

Based on analysis of chunk structures, codebase searches, and recent discoveries:

*   **Direct Implemented/Confirmed Links:**
    *   `MSLK` -> `MSPI` (via `MspiFirstIndex`): Used for defining geometry paths/points.
    *   `MSUR` -> `MSVI` (via `MsviFirstIndex`, `IndexCount`): Defines surface indices.
    *   `MDSF` -> `MSUR` (via `msur_index`): Links destruction data to specific surfaces.
    *   `MDSF` -> `MDOS` (via `mdos_index`): Links destruction data to specific destructible object state entries.
    *   **`MSPV` -> `MSLK` -> `MSPI` -> `MSVI` -> `MSVT`:** Confirmed chain for linking path nodes to world coordinates.
*   **Confirmed Doodad/Node Links:**
    *   `MSLK` -> `MSVI` (via `Unknown_0x10`): Anchors Doodad placements to vertices via MSVI->MSVT (PM4 & PD4).
*   **Potential Doodad Identification Links:**
    *   `MSLK` -> `MDBH` (via `Unk04` or other fields): **Hypothesis:** Links Doodad entries to specific filenames/model IDs in `MDBH`.
*   **External Links:**
    *   `PM4 Data` (`MDOS` via `MDSF`) <-> `ADT Object Placement` (via **UniqueID**): Allows correlating PM4 structures (e.g., `MSLK` Doodad groups) to world objects.
*   **Unknown/Unused Links:**
    *   `MSLK` -> `MSVI` (via `Unknown_0x10` for *Geometry* entries): Purpose still TBD.
*   **No Implemented Direct Links Found:**
    *   `MSLK` <-> `MSCN`: No code found directly correlating. MSCN is now confirmed to be an array of Vector3 (float) values, exported as OBJ points for visualization. Semantic purpose remains unclear.
    *   `MSLK` <-> `MSUR`: No direct code link found (but potentially linked logically via MSLK group ID / ADT UniqueID).
    *   `MPRR` -> ???: Indices within MPRR sequences are likely *not* into MPRL.
*   **Potential Semantic Links (Requires Further Research):**
    *   The `MSLK.Unk04` Group ID / ADT UniqueID might logically group related `MSUR` surfaces (via MDSF/MDOS?) or `MSCN` objects.
    *   `MSLK` Doodad properties (`Unk00`/`Unk01`/`Unk12`) might signify relationships to other chunk data (pending decoding).
    *   `MSCN` vectors might provide normal data for vertices used by `MSUR` or `MSLK`, but the indexing mechanism isn't immediately clear.
    *   `MPRR` flag value (before 0xFFFF) might indicate type or target of sequence indices.
*   **WMO Group File Handling (2024-06):**
    *   WMO files are split into a root file and multiple group files (e.g., _000.wmo, _001.wmo, etc.).
    *   **Do NOT concatenate root and group files for parsing.** The WMO format does not support monolithic concatenation; the root references group files, but does not embed their data.
    *   The correct approach is:
        1. Parse the root file for group count and metadata.
        2. Parse each group file individually for geometry.
        3. Merge the resulting meshes for analysis or export.
    *   Loader and tools have been updated to follow this pattern. Previous attempts to concatenate files led to invalid parsing and must be avoided.
    *   This pattern is now canonical for all WMO parsing and analysis in WoWToolbox.
*   **ADT/PM4 UniqueID Correlation (2024-07-21):**
    *   Implemented and tested in `AdtServiceTests.CorrelatePm4MeshesWithAdtPlacements_ByUniqueId`.
    *   For a given PM4/ADT pair, mesh groups extracted from the PM4 (by uniqueID) are matched to placements extracted from the ADT _obj0 file.
    *   The test asserts that each mesh uniqueID has a corresponding placement, confirming the mapping logic is robust and correct for the provided data.
    *   This pattern is now covered by automated tests and is a core part of the analysis pipeline.
*   **UniqueID Correlation Limitation (2024-07-21):**
    *   UniqueID-based mesh extraction and ADT correlation are ONLY possible for development_00_00.pm4. For all other PM4 files, uniqueID data and ADT correlation are NOT available—only baseline or chunk-based mesh exports are possible. This limitation is fundamental and should guide all future analysis, tests, and tooling. Do NOT attempt to generalize uniqueID grouping or ADT correlation beyond this special case.
*   **Decoupled Batch Analysis Tool Pattern (2024-07-21):**
    *   Placement extraction and matching logic is now implemented in a standalone tool (OBJ-to-WMO matcher), not in the main mesh extraction/conversion pipeline.
    *   This tool batch-compares PM4 OBJ clusters to WMO assets using rigid registration (translation + rotation, scale=1), outputs placement/match data for restoration and archival.
    *   **Rationale:** Avoids breaking or complicating the main extraction/conversion code, enables robust, automated, and archival-focused workflows, and allows for rapid iteration and validation without impacting core logic.
*   **Mesh+MSCN Boundary Output Pattern (2024-07-21):**
    - New test pattern outputs both mesh geometry and MSCN boundary points for PM4 files, writing multiple OBJ and diagnostics files.
    - All build errors were resolved, but a process hang after output highlighted the importance of resource management in test automation.

## PM4/PD4 Mesh and Node Patterns (2024-04-15)

- The most promising mapping for mesh faces is via MSUR → MSVI → MSVT.
- MSLK/MPRR are likely the semantic glue between logical/semantic structure and mesh data.
- Unk10 is confirmed as an anchor for node-to-vertex mapping, but not for faces.
- Full mesh connectivity (faces) is not present in node/object groupings alone; must analyze MSUR for surface/face data.

## PM4/PD4 Mesh Extraction Pattern (New - 2024-06)
- **Pattern:** Renderable mesh geometry is constructed using data from multiple chunks:
  - `MSVT`: Contains vertex positions (often needing transformation).
  - `MSVI`: Contains indices into the `MSVT` chunk.
  - `MSUR`: Defines surfaces (triangles) by specifying start index and count for `MSVI`.
- **Implementation:** Logic currently exists within `WoWToolbox.Tests` (`PM4FileTests.cs`/`PD4FileTests.cs`) primarily for OBJ export.
- **Refactoring Plan:** Extract this logic into a dedicated component (`Pm4MeshExtractor`) within the `WoWToolbox.MSCNExplorer` project (temporarily, outside `WoWToolbox.Core` as per user request) to enable reuse for comparison tasks.

## Key Design Patterns & Decisions

*   **Chunk-Based File Parsing:** Both WMO and PM4 files are parsed based on their chunk structure (4-byte name, size, data). `ChunkReader` utility helps manage this.
*   **Little-Endian Chunk Names:** Chunk names are read as 4 bytes and reversed (e.g., 'TVOM' -> 'MOVT') as per WoW file format specifications.
*   **Lazy Loading:** Data chunks are typically loaded on demand or when the main file object is instantiated.
*   **Separation of Concerns:**
    *   Core library (`WoWToolbox.Core`) handles file format parsing and data structures.
    *   Specific tools (`WoWToolbox.MSCNExplorer`) use the Core library to implement higher-level logic (e.g., mesh extraction, comparison).
    *   Tests (`WoWToolbox.Tests`, `WoWToolbox.MSCNExplorer.Tests`) verify the functionality of Core and tool components.
*   **OBJ Export for Visualization:** Using the simple OBJ format as a common ground for visually inspecting extracted mesh geometry from different sources (PM4, WMO).
*   **Common Mesh Data Structure (`MeshData`):** Introduced `WoWToolbox.Core.Models.MeshData` (`List<Vector3> Vertices`, `List<int> Indices`) as a standardized intermediate representation for extracted mesh geometry, replacing previous ad-hoc or format-specific structures like `WmoGroupMesh` for this purpose. This facilitates comparison.
*   **Coordinate System Transformation:** Specific transformations are applied during mesh extraction to convert from file-local coordinates to a consistent world coordinate system suitable for visualization/comparison (e.g., `MsvtToWorld_PM4` in `Pm4MeshExtractor`, potential similar logic in `WmoGroupMesh`).

## Mesh Comparison Utility Pattern (Planned, 2024-07-21)

### Intent
Provide a robust, extensible utility for comparing 3D meshes (MeshData) extracted from PM4 and WMO files. Move beyond simple vertex/triangle count checks to geometric and shape-based analysis.

### API Design
- Static utility class or interface: `MeshComparisonUtils.CompareMeshes(MeshData a, MeshData b) : MeshComparisonResult`
- Result object includes: match/mismatch, similarity score, detailed diagnostics (e.g., vertex/triangle differences, bounding box overlap, centroid distance).

### Comparison Metrics
- Vertex/triangle count
- Bounding box overlap/intersection
- Centroid distance
- Surface area comparison
- (Optional) Hausdorff or other shape similarity metrics
- Tolerance for translation, rotation, and scale differences

### Diagnostic Output
- Detailed report of mismatches (which vertices/triangles differ, by how much)
- Summary statistics (counts, scores, pass/fail)

### Integration
- Used in test project to compare PM4 and WMO meshes for validation and regression testing.
- Can be extended for future mesh analysis needs.

### Broader Vision
- Mesh comparison and matching logic will be used to reconstruct WMO placements from PM4 data, enabling the recreation and preservation of historical game worlds.
- Placement data will be output as YAML for now, supporting transparency, review, and future extensibility (e.g., chunk creation/export).
- The system is designed to inspire and empower others to explore, understand, and preserve digital history.

## New Pattern: Direct Chunk Assembly (2025-04-19)
- For legacy formats (e.g., WMO v14) that do not store explicit mesh data, implement a 'Direct Chunk Assembly' pattern:
  - Parse all relevant geometry subchunks (MOVT, MONR, MOTV, MOPY, MOVI, etc.) directly from the group data.
  - Assemble mesh structures (vertices, normals, UVs, triangles, materials) from the decoded arrays.
  - This approach bypasses legacy group file parsing and enables geometry extraction from raw chunk data.

## Patterns Update (2024-07-21)

### Liquid Handling Removal
- All code and dependencies related to liquid handling (including DBCD) have been removed from the project. Liquid support is out of scope and will be handled in a separate project.

### WMO Texturing Data Handling (v14/v17)
- All WMO chunk data related to texturing (including all string fields such as texture names, material names, etc.) must be mapped and unified for both v14 (mirrormachine) and v17 (wow.export) formats.
- The data model and export logic must support full WMO/OBJ+MTL reconstruction with correct texturing for both legacy and modern formats.
- String extraction and encoding must be robust and support all relevant chunk types (e.g., MATS, MOGN, MOGI, MOPY, MODS, etc.).
- Crosswalks between v14 and v17 chunk structures should be documented and implemented.

### Investigation Plan for Texturing Chunk Handling
- Review wow.export for v17 chunk/texturing support and data structures.
- Crosswalk v14 (mirrormachine) knowledge to v17 structures.
- Enumerate all relevant chunks and string fields for texturing.
- Design a unified data model for texturing.
- Update parsers and export logic accordingly.

### Pattern: Local Alias for Static Chunk Struct Methods (2024-07-21)
- When calling static methods on chunk structs (e.g., MOTX.ReadStrings) from model factories, use a local alias (e.g., using MOTXStruct = WoWToolbox.Core.WMO.MOTX;) to avoid ambiguity and ensure correct method resolution.
- This pattern resolved a persistent CS0149 error and should be considered best practice for future chunk struct integrations.