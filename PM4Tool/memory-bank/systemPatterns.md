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
   - The base `ChunkedFile` constructor uses reflection to find these properties and attempts to load corresponding chunks from the byte data provided for *that file*.
   - Chunk properties can be marked `[ChunkOptional]` if their presence is not guaranteed.
   - Chunk definitions (`IIFFChunk` implementations) handle their own internal data parsing.
   - Handling composite file formats like ADT (which consists of multiple physical files) requires loading each relevant split file into its corresponding `Warcraft.NET` class (e.g., `Terrain`, `TerrainObjectZero`) and then combining the data logically in services like `AdtService`.

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

## Emerging Patterns / Discoveries
*   **MSLK Hierarchical Structure (PM4 Confirmed, PD4 Different - Tied to File Scope):**
    *   **Context:** PM4 files represent multi-object map tiles, while the tested PD4 files represent single WMO objects.
    *   **PM4 (Multi-Object):** Analysis using `WoWToolbox.AnalysisTool` on PM4 log data confirmed `Unknown_0x04` acts as a group/object identifier. It creates **"Mixed Groups"** linking metadata node entries (`MspiFirstIndex == -1`) directly to their corresponding geometry path entries (`MspiFirstIndex >= 0`) for a specific object within the collection. Different node types (`Unk00`/`Unk01`) were identified.
    *   **PD4 (Single Object):** Analysis of tested PD4 files (`6or_garrison...`) showed `Unknown_0x04` still acts as a group ID, but *not* to link nodes and geometry directly. It creates separate **"Node Only"** and **"Geometry Only"** groups.
*   **ADT/PM4 Correlation:** PM4 data (`m_destructible_building_index` in `MDOS` via `MDSF`) can be linked to ADT object placements via **Unique IDs** (`UniqueID` field in `MDDFEntry`/`MODFEntry`). This is key for understanding PM4/MSLK context.
*   **Surface Definition (Hypothesis -> Confirmed Links):** `MSUR` defines surface geometry using indices from `MSVI` (which point to `MSVT` vertices). `MDSF` acts as a mapping layer, linking `MSUR` surfaces (`msur_index`) to `MDOS` destructible object states (`mdos_index`).

## Chunk Correlations (Investigated & Updated)

Based on analysis of chunk structures, codebase searches, and recent discoveries:

*   **Direct Implemented/Confirmed Links:**
    *   `MSLK` -> `MSPI` (via `MspiFirstIndex`): Used for defining geometry paths/points.
    *   `MSUR` -> `MSVI` (via `MsviFirstIndex`, `IndexCount`): Defines surface indices.
    *   `MSUR` -> `MDOS` (via `MdosIndex`): Was listed, now understood to be via MDSF.
    *   `MDSF` -> `MSUR` (via `msur_index`): **NEW** Links destruction data to specific surfaces.
    *   `MDSF` -> `MDOS` (via `mdos_index`): **NEW** Links destruction data to specific destructible object state entries.
*   **Confirmed Node Links:**
    *   `MSLK` -> `MSVI` (via `Unknown_0x10`): Anchors nodes to vertices via MSVI->MSVT (PM4 & PD4).
*   **External Links:**
    *   `PM4 Data` <-> `ADT Object Placement` (via **UniqueID**): **NEW** Allows correlating PM4 structures (e.g., MSLK groups) to world objects.
*   **Unknown/Unused Links:**
    *   `MSLK` -> `MSVI` (via `Unknown_0x10` for *Geometry* entries): Purpose still TBD.
*   **No Implemented Direct Links Found:**
    *   `MSLK` <-> `MSCN`: No code found directly correlating.
    *   `MSLK` <-> `MSUR`: No direct code link found (but potentially linked logically via MSLK group ID / ADT UniqueID).
*   **Potential Semantic Links (Requires Further Research):**
    *   The `MSLK.Unk04` Group ID / ADT UniqueID might logically group related `MSUR` surfaces (via MDSF/MDOS?) or `MSCN` objects.
    *   `MSLK` node types (`Unk00`/`Unk01`) might signify relationships to other chunk data (pending visualization).
    *   `MSCN` vectors might provide normal data for vertices used by `MSUR` or `MSLK`, but the indexing mechanism isn't immediately clear.