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
   - Specific file format classes (e.g., `PM4File`, `PD4File`) inherit from `Warcraft.NET.Files.ChunkedFile`.
   - Derived classes define properties for expected chunks (e.g., `public MVERChunk? MVER { get; }`).
   - The base `ChunkedFile` constructor uses reflection to find these properties and attempts to load corresponding chunks from the provided byte data.
   - Chunk properties can be marked `[ChunkOptional]` if their presence is not guaranteed.
   - Chunk definitions (`IIFFChunk` implementations like `MVERChunk`, `MSPVChunk`, `MCRCChunk`) handle their own internal data parsing via `LoadBinaryData`.
   - This pattern relies on the base class handling the overall file structure and chunk discovery, while specific chunk classes handle their internal structure.

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
     /PD4   // <-- New
       - PD4File.cs
       /Chunks
         - MCRCChunk.cs
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
    *   **PD4 (Single Object):** Analysis of tested PD4 files (`6or_garrison...`) showed `Unknown_0x04` still acts as a group ID, but *not* to link nodes and geometry directly. It creates separate **"Node Only"** and **"Geometry Only"** groups. This structure is likely sufficient because the file represents a single object, so explicit node-geometry linking within the group ID isn't required. The group ID likely categorizes related paths or related nodes pertinent to that single object.

## Chunk Correlations (Investigated)

Based on analysis of chunk structures (`MSLKEntry`, `MsurEntry`, `MSCNChunk`) and codebase searches:

*   **Direct Implemented Links:**
    *   `MSLK` -> `MSPI` (via `MspiFirstIndex`): Used for defining geometry paths/points.
    *   `MSUR` -> `MSVI` (via `MsviFirstIndex`, `IndexCount`): Used for defining surfaces via vertex indices.
    *   `MSUR` -> `MDOS` (via `MdosIndex`): Explicitly links surfaces to `MDOS` entries.
*   **Likely Unused Links:**
    *   `MSLK` -> `MSVI` (via `Unknown_0x10`): This field likely indexes `MSVI`, but its purpose is currently unclear and not explicitly used in the examined code (tests/export logic). It might associate metadata or properties from `MSVI` to `MSLK` nodes or paths.
*   **No Implemented Direct Links Found:**
    *   `MSLK` <-> `MSCN`: No code found directly correlating `MSLK` entries with the `Vector3` data in `MSCN`.
    *   `MSLK` <-> `MSUR`: No code found directly correlating `MSLK` entries with `MSUR` surface definitions.
*   **Potential Semantic Links (Requires Further Research):**
    *   The `MSLK.Unk04` Group ID (especially in PM4) might logically group related `MSUR` surfaces or `MSCN` objects, even if not directly coded.
    *   `MSLK` node types (`Unk00`/`Unk01` in PM4) might signify relationships to other chunk data.
    *   `MSCN` vectors might provide normal data for vertices used by `MSUR` or `MSLK`, but the indexing mechanism isn't immediately clear from the structures alone.