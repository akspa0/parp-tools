# Technical Context

## Technology Stack
- C# (.NET 8.0)
- Warcraft.NET for base file handling (including `ChunkedFile` base class)
- DBCD for DBC/DB2 operations

## Implementation Status

1. Core Framework
   ```csharp
   // Implemented Interfaces
   ILegacyChunk : IIFFChunk
   IVersionConverter<T> where T : IIFFChunk
   IIFFChunk // Base for all chunks
   
   // Base Classes
   ChunkConverterBase<T>
   Warcraft.NET.Files.ChunkedFile // Base for PM4File, PD4File
   
   // Concrete File Types
   PM4File // Handles PM4 structure
   PD4File // Handles PD4 structure
   ADTFile // Handles ADT structure (New)

   // Concrete Chunk Types (Examples - Not exhaustive)
   MVER // Shared MVER class 
   MSHDChunk
   MSPVChunk
   MCRCChunk // PD4 Specific
   MDDFChunk // ADT Specific (via Warcraft.NET)
   MODFChunk // ADT Specific (via Warcraft.NET)
   // ... many others reused between PM4/PD4
   
   // Core Models (New ADT)
   Placement
   ModelPlacement
   WmoPlacement

   // Extension Methods
   ChunkedFileExtensions
   ```

2. Validation System (Removed from solution)
   ```csharp
   // ... (Classes exist but project not built)
   ```

## Technical Constraints

1. Performance Requirements
   - Efficient memory usage for large files
   - Stream-based processing where possible
   - Parallel processing support
   - Reflection optimization for validation

2. Compatibility Requirements
   - Support for all WoW versions
   - Backward compatibility
   - Forward compatibility considerations
   - Warcraft.NET integration

3. Validation Requirements
   - Strict chunkvault compliance
   - Field-level validation
   - Relationship validation
   - Version compatibility checks

## Dependencies

1. External Libraries
   - Warcraft.NET (latest)
     - Location: lib/Warcraft.NET/Warcraft.NET
     - Usage: Base chunk handling, `ChunkedFile` base class, `IIFFChunk` interface, various attributes.
   - DBCD (latest)
     - Location: lib/DBCD
     - Usage: DBC/DB2 operations
   - SixLabors.ImageSharp (via Warcraft.NET dependency)

2. Project Dependencies
   ```xml
   <!-- WoWToolbox.Core -->
   <ItemGroup>
     <ProjectReference Include="..\..\lib\Warcraft.NET\Warcraft.NET\Warcraft.NET.csproj" />
     <ProjectReference Include="..\..\lib\DBCD\DBCD\DBCD.csproj" />
     <ProjectReference Include="..\..\lib\DBCD\DBCD.IO\DBCD.IO.csproj" />
   </ItemGroup>

   <!-- WoWToolbox.Tests -->
   <ItemGroup>
     <ProjectReference Include="..\..\src\WoWToolbox.Core\WoWToolbox.Core.csproj" />
     <ProjectReference Include="..\..\lib\Warcraft.NET\Warcraft.NET\Warcraft.NET.csproj" />
     <!-- PackageReference for Microsoft.NET.Test.Sdk, xunit, etc. -->
   </ItemGroup>

   <!-- WoWToolbox.AnalysisTool -->
   <ItemGroup>
     <ProjectReference Include="..\WoWToolbox.Core\WoWToolbox.Core.csproj" />
     <!-- YamlDotNet (To be added) -->
   </ItemGroup>
   ```

## Development Tools
- Visual Studio 2022 / Rider
- Git for version control
- xUnit for testing
- Markdown support for documentation

## New Development Tools/Components
*   **WoWToolbox.AnalysisTool:** Separate console application project for running data analysis tasks (e.g., MSLK log parsing and grouping).

## Technical Debt

1. Implementation Gaps
   - Field validation using reflection (Partially addressed by Warcraft.NET base loader?)
   - Legacy chunk loading logic
   - Version detection system
   - Relationship validation (Chunk index validation partially exists)
   - PD4File serialization (`NotImplementedException`)

2. Performance Considerations
   - Reflection usage in validation and base loading
   - Stream handling optimization
   - Memory management for large files
   - Caching opportunities

3. Testing Requirements
   - Unit test coverage (Needs expansion for PD4)
   - Integration test suite (PM4 exists, PD4 basic exists)
   - Performance benchmarks
   - Documentation validation

## Future Considerations

1. Extension Points
   - Custom validation rules
   - Format detection plugins
   - Conversion pipeline hooks
   - Documentation parsers

2. Performance Optimization
   - Cached reflection
   - Parallel validation
   - Lazy loading
   - Memory pooling

3. Documentation Integration
   - Automated spec validation
   - Live documentation updates
   - Validation reporting
   - Relationship mapping

## Reinforcement Framework
1. Documentation Integration
   - Automated chunkvault compliance checking
   - Format specification validation
   - Version compatibility matrix

2. Testing Strategy
   - Format conversion validation
   - Cross-version compatibility tests (PD4 vs PM4 insights)
   - Performance benchmarking
   - Documentation compliance tests

3. Integration Points
   - Warcraft.NET extension mechanisms
   - Version handling strategies
   - Format conversion pipelines
   - Validation frameworks 

## Additional Information
- `ushort Unk10`: Variable. **Confirmed Hypothesis:** Debug log analysis shows values fall within the range of MSVI indices. Likely an index into the `MSVI` chunk. The exact purpose of the linked MSVI entry requires further investigation.
- Confirmation and details of `Unk10`'s relationship to `MSUR`/`MSVI`. -> **Confirmed: Likely MSVI index. Purpose TBD.**

#### MSLK Chunk (Map SiLiK - Map Sill K?)
- Contains entries defining paths (`MSPICount >= 2`), single points (`MSPICount == 1`), or structural/metadata nodes (`MSPICount == 0`, `MSPIFirst == -1`).
- **Structure:** `MSLKEntry` (Defined in `WoWToolbox.Core/Navigation/PM4/Chunks/MSLK.cs`)
  - `byte Unk00`: Node type indicator? Observed: `0x01`, `0x11` in PM4 nodes; consistently `0x01` in preliminary PD4 node analysis. Geometry entries often `0x02`, `0x04`, `0x0A`, `0x0C`.
  - `byte Unk01`: Node sub-type/flag? Observed: `0x00`, `0x01`, `0x02`, `0x03`, `0x04` in nodes (PM4/PD4). Geometry often `0x00`-`0x03` or `0xFF`.
  - `ushort Unk02`: Consistently `0x0000` observed.
  - `uint Unk04`: **Group ID (Behaviour differs PM4 vs PD4 due to file scope):**
    - **PM4 (Multi-Object):** Creates **"Mixed Groups"** linking node entries (`MSPIFirst == -1`) with their corresponding geometry entries (`MSPIFirst >= 0`) for a specific object/group within the map tile.
    - **PD4 (Single Object):** Creates separate **"Node Only"** and **"Geometry Only"** groups. The ID likely categorizes related nodes or paths for the single WMO object.
  - `int MspiFirstIndex`: Index into `MSPI` chunk for geometry entries, `-1` for node entries. (Note: 24-bit signed integer).
  - `byte MspiIndexCount`: Number of points in `MSPI` chunk associated with a geometry entry, `0` for node entries.
  - `uint Unk0C`: Consistently `0xFFFFFFFF` observed.
  - `ushort Unknown_0x10`: Variable. **Confirmed MSVI index providing Node anchor for both PM4 and PD4.** Links MSLK Node entries to a specific vertex via `MSVI[Unknown_0x10] -> MSVT[index]`, defining the node's location. Also present in geometry entries (purpose TBD).
  - `ushort Unknown_0x12`: Consistently `0x8000` observed. Potential flag or bitmask.
- **Dependencies:** `MSPI` (geometry), `MSVI` (nodes via `Unknown_0x10`, potentially geometry), `MSVT` (nodes via `Unknown_0x10`/`MSVI`).
- **Analysis:**
  - **Node Identification (`Unk00`):** PM4 uses `0x11`; PD4 uses `0x01`.
  - **Geometry Identification (`Unk00`, PM4):** PM4 seems to use `0x12`, `0x14` for geometry entries.
  - **Node Sub-Types (`Unk01`):** Varies in both formats (PM4: `0x00-0x03`, `0x06` seen; PD4: `0x00-0x04` seen). Exact meaning TBD (likely via visualization).
  - **Node Anchor (`Unknown_0x10`):** Confirmed link to MSVI->MSVT for nodes in both PM4/PD4. Export logic implemented.
  - **Grouping (`Unk04`):** PM4 uses "Mixed" groups linking nodes/geometry; PD4 uses separate "Node Only" / "Geometry Only" groups. Difference likely due to file scope (multi- vs single-object).
  - **Data Integrity:** Some PD4 node entries show invalid `Unknown_0x10` links in logs.
- **Open Questions (MSLK):**
  - Precise meaning of different node sub-types/flags (`Unk00`, `Unk01`) based on anchor visualization (PM4 & PD4).
  - Purpose of `Unknown_0x10` link for *geometry* entries.
  - Meaning of the `Unk12` flag (`0x8000`).
  - Relationship between MSLK groups/nodes and other chunks (`MSUR`, `MSCN`, `MDOS`) beyond direct indices.