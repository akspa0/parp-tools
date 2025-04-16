# Technical Context

## Core Technologies
-   **Language:** C#
-   **Framework:** .NET 9.0 (across all projects: `Core`, `Common`, `Mpq`, `Tests`, `MSCNExplorer`)
-   **Build System:** .NET SDK (using `dotnet` CLI)
-   **Testing:** xUnit
-   **Logging:** Basic `Debug.WriteLine` for now.

## Project Structure
-   `WoWToolbox.Core`: Core data structures (`MeshGeometry`), WMO parsing logic (`WmoRootFile`, `WmoGroupFile`, `WmoGroupMesh`).
-   `WoWToolbox.Common`: Shared utilities, chunk reading (`ChunkReader`), low-level data types, PM4 parsing logic (`PM4File`, chunk definitions like `MSVTChunk`, `MSVIChunk` etc.).
-   `WoWToolbox.Mpq`: MPQ archive handling (likely used for accessing game files, dependency for tests).
-   `WoWToolbox.MSCNExplorer`: Contains the `Pm4MeshExtractor` for converting PM4 data to `MeshGeometry`.
-   `WoWToolbox.Tests`: Consolidated xUnit tests for all projects, including WMO and PM4 mesh extraction/export.

## Development Setup
-   Requires .NET 9.0 SDK.
-   Standard `dotnet build`, `dotnet test` commands.
-   Test data stored in `test_data` directory (needs population).
-   Output artifacts (like generated OBJ files) typically go into `bin/Debug/net9.0/TestOutput` directories within the respective test project's build output.

## Technical Constraints & Decisions
-   **Endianness:** Chunk names are little-endian (reversed 4-char codes), chunk data is native endian (handled by `ChunkReader`).
-   **Coordinate Systems:** WMO and PM4 use different coordinate systems/scales. Transformation logic is required (e.g., `MsvtToWorld_PM4`).
-   **Target Framework:** Standardized on .NET 9.0 for modern features.
-   **Testing Strategy:** Unit tests verify loading and export functionalities. Visual inspection of OBJ files provides qualitative validation.

## Dependencies
-   **External:** xUnit (for testing).
-   **Internal:** Projects depend on each other (e.g., `Tests` depends on `Core`, `Common`, `Mpq`, `MSCNExplorer`; `MSCNExplorer` depends on `Common`, `Core`).

## Key Projects & Dependencies

- `WoWToolbox.Core`:
    - **Purpose:** Foundational library for WoW data structures, file parsing (ADT, WMO, M2, etc.), and common utilities.
    - **Status:** Actively developed. Aligned to .NET 9.0. DBCD dependency removed.
    - **Key Classes:** `ADTFile`, `WmoRootLoader`, `WmoGroupMesh`, `ChunkReader`, `PM4File`, `Vector3`, `Quaternion`.
- `WoWToolbox.Tests`:
    - **Purpose:** Unit and integration tests for `Core` and other libraries.
    - **Status:** Actively developed. Aligned to .NET 9.0. Contains tests for WMO group loading/saving (`WmoGroupMeshTests`) and PM4 extraction (`Pm4MeshExtractorTests` - moved from MSCNExplorer).
- `WoWToolbox.MSCNExplorer`:
    - **Purpose:** Library focused on parsing and analysing MSCN/PM4/PD4 navigation data.
    - **Status:** Under development. Aligned to .NET 9.0. Contains `Pm4MeshExtractor` for render mesh extraction (currently facing build errors).
    - **Key Classes:** `Pm4MeshExtractor`, `PM4File` (likely uses `Core`'s version), potentially others for MSCN/PD4.
- `WoWToolbox.MPRRExplorer`:
    - **Purpose:** Library focused on parsing and analysing MPRR pathfinding data.
    - **Status:** Existing, likely stable. Aligned to .NET 9.0.
- `WoWToolbox.AnalysisTool`:
    - **Purpose:** Command-line tool for running various analyses on WoW data, potentially integrating functionality from other libraries.
    - **Status:** Existing, likely stable. Aligned to .NET 9.0.

## Development Setup
- **Prerequisites:** .NET 9.0 SDK.
- **Build:** `dotnet build` in the root or specific project directories.
- **Testing:** `dotnet test` in the root or specific project directories. Filtering can be used (e.g., `--filter FullyQualifiedName~<Namespace.ClassName>`).
- **Test Data:** Located in `test_data/` directory, potentially with subdirectories for specific data types (e.g., `335_wmo`). Tests reference files relative to this structure or via absolute paths.
- **Output:** Test outputs (like generated OBJ files) are typically placed in `bin/Debug/net9.0/TestOutput/` within the respective test project directory.

## Technical Constraints & Considerations
- **Endianness:** WoW file chunks often use specific endianness (e.g., chunk names are little-endian). Parsers must handle this correctly.
- **Data Formats:** Strict adherence to Blizzard's binary file formats is required.
- **Performance:** Parsing large game files can be resource-intensive.
- **Coordinate Systems:** WoW uses different coordinate systems. Transformations (like `MsvtToWorld_PM4`) are crucial.
- **Data Interdependencies:** Chunks within files often reference each other (e.g., `MSUR` indices into `MSVI`). Robust index validation is necessary.
- **Build Errors:** Current build errors in `Pm4MeshExtractor` related to `UInt32` vs `Int32` index types need resolution.

## Technology Stack
- C# (.NET 9.0)
- Warcraft.NET for base file handling (including `ChunkedFile` base class)

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
   // ADT is handled by loading specific Warcraft.NET classes for each split file:
   Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain // Example: Base .adt file
   Warcraft.NET.Files.ADT.TerrainObject.Zero.TerrainObjectZero // Example: _obj0.adt file
   // ... potentially others like TerrainTexture, TerrainLOD ...

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

2. Project Dependencies
   *The main solution file (`src/WoWToolbox.sln`) now includes `WoWToolbox.Core`, `WoWToolbox.AnalysisTool`, `WoWToolbox.MPRRExplorer`, `WoWToolbox.MSCNExplorer`, and `WoWToolbox.Tests`.*
   ```xml
   <!-- WoWToolbox.Core (Targets net9.0) -->
   <ItemGroup>
     <ProjectReference Include="..\..\lib\Warcraft.NET\Warcraft.NET\Warcraft.NET.csproj" />
     <!-- Removed DBCD references -->
     <!-- <ProjectReference Include="..\..\lib\DBCD\DBCD\DBCD.csproj" /> -->
     <!-- <ProjectReference Include="..\..\lib\DBCD\DBCD.IO\DBCD.IO.csproj" /> -->
   </ItemGroup>

   <!-- WoWToolbox.Tests (Targets net9.0) -->
   <ItemGroup>
     <ProjectReference Include="..\..\src\WoWToolbox.Core\WoWToolbox.Core.csproj" />
     <ProjectReference Include="..\..\lib\Warcraft.NET\Warcraft.NET\Warcraft.NET.csproj" />
     <ProjectReference Include="..\..\src\WoWToolbox.MSCNExplorer\WoWToolbox.MSCNExplorer.csproj" />
     <!-- PackageReference for Microsoft.NET.Test.Sdk, xunit, etc. -->
   </ItemGroup>

   <!-- WoWToolbox.AnalysisTool (Targets net9.0) -->
   <ItemGroup>
     <ProjectReference Include="..\WoWToolbox.Core\WoWToolbox.Core.csproj" />
     <!-- YamlDotNet -->
     <PackageReference Include="YamlDotNet" Version="15.1.2" /> 
   </ItemGroup>

   <!-- WoWToolbox.MSCNExplorer (Targets net9.0) -->
   <ItemGroup>
     <ProjectReference Include="..\WoWToolbox.Core\WoWToolbox.Core.csproj" />
     <!-- System.CommandLine -->
     <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" /> 
   </ItemGroup>

   <!-- WoWToolbox.MPRRExplorer (Targets net9.0) -->
   <ItemGroup>
     <ProjectReference Include="..\..\lib\Warcraft.NET\Warcraft.NET\Warcraft.NET.csproj" />
     <ProjectReference Include="..\..\lib\DBCD\DBCD\DBCD.csproj" />
     <ProjectReference Include="..\..\lib\DBCD\DBCD.IO\DBCD.IO.csproj" />
     <ProjectReference Include="..\..\src\WoWToolbox.Core\WoWToolbox.Core.csproj" />
   </ItemGroup>
   ```

## Development Tools
- Visual Studio 2022 / Rider
- Git for version control
- xUnit for testing
- Markdown support for documentation
- **Batch Scripts:**
  - `build.bat`: Located in the workspace root. Runs `dotnet build src/WoWToolbox.sln` to compile the entire solution.
  - `clean.bat`: Located in the workspace root. Runs `dotnet clean src/WoWToolbox.sln` to clean build artifacts.
  - `test.bat`: Located in the workspace root. Runs `dotnet test src/WoWToolbox.sln`.
  - `run_all.bat`: Located in the workspace root. Runs clean, build, and test sequentially.

## New Development Tools/Components
*   **WoWToolbox.AnalysisTool:** Separate console application project for running data analysis tasks (e.g., PM4/ADT correlation, MSLK log analysis).
*   **WoWToolbox.FileDumper:** (NEW) Console application for detailed YAML dumping of PM4/ADT (`_obj0.adt`) file structures. Uses `WoWToolbox.Core`, `Warcraft.NET`, and `YamlDotNet`.

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
- Contains entries defining paths (`MSPICount >= 2`), single points (`MSPICount == 1`), or **Doodad placements** (`MSPICount == 0`, `MSPIFirst == -1`, representing nodes/metadata for M2/MDX models).
- **Structure:** `MSLKEntry` (Defined in `WoWToolbox.Core/Navigation/PM4/Chunks/MSLK.cs`)
  - `byte Unk00`: **Doodad Type/Flags?** Observed: `0x01`, `0x11` in PM4 nodes; consistently `0x01` in preliminary PD4 node analysis. Geometry entries often `0x02`, `0x04`, `0x0A`, `0x0C`. Likely encodes part of the Doodad's properties or identity.
  - `byte Unk01`: **Doodad Sub-type/Flags?** Observed: `0x00`, `0x01`, `0x02`, `0x03`, `0x04`, `0x06` in nodes (PM4/PD4). Geometry often `0x00`-`0x03` or `0xFF`. Likely encodes part of the Doodad's properties (e.g., rotation, scale flags).
  - `ushort Unk02`: Consistently `0x0000` observed.
  - `uint Unk04`: **Group ID / Doodad Identifier?** (Behaviour differs PM4 vs PD4 due to file scope):
    - **PM4 (Multi-Object):** Creates **"Mixed Groups"** linking Doodad node entries with their corresponding geometry path entries (if any) for a specific object/group within the map tile. May also link to the Doodad model's ID (e.g., via `MDBH`).
    - **PD4 (Single Object):** Creates separate **"Node Only"** (Doodad) and **"Geometry Only"** groups. The ID likely categorizes related Doodads or paths for the single WMO object.
  - `int MspiFirstIndex`: Index into `MSPI` chunk for geometry entries, `-1` for Doodad node entries. (Note: 24-bit signed integer).
  - `byte MspiIndexCount`: Number of points in `MSPI` chunk associated with a geometry entry, `0` for Doodad node entries.
  - `uint Unk0C`: Consistently `0xFFFFFFFF` observed.
  - `ushort Unknown_0x10`: **Anchor Point Vertex Index.** Confirmed MSVI index providing Doodad anchor position for both PM4 and PD4. Links MSLK Node entries to a specific vertex via `MSVI[Unknown_0x10] -> MSVT[index]`. Also present in geometry entries (purpose TBD). **Hypothesis that this is always 0xFFFF is REJECTED.**
  - `ushort Unknown_0x12`: Consistently `0x8000` observed. **Potential Doodad Property/Flag?** (e.g., scaling, rotation bitmask).
- **Dependencies:** `MSPI` (geometry), `MSVI` (Doodad anchor via `Unknown_0x10`, potentially geometry), `MSVT` (Doodad anchor via `Unknown_0x10`/`MSVI`), **potentially `MDBH` (for Doodad filenames/IDs via `Unk04` or other fields).**
- **Analysis (Updated):**
  - **Doodad Identification (`Unk00`, `Unk01`, `Unk04`):** These fields likely combine to identify the specific Doodad model (potentially linking to `MDBH`) and its properties (type, rotation, scale). Further investigation needed. PM4 uses `0x11`; PD4 uses `0x01` for `Unk00` in nodes.
  - **Geometry Identification (`Unk00`, PM4):** PM4 seems to use `0x12`, `0x14` for geometry entries.
  - **Doodad Anchor (`Unknown_0x10`):** Confirmed link to MSVI->MSVT for Doodad placement anchor point in both PM4/PD4. Export logic implemented.
  - **Grouping (`Unk04`):** PM4 uses "Mixed" groups linking Doodad nodes/geometry; PD4 uses separate "Node Only" / "Geometry Only" groups. Difference likely due to file scope (multi- vs single-object).
  - **Data Integrity:** Some PD4 node entries show invalid `Unknown_0x10` links in logs.
  - **Field Constraints:** `Unk00` < 32 observed. `Unk01` usually < 12 (or 0xFF), few violations seen. `Unk02` confirmed `0x0000`. `Unk0C` confirmed `0xFFFFFFFF`. `Unk12` confirmed `0x8000`.

#### MPRL Chunk
- **Structure:** `MPRLEntry` (Defined in `WoWToolbox.Core/Navigation/PM4/Chunks/MPRLChunk.cs`)
  - `ushort Unk00`: **Group ID?** No direct correlation found with `MSLK.Unk0C` in naive checks.
  - `short Unk02`: **Confirmed Constant:** `-1` (0xFFFF as ushort).
  - `ushort Unk04`: Variable.
  - `ushort Unk06`: **Confirmed Constant:** `0x8000`.
  - `C3Vector Position`: Vertex position.
  - `short Unk14`: Variable, small range (-1 to 15 observed).
  - `ushort Unk16`: Variable (0x0 or 0x3FFF observed).

#### MPRR Chunk
- **Structure:** Variable length sequences of `ushort`, each terminated by `0xFFFF`. The `ushort` immediately before the terminator is a potential flag/type (values like 0x300, 0x1100 seen). The preceding `ushort` values are indices, but their target is **unknown** (likely *not* MPRL based on observed index ranges).
- **Current Implementation:** `MPRRChunk.cs` still uses the *old* fixed-pair structure (`MprrEntry`). Refactoring needed.

## Recent Technical Context (2024-04-15)

- Code now supports type/object-centric analysis and OBJ export (by unk00, unk01, object id).
- Most OBJ exports are points/lines; mesh connectivity is missing.
- MSUR (for faces/surfaces) and MSCN (for normals) are the next targets for analysis and integration.

## Planned Components (2024-06)
*   **WoWToolbox.MSCNExplorer.Pm4MeshExtractor:** (Implemented) A new class within the MSCNExplorer tool project. Encapsulates the logic for extracting renderable meshes (vertices, triangles) from PM4/PD4 files using MSVT, MSVI, and MSUR chunks. Logic was refactored from `WoWToolbox.Tests`.
*   **WoWToolbox.Tests.Navigation.Pm4MeshExtractorTests:** (Implemented) Tests for `Pm4MeshExtractor`, moved from `MSCNExplorer` project to the main `WoWToolbox.Tests` project.
*   **WoWToolbox.MSCNExplorer Comparison Logic:** (Planned) Implementation of mesh comparison algorithms within the MSCNExplorer tool to compare meshes extracted from PM4 and WMO files.

### Core Data Structures

*   `WoWToolbox.Core.PM4File`: Represents the loaded PM4/PD4 file, containing various chunk data (MSVT, MSVI, MSUR, MDOS, MDSF, MSCN, etc.).
*   `WoWToolbox.Core.WMO.WmoRoot`: Represents the root WMO file, containing header info, materials, doodad sets, and group information.
*   `WoWToolbox.Core.WMO.WmoGroup`: Represents a single WMO group file, containing geometry chunks (MOVT, MOVB, MONR, MOCV, etc.).
*   `WoWToolbox.Core.WMO.WmoGroupMesh`: (Being phased out for direct `MeshData` return) Currently used to hold processed mesh data (vertices, triangles) from a WMO group.
*   `WoWToolbox.Core.Models.MeshData`: New common structure to hold extracted mesh geometry. Contains `List<Vector3> Vertices` and `List<int> Indices`.
*   Chunk Classes (e.g., `MSVTChunk`, `MOVTChunk`, `MVERChunk`): Specific classes within `WoWToolbox.Core` namespaces representing the structure and data of individual file chunks.
*   `System.Numerics.Vector3`, `Vector2`: Used for coordinates, normals, and UVs.