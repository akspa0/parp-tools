# Technical Context

## Technology Stack

### Runtime
- **.NET 9.0**: Latest LTS version, C# 13 features
- **Target**: `net9.0` for all projects
- **Platform**: Cross-platform (Windows, Linux, macOS)

### Core Libraries

#### File Format Handling
- **Warcraft.NET**: WoW file format parsing (ADT, WDT, DBC)
  - Location: `lib/Warcraft.NET/`
  - Used for: Chunk parsing, FourCC handling, binary I/O
- **DBCD**: DBC file parsing with WoWDBDefs
  - Location: `lib/wow.tools.local/DBCD/`
  - Used for: AreaTable, Map.dbc parsing

#### 3D & Geometry
- **System.Numerics**: Vector3, Matrix4x4 for transforms
- **SharpGLTF**: GLB export for 3D terrain meshes

#### Web Viewer
- **Vanilla JavaScript**: No frameworks, minimal dependencies
- **Leaflet.js**: Map visualization and tile navigation
- **Built-in HTTP server**: C# `HttpListener` for serving viewer

### External Dependencies

#### Noggit Reference
- **Location**: `lib/noggit-red/src/noggit/Alphamap.cpp`
- **Purpose**: Reference implementation for Alpha MCAL decoding
- **Usage**: Mirrored in `McalAlphaDecoder.cs`

#### WoWDBDefs
- **Location**: `lib/WoWDBDefs/definitions/`
- **Purpose**: DBC schema definitions for DBCD

## Development Setup

### Prerequisites
```powershell
# Install .NET 9.0 SDK
winget install Microsoft.DotNet.SDK.9

# Restore dependencies
dotnet restore
```

### Build
```powershell
# Build entire solution
dotnet build WoWRollback.sln

# Build specific project
dotnet build WoWRollback.AdtConverter/WoWRollback.AdtConverter.csproj
```

### Run
```powershell
# Run CLI
dotnet run --project WoWRollback.Cli -- <command> <args>

# Run AdtConverter
dotnet run --project WoWRollback.AdtConverter -- <command> <args>
```

## Project Structure

### Solution Organization
```
WoWRollback.sln
├── WoWRollback.Core/              # Shared models, services, utilities
├── WoWRollback.AdtModule/         # ADT parsing/writing (Alpha/LK)
├── WoWRollback.DbcModule/         # DBC file parsing
├── WoWRollback.AnalysisModule/    # Spatial clustering, UniqueID analysis
├── WoWRollback.ViewerModule/      # Web viewer data generation
├── WoWRollback.Viewer/            # Static web viewer (HTML/JS/CSS)
├── WoWRollback.LkToAlphaModule/   # LK → Alpha conversion
├── WoWRollback.AdtConverter/      # Standalone conversion CLI
├── WoWRollback.Orchestrator/      # Pipeline orchestration
├── WoWRollback.Cli/               # Unified CLI entry point
└── WoWRollback.Verifier/          # Validation and comparison
```

### Key Directories
```
WoWRollback/
├── docs/                          # Documentation
│   ├── planning/                  # Feature plans
│   └── archived/                  # Old/superseded docs
├── memory-bank/                   # Project memory
├── analysis_output/               # Generated analysis data
├── project_output/                # Conversion outputs
├── debug_mcal/                    # MCAL debug dumps
└── lib/                           # External libraries
```

## Technical Constraints

### File Format Constraints

#### Alpha ADT (0.5.3-0.6.0)
- Single file: All data in one `.adt` file
- Monolithic WDT: Can embed terrain data
- MCAL format: Uncompressed or simple compression
- Model format: `.mdx` extension (not `.m2`)

#### Lich King ADT (3.3.5a)
- Split files: `_root.adt`, `_obj0.adt`, `_tex0.adt`
- MCAL compression: More complex schemes
- Model format: `.m2` extension
- MH2O liquids: Advanced liquid system

### Conversion Constraints

#### Alpha → LK
- Must preserve: Terrain geometry, texture layers, object placements
- Must handle: `.mdx` → `.m2` renaming, MCAL decompression

#### LK → Alpha
- Must preserve: Terrain geometry, texture layers, object placements
- Must strip: LK-only features (MH2O, advanced chunks)
- Must handle: `.m2` → `.mdx` renaming, MCAL compression

## Debugging & Diagnostics

### Debug Outputs

#### MCAL Dumps
- Location: `debug_mcal/YY_XX/`
- Purpose: Compare Alpha vs. LK MCAL encoding
- Enable: `--verbose-logging` flag

#### Validation Reports
- Location: `project_output/<map>/validation/`
- Format: CSV, JSON

### Logging Configuration
```csharp
// Minimal logging (default)
logger.LogLevel = LogLevel.Info;

// Verbose logging
logger.LogLevel = LogLevel.Debug;
```

## Testing Infrastructure

### Test Data
- Location: `test_data/` (not in repo, user-provided)
- Structure:
  ```
  test_data/
  ├── 0.5.3/tree/World/Maps/
  ├── 0.5.5/tree/World/Maps/
  └── 3.3.5/tree/World/Maps/
  ```

### Test Execution
```powershell
# Run all tests
dotnet test WoWRollback.sln
```

## Known Issues

### Issue 1: FourCC Reversal
- Problem: FourCCs stored reversed on disk
- Solution: Centralized `ReverseFourCC()` in `Chunk.cs`
- Status: ✅ Resolved

### Issue 2: MCAL Compression
- Problem: Multiple compression schemes in Alpha
- Solution: Mirror Noggit's algorithm
- Status: ✅ Resolved

### Issue 3: Coordinate Systems
- Problem: Alpha vs. LK use different systems
- Solution: Explicit transform functions
- Status: ⚠️ Needs verification
