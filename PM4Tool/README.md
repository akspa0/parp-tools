# WoWToolbox v3

A comprehensive toolkit for parsing and working with World of Warcraft ADT (terrain) files and related data formats. Built in C# targeting .NET 8.0, this project provides libraries for chunk decoding and manipulation of complex packed data, along with specialized tools for analysis and data extraction.

## Features

- **PM4/PD4 File Parsing**
  - Complete chunk decoding system
  - Support for all known chunk types (MVER, MSHD, MSVT, MSVI, MSUR, etc.)
  - Efficient memory handling for large files
  - Stream-based processing

- **Geometry Processing**
  - Vertex and index export
  - Face generation from MSUR/MDSF/MDOS data
  - Path node linking (MSPV -> MSVT)
  - Coordinate transformation system
  - OBJ file export

- **Doodad Handling**
  - MSLK node identification and processing
  - Doodad placement anchor point extraction
  - Support for high MPRR/MPRL ratio files
  - Raw data export for analysis

- **Analysis Tools**
  - WoWToolbox.AnalysisTool for data investigation
  - WoWToolbox.FileDumper for YAML structure dumps
  - Detailed logging and debugging support
  - Visualization tools for MPRR links

## Project Structure

```
WoWToolbox/
├── src/
│   ├── WoWToolbox.Core/          # Main library
│   │   ├── Legacy/               # Legacy format support
│   │   ├── Extensions/           # Extension methods
│   │   └── Navigation/           # PM4/PD4/ADT handling
│   ├── WoWToolbox.AnalysisTool/  # Analysis utilities
│   └── WoWToolbox.FileDumper/    # YAML dumping tool
├── test/
│   └── WoWToolbox.Tests/         # Test suite
├── lib/                          # External dependencies
│   ├── Warcraft.NET/
│   └── DBCD/
├── docs/                         # Documentation
│   └── pm4_pd4_chunks.md        # Chunk specifications
└── chunkvault/                   # Format specifications
```

## Dependencies

- [Warcraft.NET](https://github.com/ModernWoWTools/Warcraft.NET) - Base chunk handling
- [DBCD](https://github.com/wowdev/DBCD) - DBC/DB2 operations
- SixLabors.ImageSharp (via Warcraft.NET)
- .NET 8.0 SDK

## Getting Started

### Prerequisites

1. Install .NET 8.0 SDK
2. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/yourusername/WoWToolbox_v3.git
   ```

### Building

Use the provided batch scripts in the workspace root:

```bash
# Build the entire solution
./build.bat

# Clean build artifacts
./clean.bat

# Run tests
./test.bat

# Run clean, build, and test sequentially
./run_all.bat
```

### Running Tests

The test suite includes comprehensive tests for:
- PM4/PD4 file parsing
- Geometry assembly
- OBJ export
- Batch processing
- Coordinate transformations

```bash
dotnet test src/WoWToolbox.sln
```

### Using the File Dumper

The WoWToolbox.FileDumper tool can generate detailed YAML dumps of PM4/ADT file structures:

```bash
dotnet WoWToolbox.FileDumper.dll -d <input_dir> -o <output_dir>
```

## Technical Details

### Coordinate System

The project uses standardized coordinate transformations:
- Scale Factor: 36.0f
- Coordinate Offset: 17066.666f
- Expected game coordinate range: +/- 17066.66

### File Format Support

- **PM4 Files**: Complete support for loading, parsing, and geometry extraction
- **PD4 Files**: Basic implementation and OBJ export
- **ADT Files**: Parsing implementation with Warcraft.NET integration
- **Chunk Types**: Support for MVER, MSHD, MSVT, MSVI, MSUR, MDOS, MDSF, MDBH, MPRL, MPRR, MSPV, MSPI, MSLK, MCRC, MSCN

### Performance Considerations

- Stream-based processing for large files
- Memory optimization for chunk handling
- Parallel processing support where applicable
- Specialized handling for high MPRR/MPRL ratio files

## Contributing

Contributions are welcome! Please read through the existing issues and documentation before submitting pull requests.

### Development Guidelines

1. Follow existing code patterns and naming conventions
2. Add appropriate XML documentation comments
3. Include unit tests for new features
4. Update chunk documentation as needed
5. Maintain backward compatibility where possible

## Known Issues

- Log file size limitations for large files
- Incomplete Doodad property decoding (rotation, scale, model ID)
- MPRR structure partially unknown
- Some validation assertions currently bypassed
- Directory processing termination in AnalysisTool

## Acknowledgments

- ModernWoWTools team for Warcraft.NET
- WoWDev community for DBCD