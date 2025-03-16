
# WCAnalyzer: Comprehensive Warcraft File Analysis Toolkit

[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-blue)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![C#](https://img.shields.io/badge/Language-C%23-green)](https://docs.microsoft.com/en-us/dotnet/csharp/)

WCAnalyzer is a high-performance .NET 8.0 command-line toolkit designed for advanced analysis of Warcraft game files, including ADT (terrain), PM4, and unique ID analysis. Built with memory optimization and scalability in mind, it provides comprehensive functionality for processing large datasets, extracting metadata, and exporting to various formats.

## Key Features

### ADT Analysis
- **Terrain Processing**: Extract and analyze terrain height maps and textures
- **Model Placement**: Identify and extract model placement data from ADT files
- **WMO References**: Extract World Model Object references and placement data
- **Texture References**: Analyze and extract texture usage and references
- **Report Generation**: Create detailed reports of ADT file contents

### PM4 Analysis
- **Chunk Extraction**: Parse and process all PM4 chunk types (MPRL, MSVI, MSPI, etc.)
- **Special Value Correlation**: Identify relationships between special metadata values and positions
- **Memory-Optimized Processing**: Stream-based parsing designed for large datasets
- **Export Options**:
  - Multiple OBJ formats (standard, enhanced, consolidated, clustered)
  - CSV reports for all chunk types
  - Terrain data extraction
  - 2D map visualization

### UniqueID Analysis
- **FileDataID Resolution**: Resolve and map FileDataID references to file paths
- **Reference Tracking**: Track and analyze cross-references between files
- **Pattern Analysis**: Identify patterns in unique ID usage and distribution
- **Batch Processing**: Efficiently process large sets of unique IDs

## Installation

### Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later
- Windows, macOS, or Linux operating system

### Building from Source

```bash
# Clone the repository
git clone https://github.com/akspa0/parp-tools

# Navigate to the project directory
cd WCAnalyzer

# Download Warcraft.NET
git submodule init
git submodule update

# Build the solution
dotnet build

# Run the application
cd WCAnalyzer.CLI
dotnet run -- --help
```

## Usage Examples

### ADT Analysis

```bash
# Analyze ADT files
dotnet run -- -d "path/to/adt/files" -o "output/directory"

# Analyze ADT files with model extraction
dotnet run -- -d "path/to/adt/files" -o "output/directory" --extract-models

# Merge texture information from _tex0.adt files
dotnet run -- -d "path/to/adt/files" -o "output/directory" --merge-tex

# Analyze ADT files with listfile for FileDataID resolution
dotnet run -- -d "path/to/adt/files" -o "output/directory" -l "path/to/listfile.csv"
```

### PM4 Analysis

```bash
# Basic PM4 analysis
dotnet run -- pm4 -d "path/to/pm4/files" -o "output/directory"

# Generate CSV reports
dotnet run -- pm4 -d "path/to/pm4/files" -o "output/directory" --export-csv

# Export to enhanced OBJ with special value sorting
dotnet run -- pm4 -d "path/to/pm4/files" -o "output/directory" --export-enhanced-obj

# Extract terrain data and generate map visualization
dotnet run -- pm4 -d "path/to/pm4/files" -o "output/directory" --extract-terrain --generate-map

# Correlate special values
dotnet run -- pm4 --correlate-special-values --csv-directory "path/to/csv/reports" -o "correlation/output"
```

### UniqueID Analysis

```bash
# Analyze unique IDs
dotnet run -- uniqueid -l "path/to/listfile.csv" -d "path/to/files" -o "output/directory"

# Generate reference report
dotnet run -- uniqueid -l "path/to/listfile.csv" -d "path/to/files" -o "output/directory" --reference-report

# Process batch of IDs
dotnet run -- uniqueid --id-list "path/to/id_list.txt" -o "output/directory"
```

## Command Reference

### ADT Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --directory <directory>` | Directory containing ADT files to analyze | |
| `-o, --output <output>` | Directory to write analysis output to | |
| `-v, --verbose` | Enable verbose logging | `false` |
| `-r, --recursive` | Recursively search for files in subdirectories | `false` |
| `-l, --listfile <listfile>` | Path to a listfile for resolving FileDataID references | |
| `--extract-models` | Extract model placement data | `false` |
| `--extract-textures` | Extract texture references | `false` |
| `--merge-tex` | Merge texture data from _tex0.adt files | `false` |
| `--extract-terrain` | Extract terrain height data | `false` |
| `--generate-report` | Generate detailed analysis report | `true` |
| `-f, --file <file>` | Path to a specific ADT file to analyze | |

### PM4 Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --directory <directory>` | Directory containing PM4 files to analyze | |
| `-o, --output <output>` | Directory to write analysis output to | |
| `-v, --verbose` | Enable verbose logging | `false` |
| `-q, --quiet` | Suppress all but error messages | `false` |
| `-r, --recursive` | Recursively search for files in subdirectories | `false` |
| `-l, --listfile <listfile>` | Path to a listfile for FileDataID references | |
| `-c, --export-csv` | Export data to CSV files | `true` |
| `-obj, --export-obj` | Export geometry to Wavefront OBJ files | `true` |
| `-cobj, --export-consolidated-obj` | Export to a single consolidated OBJ file | `true` |
| `-clobj, --export-clustered-obj` | Export to clustered OBJ format | `false` |
| `-eobj, --export-enhanced-obj` | Export to enhanced OBJ format | `false` |
| `-terrain, --extract-terrain` | Extract terrain data from position data | `false` |
| `-map, --generate-map` | Generate 2D map visualization | `false` |
| `--map-resolution <map-resolution>` | Resolution of map image in pixels | `4096` |
| `--detailed-report` | Generate a detailed analysis report | `false` |
| `--bounds <bounds>` | Coordinate bounds for map visualization | `17066.666` |
| `--correlate-special-values` | Generate special values correlation report | `false` |
| `--csv-directory <csv-directory>` | Directory containing CSV files for correlation | |
| `-f, --file <file>` | Path to a specific PM4 file to analyze | |

### UniqueID Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `-l, --listfile <listfile>` | Path to listfile for FileDataID resolution | |
| `-d, --directory <directory>` | Directory containing files to analyze | |
| `-o, --output <output>` | Directory to write analysis output to | |
| `--id-list <id-list>` | Path to file containing list of IDs to analyze | |
| `--reference-report` | Generate cross-reference report | `false` |
| `--update-listfile` | Update listfile with new mappings | `false` |
| `--extract-references` | Extract references to separate files | `false` |

## Architecture

WCAnalyzer is built with a modular architecture consisting of four main components:

### WCAnalyzer.Core
Contains the core functionality and services for processing and analyzing various Warcraft file formats:
- File format parsers (ADT, PM4)
- Chunk processing logic
- Report generation services
- Export utilities
- Data correlation algorithms

### WCAnalyzer.CLI
Command-line interface built with System.CommandLine library:
- Command structure and parsing
- User interaction
- Logging infrastructure
- Process management

### Warcraft.NET
Library for handling Warcraft file formats:
- Binary reading utilities
- Format specifications
- Chunk definitions
- Common structures

### WCAnalyzer.UniqueIdAnalysis
Specialized component for handling unique ID analysis:
- FileDataID resolution
- Reference tracking
- Batch processing utilities
- ID correlation services

## Memory Optimization

WCAnalyzer has been extensively optimized for processing large datasets with minimal memory footprint:

- **Streaming Data Processing**: Files are processed in streams rather than loaded entirely into memory
- **Batch Processing**: Data is processed in configurable batches to control memory usage
- **File Sharing**: Support for reading files that may be open in other applications
- **On-Demand Loading**: Data is loaded only when needed and released immediately afterward
- **Efficient Data Structures**: Specialized collections and data structures for specific use cases
- **Memory Monitoring**: Built-in memory usage monitoring for long-running operations

## Development

### Project Structure

```
WCAnalyzer/
├── WCAnalyzer.Core/           # Core functionality
│   ├── Models/                # Data models
│   ├── Services/              # Processing services
│   └── Utilities/             # Helper utilities
├── WCAnalyzer.CLI/            # Command-line interface
│   ├── Commands/              # Command definitions
│   └── Program.cs             # Entry point
├── Warcraft.NET/              # File format library
│   ├── ADT/                   # ADT format handlers
│   ├── PM4/                   # PM4 format handlers
│   └── IO/                    # I/O utilities
└── WCAnalyzer.UniqueIdAnalysis/ # ID analysis components
    ├── Services/              # ID analysis services
    └── Models/                # ID-related data models
```

### Building and Testing

```bash
# Build the solution
dotnet build

# Run tests
dotnet test

# Create a release build
dotnet build -c Release

# Publish a self-contained executable
dotnet publish -c Release -r win-x64 --self-contained
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Contributors who have helped improve and optimize this toolkit
- The .NET community for excellent libraries and tools
- The Warcraft modding community for ongoing research and documentation of file formats
