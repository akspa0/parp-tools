# WCAnalyzer

A clean, modern implementation of a World of Warcraft file analyzer tool.

## Overview

WCAnalyzer is designed to parse and analyze various file formats used in World of Warcraft, including:
- ADT (terrain) files
- WMO (world model object) files
- M2 (model) files
- And more to come...

## Features

- Clean, modern .NET 7.0 implementation
- Strong error handling and recovery
- Well-documented code following best practices
- Extensible architecture for adding new file formats
- Command-line interface for easy integration into workflows

## Project Structure

- `WCAnalyzer.Core`: Core library with file parsing and analysis functionality
  - `Common`: Shared interfaces, utilities, and helpers
  - `Files`: File format implementations
    - `ADT`: ADT file format parsers
    - More formats to come...
  - `Services`: Higher-level services for analysis and processing

- `WCAnalyzer.CLI`: Command-line interface for the analyzer

## Getting Started

### Prerequisites

- .NET 7.0 SDK or later

### Building the Project

```bash
cd NewImplementation
dotnet build
```

### Running the CLI

```bash
cd NewImplementation
dotnet run --project WCAnalyzer.CLI/WCAnalyzer.CLI.csproj -- adt --file path/to/your.adt
```

## Architecture

The project follows a clean architecture with:

1. **Interfaces**: Clear contracts for components
2. **Base Classes**: Common functionality shared across implementations
3. **Specific Implementations**: Concrete classes for each file format
4. **Services**: Higher-level functionality that consumes the lower-level implementations

### Key Concepts

- `IChunk`: Interface for all data chunks in various file formats
- `IBinarySerializable`: Interface for types that can be serialized to/from binary data
- `IFileParser`: Interface for parsers that can read entire files
- `BaseChunk`: Common functionality for all chunk types
- `ChunkFactory`: Factory for creating chunk objects based on signatures

## License

MIT License

## Acknowledgements

- Based on documentation of World of Warcraft file formats 