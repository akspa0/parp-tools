# PM4 Reader

A C# utility to read and parse PM4 (World of Warcraft physics data) files.

## Features

- Read and parse PM4 file format
- Support for known chunks:
  - MVER (Version chunk)
  - MSPL (Map Splitter chunk)
- Extensible design to add more chunk types
- Graceful handling of unknown chunks
- Simple command-line interface

## Usage

```bash
dotnet run -- <path-to-pm4-file>
```

## Output

The program will display information about the PM4 file, including:
- The file version 
- List of chunks found in the file
- Detailed content for each known chunk type

## Structure

- `PM4/` - Contains classes for the PM4 file format
  - `Chunks/` - Implementation of specific PM4 chunk types
  - `PM4File.cs` - Main class for loading and working with PM4 files
  - `ChunkFactory.cs` - Factory to create appropriate chunk objects
  - `MapSplit.cs` - Struct for map split data
- `Interfaces/` - Contains interface definitions
  - `IPM4Chunk.cs` - Interface for all PM4 chunks

## Requirements

- .NET 6.0 or higher

## Build

```bash
dotnet build
```

## Adding New Chunk Types

To add support for a new chunk type:

1. Create a new class in the `PM4/Chunks/` directory
2. Implement the `IPM4Chunk` interface
3. Add the new chunk type to the `ChunkFactory.CreateChunk` method 