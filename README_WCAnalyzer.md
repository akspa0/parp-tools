# WCAnalyzer

WCAnalyzer is a powerful .NET 8.0 tool designed for analyzing and extracting data from World of Warcraft file formats, with a particular focus on PM4 files and ADT terrain files containing mesh, terrain, and position data.

## Features

- **PM4 File Analysis**: Parse and extract data from PM4 files, including vertices, triangles, and position records
- **ADT Terrain Parsing**: Extract and analyze terrain data from ADT (Alpha/Development Terrain) files
- **3D Model Export**: Convert PM4 and ADT mesh data to industry-standard Wavefront OBJ format
- **Terrain Reconstruction**: Extract and visualize terrain data from position records
- **Advanced Clustering**: Organize complex 3D models into meaningful groups based on vertex proximity
- **CSV Reporting**: Generate detailed reports of file contents for further analysis
- **Batch Processing**: Process multiple files recursively with parallel execution
- **Listfile Support**: Use community listfiles to enhance file identification

## Installation

### Prerequisites

- .NET 8.0 SDK or later
- Windows, macOS, or Linux operating system

### Building from Source

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/WCAnalyzer.git
   cd WCAnalyzer
   ```

2. Build the project:
   ```
   dotnet build -c Release
   ```

3. Run the CLI tool:
   ```
   dotnet run --project WCAnalyzer.CLI/WCAnalyzer.CLI.csproj -- --help
   ```

## Usage

### Command Line Interface

WCAnalyzer offers a comprehensive CLI for various analysis tasks:

```
dotnet run --project WCAnalyzer.CLI/WCAnalyzer.CLI.csproj -- [command] [options]
```

### PM4 Analysis

Analyze PM4 files and extract their contents:

```
dotnet run -- pm4 --directory <input_directory> --output <output_directory> [options]
```

#### Options:

- `--directory`, `-d`: Source directory containing PM4 files
- `--output`, `-o`: Output directory for reports (default: ./output)
- `--recursive`, `-r`: Process files in subdirectories
- `--verbose`, `-v`: Enable verbose logging
- `--quiet`, `-q`: Suppress non-error output
- `--listfile`: Path to a listfile to use for file identification
- `--export-csv`: Export data to CSV reports
- `--export-obj`: Export mesh data to OBJ format
- `--export-consolidated-obj`: Export mesh data to consolidated OBJ format
- `--export-clustered-obj`, `-clobj`: Export mesh with vertex proximity clustering
- `--extract-terrain`, `-terrain`: Extract and export terrain data from position records

### ADT Analysis

Analyze ADT terrain files and extract height maps, textures, and other terrain data:

```
dotnet run -- adt --directory <input_directory> --output <output_directory> [options]
```

#### Options:

- `--directory`, `-d`: Source directory containing ADT files
- `--output`, `-o`: Output directory for reports (default: ./output)
- `--recursive`, `-r`: Process files in subdirectories
- `--verbose`, `-v`: Enable verbose logging
- `--quiet`, `-q`: Suppress non-error output
- `--export-obj`: Export terrain mesh to OBJ format
- `--export-height-map`: Export height map data as grayscale images
- `--export-csv`: Export terrain data to CSV reports

### Examples

Extract and analyze PM4 files with terrain reconstruction:

```
dotnet run -- pm4 -d "C:\WoW\Data" -o "C:\WoW\Analysis" -r --listfile listfile.csv --extract-terrain
```

Generate comprehensive reports with 3D models:

```
dotnet run -- pm4 -d "C:\WoW\Data" -o "C:\WoW\Analysis" --export-csv --export-obj --verbose
```

Extract and analyze ADT terrain files:

```
dotnet run -- adt -d "C:\WoW\Data\World\Maps" -o "C:\WoW\Analysis" --export-obj --export-height-map
```

## Project Architecture

WCAnalyzer is built using a modular architecture following .NET best practices:

### Core Components

- **WCAnalyzer.Core**: Core library containing models, parsers, and services
  - `Models`: Data structures representing WoW file formats
  - `Services`: Business logic for parsing and processing files
  - `Utilities`: Helper functions and extensions

- **WCAnalyzer.CLI**: Command-line interface for the analyzer
  - Built using System.CommandLine for modern CLI capabilities
  - Provides rich command structure with help documentation

- **Warcraft.NET**: Supporting library for general Warcraft file formats

### Key Services

- **PM4Parser**: Parses PM4 file structures into object models
- **PM4ObjExporter**: Exports PM4 data to Wavefront OBJ format
- **PM4ClusteredObjExporter**: Creates optimized OBJ files with vertex clustering
- **PM4CsvGenerator**: Produces detailed CSV reports of file contents
- **ADTParser**: Parses ADT terrain file structures into object models
- **ADTObjExporter**: Exports ADT terrain data to Wavefront OBJ format
- **ADTHeightMapGenerator**: Creates height maps from ADT terrain data
- **VertexClusteringService**: Implements K-means clustering for 3D mesh data

## File Formats

### PM4 Files

PM4 files are a proprietary World of Warcraft format used for storing terrain, mesh, and position data. The files follow a chunked structure:

- **MVER**: Version chunk
- **GOBJ**: Generic object data
- **POSI**: Position records for terrain and objects
- **VERT**: Vertex position data
- **VIDX**: Vertex indices for triangles
- **NORM**: Normal vectors
- **TEXC**: Texture coordinates
- **MTLS**: Material references

### ADT Files

ADT (Alpha/Development Terrain) files are World of Warcraft's terrain format used for storing landscape data. The files follow a chunked structure:

- **MVER**: Version chunk
- **MHDR**: Header information
- **MCIN**: Cell information
- **MTEX**: Texture filenames
- **MMDX**: Model filenames
- **MMID**: Model instance IDs
- **MCNK**: Map chunks containing:
  - Height map data
  - Texture layers
  - Shadow maps
  - Alpha maps
  - Model instances
  - Liquid data

### Output Formats

- **OBJ**: Industry-standard 3D model format supported by most 3D applications
- **CSV**: Comma-separated values for data analysis in spreadsheet applications

## Advanced Usage

### Terrain Reconstruction

The terrain reconstruction feature creates a 3D mesh from position data:

```
dotnet run -- pm4 -d <input_directory> -o <output_directory> --extract-terrain
```

This will generate a `terrain_reconstruction.obj` file in the output directory, which can be opened in any 3D modeling application.

### ADT Terrain Analysis

The ADT parsing feature allows you to extract and visualize World of Warcraft terrain:

```
dotnet run -- adt -d <input_directory> -o <output_directory> --export-obj --export-height-map
```

This will generate:
- OBJ files containing 3D terrain meshes
- Height map images showing terrain elevation
- Texture blend maps showing texture distribution

### Vertex Clustering

For complex models, vertex clustering can create more meaningful object groups:

```
dotnet run -- pm4 -d <input_directory> -o <output_directory> -clobj
```

The algorithm uses K-means clustering to group vertices based on spatial proximity.

## Contributing

Contributions to WCAnalyzer are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- The WoW community for their extensive research on file formats
- Contributors to the community listfile project 