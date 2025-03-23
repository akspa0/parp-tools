# WCAnalyzer

A .NET 8.0 command-line tool for analyzing World of Warcraft ADT (terrain) files. The tool processes ADT files and their associated _obj0 and _tex0 components, generating comprehensive analysis reports.

## Requirements

- .NET 8.0 SDK
- SQLite (System.Data.SQLite.Core 1.0.118)
- Microsoft.Extensions.Logging.Console 8.0.0
- Microsoft.Extensions.Configuration.Json 8.0.0

## Building

```bash
dotnet build
```

## Usage

```bash
dotnet run --project WCAnalyzer.CLI -- --directory <input_directory> [options]
```

### Required Arguments

- `--directory`: Directory containing ADT files to analyze

### Optional Arguments

- `--output` or `-o`: Output directory for analysis results. Results will be saved in a timestamped subfolder (e.g., `run_20240321_123456`)
- `--listfile` or `-l`: Path to a listfile containing known file paths and FileDataIDs in format: `<FileDataID>;<asset path>`
- `--verbose`: Enable verbose logging

### Example

```bash
dotnet run --project WCAnalyzer.CLI -- --directory "./wow/adt/files" --output "./analysis" --listfile "./listfile.csv"
```

## Output Structure

When an output directory is specified, the tool generates:

### Per-ADT Analysis
- JSON files containing detailed analysis of each ADT unit
- Includes terrain chunks, texture references, model placements, and WMO data

### Summary Files
- `summary.txt`: Overall analysis results including:
  - Analysis timestamp
  - Total ADT units processed
  - Success/failure counts
  - Reference totals (textures, models, WMOs)
  - Processing duration

### Generated Reports
- CSV reports for terrain data
- Markdown reports for analysis summaries
- JSON reports for detailed data export

## Features

- Processes and consolidates data from related ADT files (base, _obj0, _tex0)
- Validates file references against a provided listfile
- Generates detailed analysis of:
  - Terrain chunks and height data
  - Texture layers and references
  - Model placements and references
  - WMO (World Model Object) placements and references
- Provides formatted JSON output with controlled precision for numeric values
- Supports batch processing with detailed logging

## Project Structure

- `WCAnalyzer.CLI`: Command-line interface and program logic
- `WCAnalyzer.Core`: Core analysis functionality and models (referenced project)

## Dependencies

```xml
<ItemGroup>
    <PackageReference Include="Microsoft.Extensions.Logging.Console" Version="8.0.0" />
    <PackageReference Include="Microsoft.Extensions.Configuration.Json" Version="8.0.0" />
    <PackageReference Include="System.Data.SQLite.Core" Version="1.0.118" />
</ItemGroup>
```
