# ADT Parser Test Application

A command-line tool for testing the ADT parser and generating analysis reports.

## Overview

This application parses Warcraft ADT (terrain) files and generates detailed reports about their content. It can analyze a single ADT file or all ADT files in a directory, and produces various reports in different formats (TXT, CSV, JSON, and Markdown).

## Features

- Parse a single ADT file or a directory of ADT files
- Validate file references against a listfile
- Generate comprehensive reports in multiple formats:
  - CSV reports for terrain data, heightmaps, and textures
  - JSON reports for all ADT data
  - Markdown reports for easy viewing
  - Text summary reports

## Usage

```
AdtParserTest [options]

Options:
  -f, --file <file>              The path to a single ADT file to parse
  -d, --directory <directory>    Directory containing ADT files to parse
  -l, --listfile <listfile>      Optional path to a listfile for reference validation
  -o, --output <output>          Output directory for reports (required)
  -v, --verbose                  Enable verbose logging
  --help                         Show help information
```

### Examples

Parse a single ADT file:
```
AdtParserTest --file path/to/file.adt --output ./reports
```

Parse all ADT files in a directory:
```
AdtParserTest --directory path/to/adt/files --output ./reports
```

Use a listfile for reference validation:
```
AdtParserTest --file path/to/file.adt --listfile path/to/listfile.txt --output ./reports
```

Enable verbose logging:
```
AdtParserTest --file path/to/file.adt --output ./reports --verbose
```

## Output Reports

The application generates several reports in the specified output directory:

### Text Reports
- `summary.txt` - Overall analysis summary

### CSV Reports
- Height maps, normal vectors, texture layers, and alpha maps

### JSON Reports
- `json/summary.json` - Analysis summary
- `json/results.json` - Detailed ADT analysis results
- `json/heightmaps.json` - Height map data
- `json/texture_layers.json` - Texture layer data

### Markdown Reports
- `markdown/summary.md` - Analysis summary
- `markdown/adt_files.md` - Overview of all ADT files
- `markdown/texture_references.md` - All texture references
- `markdown/model_references.md` - All model references
- `markdown/wmo_references.md` - All WMO references

## Building

This project targets .NET 8.0. To build the application:

```
dotnet build -c Release
```

To run the application:

```
dotnet run -c Release -- [options]
```

## Dependencies

- .NET 8.0
- System.CommandLine - For command-line parsing
- Microsoft.Extensions.Logging - For logging
- Microsoft.Extensions.DependencyInjection - For dependency injection 