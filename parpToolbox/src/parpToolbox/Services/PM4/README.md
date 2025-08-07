# PM4 MSUR Grouping Tool

## Overview
This tool provides optimized grouping of PM4 geometry using MSUR raw fields to produce coherent 3D object assemblies. After extensive testing of multiple grouping strategies, the MSUR raw fields approach consistently produces the most meaningful object grouping results, especially when dealing with fragmented tile data.

## How It Works
The tool analyzes MSUR chunk data within PM4 files and groups geometry based on raw field values. This approach has proven most effective for reconstructing complete objects from the PM4 format, creating horizontal object slices that maintain proper structural integrity.

### Key Features
- **MSUR Raw Fields Grouping**: Groups objects based on MSUR field values for optimal object assembly
- **Object-Based OBJ Export**: Outputs separate OBJ files for each distinct object group
- **Vertex Deduplication**: Eliminates redundant vertices for cleaner geometry
- **Real-Time Progress Reporting**: Provides detailed console output during processing

## Technical Details

### Grouping Algorithm
The algorithm uses two key MSUR fields as the primary grouping criteria:
1. `FlagsOrUnknown_0x00` - Appears to define object type or category
2. `Unknown_0x02` - Further subdivides objects into logical sections

These fields provide the most semantically meaningful segmentation of PM4 geometry, creating horizontal slices that correspond to actual in-game object boundaries.

### Known Limitations
- Cannot resolve cross-tile vertex references (requires global tile loading)
- Approximately 64% data loss due to out-of-bounds vertex indices
- Some objects may still be fragmented due to missing vertices from adjacent tiles

## Usage
The tool is integrated into parpToolbox as the `pm4-test-grouping` command:

```
parpToolbox pm4-test-grouping <input_file.pm4> [--no-faces]
```

### Arguments
- `<input_file.pm4>`: Path to the PM4 file to process
- `--no-faces`: Optional flag to export only vertices without faces (for debug purposes)

### Output
All output is written to a timestamped directory under the `project_output` folder using the naming pattern:
```
project_output/<input_file>_msur_grouping_<timestamp>/
```

The output directory contains:
- Multiple OBJ files (one per distinct object group)
- A basic MTL material file

## Future Improvements
- Global tile loading to resolve cross-tile vertex references
- Merging of split object components based on spatial adjacency
- Further refinement of the MSUR field interpretation

## Example
```
parpToolbox pm4-test-grouping development_22_18.pm4
```

This command will process the `development_22_18.pm4` file and output grouped objects to the project_output directory.
