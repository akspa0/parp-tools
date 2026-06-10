# Gillijim Project – C# Refactor (net9.0)

A complete 1:1 C# port of the original C++ gillijimproject library for parsing Alpha WDT files.

## Features

### Implemented
- **Streaming IO Architecture**: FileStream + BinaryReader throughout, no byte[] file-wide buffers
- **Alpha WDT Container Handling**: Tolerant MAIN discovery with 4-byte aligned scanning
- **Complete Chunk Pipeline**: MAIN → MHDR → MCIN → MCNK with Alpha header-driven offsets
- **Terrain Subchunks**: MCVT, MCNR, MCCV, MCLQ, MCSE, MCBB parsers
- **World Object Chunks**: MMDX/MMID, MWMO/MWID, MDDF/MODF parsers
- **WDL Support**: Basic Alpha WDL file handling
- **Enhanced CLI**: Full diagnostic options with streaming, bounded processing

### Core Architecture
- **Streaming First**: All APIs accept `(Stream s, long offset)` for large file efficiency
- **Alpha Semantics**: Preserves exact binary semantics, offsets, and bounds from C++ reference
- **Bounds Validation**: Strict subchunk bounds checking within MCNK chunksSize limits
- **Error Handling**: Precise error messages with offsets/tags for debugging

## Usage

### Basic WDT Analysis
```bash
GillijimProject dump-wdt Kalimdor.wdt
```

### Advanced Diagnostics
```bash
# Process specific tile range with verbose output
GillijimProject dump-wdt Kalimdor.wdt --start 5 --tiles 3 --verbose

# Process all tiles and MCNKs (no limits)
GillijimProject dump-wdt Kalimdor.wdt --all --verbose

# Limit MCNK processing per tile
GillijimProject dump-wdt Kalimdor.wdt --mcnk 10 --verbose
```

### CLI Options
- `--start N` - Start processing from tile N (default: 0)
- `--tiles K` - Process maximum K tiles (default: 10)
- `--mcnk M` - Process maximum M MCNKs per tile (default: 5)
- `--verbose` - Show detailed offset/bounds information
- `--all` - Process all tiles and MCNKs (overrides limits)

## Architecture

### Directory Structure
```
WowFiles/
├── Chunk.cs              # Core chunk header reading
├── ChunkHeaders.cs       # FourCC tag constants
├── WowChunkedFormat.cs   # MAIN discovery & top-level enumeration
├── Wdt.cs               # Main WDT orchestrator
├── Main.cs              # Alpha MAIN parser (16-byte stride)
├── Mhdr.cs              # Alpha MHDR parser
├── Mcin.cs              # Alpha MCIN parser (256 entries)
├── Mcnk.cs              # Alpha MCNK header & subchunk validation
├── Terrain/             # Terrain subchunk parsers
│   ├── Mcvt.cs         # Height vertices (145 * 4 = 580 bytes)
│   ├── Mcnr.cs         # Normal vectors (145 * 3 = 435 bytes)
│   ├── Mccv.cs         # Vertex colors (145 * 4 = 580 bytes)
│   ├── Mclq.cs         # Liquid data (variable size, Alpha fallback)
│   ├── Mcse.cs         # Sound emitters (16 bytes per entry)
│   └── Mcbb.cs         # Bounding boxes (24 bytes per entry)
├── Objects/             # World object parsers
│   ├── Mmdx.cs         # M2 model filenames (null-terminated strings)
│   ├── Mmid.cs         # M2 model indices (4 bytes per index)
│   ├── Mwmo.cs         # WMO model filenames (null-terminated strings)
│   ├── Mwid.cs         # WMO indices (4 bytes per index)
│   ├── Mddf.cs         # M2 placement data (36 bytes per entry)
│   └── Modf.cs         # WMO placement data (64 bytes per entry)
└── Wdl/
    └── Wdl.cs           # Basic WDL support
```

### Key Semantic Preservation
- **MAIN**: 16-byte stride entries, tile presence by `Offset != 0`
- **MHDR**: First uint = MCIN offset relative to MHDR payload base
- **MCIN**: 256 entries, absolute MCNK offset = `tileStart + ofsRel`
- **MCNK**: Alpha header size = 0x80, strict subchunk bounds validation
- **MCVT**: Exactly 580 bytes (145 * 4), **MCNR**: Exactly 435 bytes (145 * 3)

## Validation

Tested against real Alpha 0.5.3 data:
- `test_data/0.5.3/alphawdt/Kalimdor.wdt` (1GB+ file)
- Streaming IO handles multi-GB files efficiently
- MAIN discovery with tolerant scanning
- All chunk parsers validate bounds correctly
- Terrain subchunks parse with expected sizes
- World object chunks handle variable-length data

## Building

```bash
dotnet build
dotnet run -- dump-wdt path/to/alpha.wdt --verbose
```

## Dependencies

- .NET 9.0
- No external dependencies (pure .NET implementation)

## Implementation Notes

This is a **complete 1:1 port** of the original C++ gillijimproject library, preserving:
- Binary semantics and offset calculations
- Alpha-specific header structures and sizes  
- Streaming IO patterns for large file handling
- Error handling with precise diagnostic information

The implementation prioritizes correctness and Alpha format fidelity over performance optimizations.
