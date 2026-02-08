# DBC Texturing Implementation Documentation

## Overview

This documentation provides comprehensive coverage of the DBC (Database Client) texturing implementation within the MDX Viewer application. The system handles loading, parsing, and applying texture references from DBC files to 3D models in the MDX format.

## Table of Contents

1. [Introduction](01_introduction.md)
2. [DBC File Structure](02_dbc_file_structure.md)
3. [Texture Loading Workflow](03_texture_loading_workflow.md)
4. [Database Schema](04_database_schema.md)
5. [Texture Path Resolution](05_texture_path_resolution.md)
6. [Texture Variations and Skins](06_texture_variations.md)
7. [Optimization Techniques](07_optimization.md)
8. [Error Handling](08_error_handling.md)
9. [MDX Rendering Integration](09_mdx_integration.md)
10. [Texture Formats](10_texture_formats.md)
11. [API Reference](11_api_reference.md)
12. [Code Examples](12_code_examples.md)
13. [Troubleshooting Guide](13_troubleshooting.md)
14. [Performance Benchmarks](14_performance.md)
15. [Version Compatibility](15_compatibility.md)

## Quick Start

For developers new to the DBC texturing system, start with:
1. [Introduction](01_introduction.md) - System overview and architecture
2. [Texture Loading Workflow](03_texture_loading_workflow.md) - Step-by-step process
3. [Code Examples](12_code_examples.md) - Practical implementation examples

## Key Components

The DBC texturing system consists of several core components:

- **DBC Cache System**: Manages in-memory database records
- **Texture Hash System**: Efficient texture lookup and caching
- **MDX Reader**: Processes texture references from model files
- **Texture Loader**: Handles texture file loading and format conversion
- **Rendering Pipeline**: Integrates textures with 3D model rendering

## System Architecture

```
┌─────────────────┐
│   DBC Files     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  DBC Cache      │────▶│  Texture Lookup  │
│  System         │     │  (Hash Tables)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  MDX Reader     │────▶│  Texture Loader  │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌────────────────────┐
         │ Rendering Pipeline │
         └────────────────────┘
```

## Related Systems

The DBC texturing system interfaces with:
- **CreatureDisplayInfo DBC**: Model and texture references for creatures
- **Item DBC**: Item model and texture data
- **CharTextures DBC**: Character customization textures
- **BLP Texture System**: Blizzard's proprietary texture format

## Technical Requirements

- WoW Client version 0.5.3-0.5.5
- Support for DBC format version 1
- BLP texture format support (BLP0 and BLP1)
- Memory-efficient texture caching
- Multi-threaded texture loading support

## License and Attribution

This documentation is based on reverse engineering analysis of World of Warcraft client binaries using Ghidra. All information is provided for educational and interoperability purposes.

---

**Last Updated**: 2026-02-08  
**Version**: 1.0  
**Analyzed Binary**: WoWClient.exe 0.5.3
