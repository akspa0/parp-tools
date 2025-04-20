# WoWToolbox v3

## Project Overview
A comprehensive toolkit for parsing and working with World of Warcraft ADT (terrain) files and related data formats. Built in C# targeting .NET 8.0, this project provides libraries for chunk decoding and manipulation of complex packed data, along with specialized tools for analysis and data extraction.

## Core Goals
1. Create efficient and reliable chunk decoders for WoW ADT files
2. Build reusable libraries for each numbered chunk type
3. Develop a foundation including specialized tools for future parsers and exporters
4. Maintain compatibility with modern WoW file formats

## Core Strategy
1. Reinforcement-Based Programming Approach
   - Build upon Warcraft.NET's modern format implementations
   - Focus on backward compatibility for older formats
   - Create conversion pipelines from legacy to modern formats
   - Regular audits against chunkvault documentation

2. Integration Strategy
   - Use Warcraft.NET as the primary parser for modern formats
   - Extend functionality for legacy format support
   - Create bridges between old and new format handlers
   - Maintain compatibility with Warcraft.NET's architecture

## Audit Framework
1. Documentation Compliance
   - Regular checks against chunkvault specifications
   - Version compatibility verification
   - Format evolution tracking

2. Code Quality
   - Integration tests with Warcraft.NET
   - Cross-version format validation
   - Performance benchmarking

## Dependencies
- Warcraft.NET (https://github.com/ModernWoWTools/Warcraft.NET)
- DBCD (https://github.com/wowdev/DBCD)

## Project Structure
- `/src/lib/` - External dependencies (Warcraft.NET, DBCD)
- `/chunkvault/` - Documentation and specifications for chunk formats
- `/src/` - Source code for chunk decoders and libraries 

## Recent Developments (2024-07-21)
- Added a new test for mesh extraction and MSCN boundary output, which writes OBJ and diagnostics files for key PM4 files.
- All build errors related to type mismatches have been resolved.
- Current focus is on ensuring robust test automation and resource management, as a recent process hang after file output highlighted the importance of proper cleanup and test completion.
- Implemented direct parsing and mesh assembly for v14 WMO group chunks, enabling geometry extraction from raw chunk data (MOVT, MONR, MOTV, MOPY, MOVI, etc.) instead of relying on legacy group file parsing. This breakthrough supports legacy formats that do not store explicit mesh data. 