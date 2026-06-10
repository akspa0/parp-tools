# Implementation Lessons and Discoveries

This document captures key findings, unexpected behaviors, and lessons learned during the implementation and testing of parsers. It serves as a reference for future development to avoid rediscovering the same issues or insights.

## Alpha WDT Format Discoveries

### File Structure (2023-03-28)
- Alpha WDT files appear to be a hybrid of modern WDT and ADT formats
- The Azeroth.wdt file (~752MB) is much larger than typical modern WDT files, suggesting it contains actual terrain data
- Found ADT-style chunks embedded directly in the WDT:
  - MHDR (header) chunks - normally found in ADT files
  - MCIN (chunk index) chunks - normally used in ADT files to locate MCNK chunks
  - MTEX (texture list) chunks - normally contain texture references in ADT files
  - MDDF (doodad placement) chunks - normally contain doodad placement in ADT files

### Chunk Detection (2023-03-28)
- MCNK chunks appear in large numbers (hundreds per file), suggesting each ADT's terrain data is embedded directly
- The Alpha format seems to consolidate what would later be split across multiple files into a single file
- Chunk signatures are stored in little-endian format (e.g., "REVM" for MVER, "DHPM" for MPHD)

### Flag Interpretation (2023-03-28)
- Some header flags (RawFlags: 3090) indicate values not documented in the modern format
- Tile flags show interesting combinations - some tiles have both HasADT=true and IsAllWater=true

### Performance Considerations (2023-03-28)
- Parsing the entire Alpha WDT file can be memory-intensive due to its large size
- For practical applications, implementing partial loading or streaming might be necessary
- The YAML serialization of a full file is unwieldy; consider more focused output formats

### MCNK Implementation (2023-03-29)
- Implemented parsing for key MCNK sub-chunks (MCVT, MCNR, MCLY, MCAL, MCCV, MCLQ, MCSH)
- MCNK chunks are organized in a dictionary keyed by their X,Y coordinates
- Each coordinate can contain multiple MCNK chunks
- Used a 64-byte estimation for the MCNK header size; may need adjustment based on testing
- Encountered challenges with alpha map handling - currently using simplified approach

### MCNK Header Structure (2023-03-29)
- Updated MCNK parsing to use the correct 128-byte header size specified in Alpha.md documentation
- The MCNK header in Alpha format contains fields for flags, coordinates, layer counts, and offsets to sub-chunks
- Sub-chunks don't have standard chunk headers with names and sizes in the Alpha format

### ADT-Style Chunks Implementation (2023-03-30)
- Added support for ADT-style chunks embedded directly in the Alpha WDT files
- Implemented parsing for MHDR, MCIN, MTEX, and MDDF chunks
- Created corresponding model classes that mirror the structures found in the Alpha format
- The MHDR chunk serves as a container header with offsets to other chunks, similar to modern ADT files
- Each MCIN chunk contains 256 entries (16x16 grid) pointing to MCNK chunks
- The MTEX chunk contains a list of null-terminated strings with texture filenames
- The MDDF chunk contains doodad placement information
- Challenges remain in accurately matching these chunks to their corresponding map tiles
- The current implementation uses a simple heuristic based on the HasAdt flag in tile entries

### Integration with Existing MCNK Parsing (2023-03-30)
- Combined the ADT-style chunk parsing with the existing MCNK chunk parsing
- Added an important MDNM chunk parser to extract model filenames referenced by MDDF entries
- Organized the parsed data into dictionaries keyed by tile coordinates (X,Y) for easy access
- Enhanced the AlphaWDTFile model with additional properties for the new data types
- Refined the chunk signature constants to include ADT-style chunks

### Memory Optimization for Large Files (2023-03-31)
- Implemented a streaming approach to reduce memory usage when parsing large WDT files
- Created a `StreamingYamlWriter` class that writes data incrementally without keeping everything in memory
- Added a `TileFilterOptions` class to allow processing only specific map tiles or ranges
- Implemented tile-based processing that parses and outputs one tile at a time
- Added garbage collection after processing each tile to free memory
- Implemented a chunked approach for MCNK data that groups chunks by their parent tile
- Added command-line options for controlling which chunks to parse and how many to process
- The parser can now handle files with 175,000+ MCNK chunks that previously caused out-of-memory errors
- Streaming output for Azeroth.wdt reduced memory usage from several gigabytes to less than 1GB
- Added summary statistics to replace full data dumps for large collections
- Included a warning in the standard parser when a file contains too many MCNK chunks

### Hierarchical YAML Structure (2024-04-01)
- Decided on a hierarchical YAML output structure to improve readability and editability
- Designed a file organization that separates metadata from content
- Created a directory-based approach to organize output by tile coordinates
- Separated resources (textures, models) from their placement information
- Implemented referencing between files to maintain relationships without duplication
- Added an index file to provide a map of all contained files and their relationships
- Goals were to prioritize human readability, future-proofing, and easy editing
- Structure addresses the challenge of processing multi-gigabyte decoded data
- Provides a way to edit specific parts without loading entire dataset

### Phased Parsing Approach (2024-04-01)
- Implemented a three-phase parsing approach:
  1. Parse metadata chunks (MPHD, MAIN, headers)
  2. Build relationship graph between chunks
  3. Extract content chunks with proper context
- Used offset-based seeking for critical chunks
- Optimized memory usage with streaming and selective chunk decoding
- Improved robustness with validation before writing
- Implemented transaction-like behavior for updates
- Created clear separation of metadata from content
- Developed a system that can handle 14GB+ of decoded data with reasonable memory usage
- Approach inspired by database-like structure of the original files

## General Parser Implementation Lessons

### Chunk Processing (2023-03-28)
- Using a switch statement for chunk signatures provides clean extensibility for new chunk types
- Storing chunk signatures as constants improves code readability
- Important to check chunk bounds before accessing data to prevent buffer overruns

### Error Handling (2023-03-28)
- Warning collection in the result object is useful for flagging non-fatal issues during parsing
- Using try/catch blocks around the entire parsing operation preserves partial results even if errors occur

### Model Design (2023-03-28)
- Implementing flag interpretation methods in the model classes keeps the parser code cleaner
- Using nullable types for optional components helps represent the file structure more accurately
- Dictionary usage for map tiles provides better access patterns than nested arrays 

### Sub-Chunk Parsing (2023-03-29)
- Each sub-chunk requires specialized parsing based on its format
- For height maps (MCVT) and normals (MCNR), careful counting of vertices is critical (145 = 9×9 + 8×8)
- Converting compressed normals requires normalization from byte representation to float values
- Alpha maps may be compressed, requiring specific decompression logic (placeholder implemented)
- Texture layers need to be processed before alpha maps to properly organize the data

### Alpha Format vs Modern Format (2023-03-29)
- Alpha format primarily uses embedded sub-chunks without their own headers
- Modern formats typically use properly structured sub-chunks with name/size
- Alpha format uses offsets relative to the end of the MCNK header
- Some fields have different sizes or interpretations in Alpha format
- Organization of vertices in Alpha format is different - outer vertices first, then inner vertices

### Data Serialization (2023-03-29)
- Floating point values should be displayed in standard notation rather than scientific notation
- Using scientific notation (e.g., 1.23e-5) can hide the full precision of float values
- Implemented a custom YAML serializer to ensure floats are displayed with full precision
- The custom serializer uses format specifier "0.0#################" and CultureInfo.InvariantCulture
- This preserves exact values from the original binary file for more accurate analysis 

### Alpha WDT Chunk Parsing Strategy (YYYY-MM-DD)
- Initial attempts to parse Alpha WDT chunks purely sequentially failed due to missing key chunks (MPHD, MAIN, MDNM, MONM).
- Discovered through comparison with the `gillijimproject` reference implementation that the `MPHD` chunk contains crucial absolute file offsets for locating `MDNM`, `MONM`, and `MODF` chunks.
- Simple sequential parsing is insufficient; the file structure does not guarantee these chunks appear in a predictable order after `MAIN`.
- Refactored `AlphaWDTParser.cs` to use a two-phase approach:
  1. **Phase 1:** Scan sequentially to find and parse only `MVER`, `MPHD`, and `MAIN`.
  2. **Phase 2:** Use the offsets obtained from the parsed `MPHD` header to explicitly seek and parse `MDNM`, `MONM`, and `MODF` chunks.
- This offset-based approach mirrors the `gillijimproject` logic and is necessary for robustly handling the Alpha WDT format.
- Build errors encountered during refactoring highlighted mismatches between updated models (`AlphaWDTFile`, `AlphaWDTHeader`) and code still using the old model structure (in writers and other parser sections). Emphasizes the need to update *all* usages when models change. 

## ADT Alpha Map Handling (2024-03-31)

### Key Findings
1. **Alpha Map Formats**: ADT files support three distinct alpha map formats:
   - Standard uncompressed (2048 bytes)
   - Big alpha uncompressed (4096 bytes)
   - Compressed with run-length encoding

2. **Format Detection**:
   - Format is determined by flags in the MCLY chunk entry
   - `MCLYFlags.UseAlpha` indicates presence of alpha data
   - `MCLYFlags.UseBigAlpha` indicates 4096-byte format
   - `MCLYFlags.CompressedAlpha` indicates RLE compression

3. **Alpha Map Dimensions**:
   - All alpha maps use a 64x64 grid (4096 pixels)
   - Standard format packs two 4-bit values per byte
   - Big alpha format uses one byte per pixel
   - Compressed format uses RLE to reduce size

4. **Error Handling**:
   - Missing alpha maps should be replaced with empty 64x64 grids
   - Invalid sizes should trigger warnings but not fail parsing
   - Compression errors should fall back to empty maps
   - All errors should be logged but allow parsing to continue

5. **Implementation Notes**:
   - Alpha maps are stored per texture layer
   - Each MCNK can have multiple texture layers
   - Alpha maps define texture blending between layers
   - Last row/column should be duplicated from second-to-last

### Best Practices
1. Always validate alpha map dimensions (expect 4096 pixels)
2. Handle missing or corrupt data gracefully
3. Implement proper debug logging for troubleshooting
4. Use empty alpha maps (filled with zeros) as fallback
5. Check texture layer count against available alpha maps

### Common Issues
1. Texture layer count mismatches
2. Corrupted compressed data
3. Missing alpha maps for layers that require them
4. Invalid alpha map offsets
5. Incorrect handling of the last row/column duplication

### Performance Considerations
1. Pre-allocate alpha map arrays to avoid resizing
2. Use array pooling for temporary buffers if processing many chunks
3. Consider lazy loading of alpha maps for large ADT files
4. Cache decompressed results if the same alpha map is referenced multiple times 

## YAML Output Strategy (2024-04-01)

### YAML Structure Design
1. **Organization Principles**:
   - Hierarchical organization by map > tile > chunk
   - Separation of metadata from content
   - References between files to maintain relationships
   - Human-readable format prioritized

2. **Key Design Goals**:
   - Future-proof format (readable without specialized tools)
   - Easy manual editing capability
   - Version control friendly
   - Minimal data duplication
   - Organized structure for browsing

3. **Memory Considerations**:
   - Progressive writing to avoid full in-memory representation
   - Chunked output for large datasets
   - Selective loading of only needed components
   - Metadata/content separation to enable partial loading

### File Format Design
1. **Directory Structure**:
   - Map-level directory containing all related files
   - Tile-level directories organized by coordinates
   - Resource directories for shared assets
   - Optional raw chunk storage for preservation

2. **File Types**:
   - Index files to map relationships
   - Metadata files for properties and settings
   - Content files for actual terrain/texture data
   - Reference files for shared resources

3. **Implementation Approach**:
   - Transaction-like writing (temp files → validation → swap)
   - Integrity checks before finalization
   - Progressive processing to manage memory usage
   - Clear schema documentation for future readers

4. **Benefits Over Binary Format**:
   - Human inspection and modification
   - Corruption resistance (text vs binary)
   - Version control differencing
   - Long-term accessibility
   - No special tools required for basic access 

## WoW's Database Origins and YAML Storage Strategy

### Historical Context
- WoW evolved from Warcraft 3's object-oriented engine, which itself was an OOP reimagining of Warcraft 2
- Early WoW (Alpha) used monolithic WDT files that were massive (700MB+ for Azeroth)
- These files were likely SQL database dumps, with each map being a separate database and chunks as tables
- This explains the database-like structure still visible in modern WoW files (DBCs, ADTs, etc.)
- Version 18 format was established in Alpha and persists today, with changes coming via new chunks/structures

### Memory and Loading Considerations
- Alpha-era computers had 64-128MB RAM, far too small for 700MB+ map files
- Suggests early WoW used database-style paging/loading
- Split into WDT+ADT format occurred in v0.6.0 (first public beta)
- This split enabled more efficient streaming and memory management
- Modern chunk-based structure still reflects these database origins

### Why YAML Storage
1. **Return to Original Form**
   - Breaking binary files back into logical database-like components
   - More closely mirrors how data was likely managed internally at Blizzard
   - Clear separation of different data types (terrain, textures, objects, etc.)

2. **Data Protection**
   - Prevents loss of work from map editor corruption
   - Each component saved separately
   - Easy to version control
   - Simple to backup and restore specific changes

3. **Accessibility**
   - Human-readable format
   - Standard text editors can be used
   - No specialized tools required
   - Clear structure and relationships

4. **Portability**
   - Engine-agnostic storage
   - Can be imported into other engines (Unreal, Unity, etc.)
   - Artistic work preserved in universal format
   - Not locked to WoW's binary formats

5. **Modern Workflow**
   - Git-friendly
   - Easy to diff and merge
   - Supports collaboration
   - Can validate before converting to binary

### Implementation Philosophy
- Treat WoW data like the database it originally was
- Use clean, standard formats for storage
- Make data manipulation explicit and traceable
- Avoid "black box" transformations
- Enable proper asset pipeline workflow
- Preserve artistic work in durable, portable formats

This understanding fundamentally shapes our approach to the toolbox - we're not just creating another binary editor, we're providing proper database tools for WoW modding that protect and preserve creative work. 