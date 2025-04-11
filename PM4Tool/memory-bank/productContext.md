# Product Context

## Problem Space
World of Warcraft's terrain files (ADT) use a complex chunked format that requires specialized tools for parsing and manipulation. These files contain various data types including terrain height maps, textures, model placements, and more.

## Current Solution
WoWToolbox provides:

1. Legacy Format Support
   - Version-aware chunk parsing
   - Format conversion pipeline
   - Backward compatibility handling
   - Modern format integration

2. Validation Framework
   - Chunkvault specification compliance
   - Field-level validation
   - Size and version validation
   - Relationship validation

3. Documentation Integration
   - Markdown specification parsing
   - Automated validation
   - Relationship tracking
   - Version compatibility checking

4. Core Infrastructure
   - Extension methods for chunk handling
   - Version conversion system
   - Legacy format detection
   - Stream-based processing

## Target Users
1. WoW Tool Developers
   - Format conversion needs
   - Legacy data handling
   - Modern format integration
   - Validation requirements

2. Map Editors and Creators
   - Terrain data manipulation
   - Model placement
   - Texture handling
   - Format validation

3. Data Miners and Researchers
   - Format analysis
   - Version tracking
   - Relationship mapping
   - Documentation reference

4. Addon Developers
   - Terrain data access
   - Format compatibility
   - Version handling
   - Data validation

## Use Cases

1. Format Conversion
   ```csharp
   // Converting legacy chunks to modern format
   ILegacyChunk legacyChunk = ...;
   if (legacyChunk.CanConvertToModern())
   {
       IIFFChunk modernChunk = legacyChunk.ConvertToModern();
   }
   ```

2. Validation
   ```csharp
   // Validating chunks against specifications
   var validator = new ChunkValidator(specifications);
   var errors = validator.ValidateChunk(chunk);
   ```

3. Documentation Integration
   ```csharp
   // Parsing chunkvault specifications
   var parser = new MarkdownSpecParser();
   var spec = parser.ParseChunkSpec(markdownContent);
   ```

4. Legacy Support
   ```csharp
   // Loading and processing legacy chunks
   if (file.TryLoadLegacyChunk<T>(data))
   {
       // Process legacy chunk
   }
   ```

5. Detailed File Inspection (NEW)
   ```bash
   # Generate detailed YAML dumps of PM4/ADT files for debugging/analysis
   dotnet WoWToolbox.FileDumper.dll -d <input_dir> -o <output_dir>
   ```

## Integration Points

1. Warcraft.NET Integration
   - Base chunk handling
   - Modern format support
   - Extension points
   - Conversion targets

2. DBCD Integration
   - DBC/DB2 reading
   - Data validation
   - Format reference
   - Version tracking

3. Chunkvault Integration
   - Specification source
   - Validation rules
   - Format documentation
   - Relationship mapping

## Future Expansion

1. Enhanced Validation
   - Custom validation rules
   - Complex relationships
   - Performance optimization
   - Automated testing

2. Format Support
   - Additional legacy versions
   - New format detection
   - Conversion pipelines
   - Format analysis

3. Documentation Tools
   - Spec generation
   - Validation reporting
   - Relationship visualization
   - Version tracking 