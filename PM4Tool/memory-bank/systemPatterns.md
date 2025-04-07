# System Patterns

## Implemented Patterns

1. Legacy Support Pattern
   ```csharp
   public interface ILegacyChunk : IIFFChunk
   {
       int Version { get; }
       bool CanConvertToModern();
       IIFFChunk ConvertToModern();
   }
   ```

2. Version Conversion Pattern
   ```csharp
   public interface IVersionConverter<T> where T : IIFFChunk
   {
       bool CanConvert(int fromVersion, int toVersion);
       T Convert(ILegacyChunk source);
   }
   ```

3. Validation Pattern
   ```csharp
   public class ChunkValidator
   {
       public IEnumerable<ValidationError> ValidateChunk(IIFFChunk chunk);
   }
   ```

## Architecture

1. Core Framework
   - Legacy chunk interface system
   - Version conversion infrastructure
   - Extension method support
   - Validation framework

2. Documentation System
   - Markdown parsing
   - Specification modeling
   - Validation rules
   - Relationship tracking

3. Validation System
   - Size validation
   - Version validation
   - Field validation framework
   - Relationship validation framework

## Design Patterns

1. Interface Segregation
   - ILegacyChunk for legacy support
   - IVersionConverter for conversions
   - IIFFChunk base compatibility

2. Template Method
   - ChunkConverterBase for conversion logic
   - Abstract conversion implementation
   - Version-specific handling

3. Strategy Pattern
   - Validation rule application
   - Format detection
   - Version conversion

4. Factory Pattern
   - Chunk creation
   - Converter instantiation
   - Validator creation

## Code Organization

1. Core Library (WoWToolbox.Core)
   ```
   /Legacy
     /Interfaces
       - ILegacyChunk.cs
       - IVersionConverter.cs
     /Converters
       - ChunkConverterBase.cs
   /Extensions
     - ChunkedFileExtensions.cs
   ```

2. Validation Library (WoWToolbox.Validation)
   ```
   /Chunkvault
     /Models
       - ChunkSpecification.cs
     /Parsers
       - MarkdownSpecParser.cs
     /Validators
       - ChunkValidator.cs
   ```

## Implementation Guidelines

1. Legacy Support
   - Extend ILegacyChunk for each format
   - Implement version-specific converters
   - Use ChunkConverterBase template

2. Validation Rules
   - Define in chunkvault markdown
   - Parse using MarkdownSpecParser
   - Apply using ChunkValidator

3. Extension Methods
   - Chunk loading helpers
   - Conversion utilities
   - Validation helpers

## Testing Strategy

1. Unit Tests
   - Interface implementations
   - Converter logic
   - Validation rules

2. Integration Tests
   - Format conversion
   - Documentation parsing
   - Validation pipeline

3. Documentation Tests
   - Specification compliance
   - Markdown parsing
   - Relationship validation 