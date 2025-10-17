# System Patterns

## Architecture Overview

### Modular Design
WoWRollback is organized into specialized modules, each with a single responsibility:

```
WoWRollback.Core          → Shared models, services, utilities
WoWRollback.AdtModule     → ADT parsing and writing (Alpha/LK)
WoWRollback.DbcModule     → DBC file parsing (AreaTable, Map)
WoWRollback.AnalysisModule → Spatial clustering, UniqueID analysis
WoWRollback.ViewerModule  → Web viewer data generation
WoWRollback.Viewer        → Static web viewer (HTML/JS/CSS)
WoWRollback.AdtConverter  → Standalone conversion CLI
WoWRollback.LkToAlphaModule → LK → Alpha conversion logic
WoWRollback.Orchestrator  → High-level pipeline orchestration
WoWRollback.Cli           → Unified CLI entry point
WoWRollback.Verifier      → Validation and comparison tools
```

### Key Design Patterns

#### 1. Reader/Writer Pattern
All format handling uses separate reader and writer classes:
```csharp
// Reading
IAdtReader reader = new AlphaAdtReader();
AdtData data = reader.Read(filePath);

// Writing
IAdtWriter writer = new AlphaAdtWriter();
writer.Write(data, outputPath);
```

**Benefits**:
- Clear separation of concerns
- Easy to add new formats
- Testable in isolation
- Swappable implementations

#### 2. Builder Pattern (for LK ADTs)
LK ADT construction uses strongly-typed builders:
```csharp
var lkSource = new LkAdtSource { ... };
var builder = new LkAdtBuilder();
var adtBytes = builder.Build(lkSource);
```

**Benefits**:
- Type-safe construction
- Validation at build time
- Reusable across converters
- Clear API surface

#### 3. Service Layer Pattern
Business logic lives in service classes:
```csharp
// Analysis
UniqueIdRangeCsvWriter.WritePerTileRanges(placements, outputPath);

// Conversion
TerrainConverter.ConvertMcnk(alphaChunk, lkChunk);

// Patching
AdtPatcher.PatchAdt(adtPath, config, outputPath);
```

**Benefits**:
- Testable business logic
- Reusable across CLI/GUI
- Clear dependencies
- Easy to mock

#### 4. Pipeline Pattern
Complex workflows use pipeline orchestration:
```csharp
var pipeline = new ConversionPipeline()
    .AddStep(new ReadAlphaWdt())
    .AddStep(new ConvertToLk())
    .AddStep(new WriteAdtFiles())
    .AddStep(new ValidateOutput());

var result = await pipeline.Execute(context);
```

**Benefits**:
- Composable workflows
- Progress reporting
- Error handling
- Rollback on failure

## Component Relationships

### Core Data Flow
```
Input Files → Readers → Domain Models → Converters → Domain Models → Writers → Output Files
                                ↓
                          Services (Analysis, Validation)
                                ↓
                          Exporters (CSV, JSON, GLB)
```

### Module Dependencies
```
Cli → Orchestrator → {AdtModule, DbcModule, AnalysisModule, ViewerModule}
                                    ↓
                                  Core (shared)

AdtConverter → {AdtModule, LkToAlphaModule} → Core
```

**Principle**: Dependencies flow inward toward Core, never outward

## Key Technical Decisions

### 1. Chunk-Based Parsing
All ADT/WDT parsing uses chunk-based readers:
```csharp
while (reader.Position < length)
{
    var fourCC = reader.ReadFourCC();
    var size = reader.ReadUInt32();
    var data = reader.ReadBytes(size);
    
    switch (fourCC)
    {
        case "MCNK": ParseMcnk(data); break;
        case "MCVT": ParseMcvt(data); break;
        // ...
    }
}
```

**Rationale**: Matches WoW file format structure, easy to extend, robust to unknown chunks

### 2. FourCC Reversal Handling
FourCCs are stored reversed on disk (e.g., "MCNK" → "KNCM"):
```csharp
// Centralized in Chunk.cs
public static string ReverseFourCC(string fourCC)
{
    return new string(fourCC.Reverse().ToArray());
}
```

**Rationale**: Single source of truth, prevents bugs, matches WoW conventions

### 3. Managed vs. Raw Chunks
Two approaches for chunk handling:

**Managed** (preferred for new code):
```csharp
public class LkMcnkSource
{
    public float[] Heights { get; set; }
    public byte[] Normals { get; set; }
    public List<TextureLayer> Layers { get; set; }
}
```

**Raw** (legacy compatibility):
```csharp
public class McnkChunk
{
    public byte[] RawData { get; set; }
    public int Offset { get; set; }
}
```

**Rationale**: Managed for new features, raw for byte-level parity

### 4. Noggit Reference Implementation
Alpha MCAL decoding mirrors Noggit's logic:
```csharp
// Reference: lib/noggit-red/src/noggit/Alphamap.cpp
public class McalAlphaDecoder
{
    public byte[] Decode(byte[] compressed, bool doNotFixAlphaMap)
    {
        // Exact Noggit algorithm
    }
}
```

**Rationale**: Proven correct, community-validated, well-documented

### 5. Coordinate System Handling
Alpha and LK use different coordinate systems:
```csharp
// Alpha: Y-up, right-handed
// LK: Z-up, left-handed (or vice versa - needs verification)

public static Vector3 AlphaToLk(Vector3 alpha)
{
    return new Vector3(alpha.X, alpha.Z, alpha.Y);
}
```

**Rationale**: Explicit transforms prevent subtle bugs

## Data Models

### Domain Models (Core)
```csharp
public class Placement
{
    public string ModelPath { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; }
    public uint UniqueId { get; set; }
    public int TileRow { get; set; }
    public int TileCol { get; set; }
    public string ModelType { get; set; } // "M2" or "WMO"
}

public class TerrainChunk
{
    public float[] Heights { get; set; }  // 145 vertices (9x9 + 8x8)
    public byte[] Normals { get; set; }
    public List<TextureLayer> Layers { get; set; }
    public uint AreaId { get; set; }
    public uint Flags { get; set; }
}
```

### Format-Specific Models
```csharp
// Alpha
public class AlphaAdtData
{
    public byte[] MverChunk { get; set; }
    public byte[] MhdrChunk { get; set; }
    public List<AlphaMcnk> Chunks { get; set; }
    public List<Placement> M2Placements { get; set; }
    public List<Placement> WmoPlacements { get; set; }
}

// LK
public class LkAdtSource
{
    public List<LkMcnkSource> Chunks { get; set; }
    public uint MhdrFlags { get; set; }
}
```

## Error Handling Strategy

### Validation Layers
1. **Input validation**: Check file exists, readable, correct format
2. **Parse validation**: Verify chunk structure, sizes, offsets
3. **Conversion validation**: Check data ranges, required fields
4. **Output validation**: Verify file integrity, client compatibility

### Error Recovery
```csharp
try
{
    var data = reader.Read(filePath);
}
catch (ChunkParseException ex)
{
    logger.Warn($"Skipping invalid chunk: {ex.FourCC}");
    // Continue with remaining chunks
}
catch (FileFormatException ex)
{
    logger.Error($"Invalid file format: {ex.Message}");
    return ConversionResult.Failed(ex);
}
```

**Principle**: Fail gracefully, log everything, continue when possible

## Testing Strategy

### Unit Tests
- Test each converter independently
- Mock file I/O
- Verify chunk structure
- Test edge cases (empty, malformed)

### Integration Tests
- Test full conversion pipelines
- Use real Alpha/LK files
- Verify byte-level output
- Compare with known-good files

### Validation Tests
- Load converted files in game clients
- Visual inspection of terrain/objects
- Automated diff tools
- Round-trip conversion tests

## Performance Considerations

### Memory Management
- Stream large files (don't load entirely into memory)
- Dispose readers/writers properly
- Use `Span<T>` for byte manipulation
- Pool buffers for repeated operations

### Batch Processing
- Process tiles in parallel (when safe)
- Progress reporting every N tiles
- Checkpoint progress for resume
- Fail-fast on critical errors

### Caching
- Cache parsed DBC files (AreaTable, Map)
- Cache texture ID → name mappings
- Cache spatial clusters
- Invalidate on source file change

## Logging & Diagnostics

### Structured Logging
```csharp
logger.Info("Converting ADT", new
{
    MapName = "Azeroth",
    TileX = 32,
    TileY = 48,
    Format = "Alpha"
});
```

### Debug Outputs
- MCAL dumps to `debug_mcal/YY_XX/`
- Chunk structure logs
- Validation reports
- Comparison CSVs

### Progress Reporting
```
[1/64] Converting tile 32_48... ✓
[2/64] Converting tile 32_49... ✓
...
[64/64] Converting tile 39_55... ✓

Summary:
  Tiles converted: 64
  Objects preserved: 26,384
  Warnings: 12
  Errors: 0
```

## Configuration Management

### CLI Options
- Use consistent flag names across commands
- Provide sensible defaults
- Support config files for complex scenarios
- Validate options before execution

### Output Conventions
- Default to `project_output/<map>_<timestamp>/`
- Preserve directory structure
- Generate summary files (JSON, CSV)
- Include metadata (version, timestamp, options)
