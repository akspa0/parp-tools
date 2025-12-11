# GillijimProject C# Port Audit

**Date**: 2025-10-04  
**Purpose**: Document gillijimproject-csharp foundation library before Phase 6 migration  
**Target**: Keep as dependency vs migrate to WoWRollback.Core

---

## Executive Summary

**gillijimproject-csharp** is a **bug-fixed C# reimplementation** of the legacy C++ Alpha WDT/ADT converter. It provides format readers for Alpha (0.5.x) and LichKing (3.x) WoW file formats.

### Current State
- **Location**: `src/gillijimproject-csharp/`
- **Technology**: C# .NET 9.0, library project
- **Purpose**: Convert Alpha WDT/ADT → LichKing format
- **Status**: Working, bug-fixed, production-ready
- **Architecture**: Foundation library (format readers only)

### Migration Decision
**Recommendation**: **Keep as dependency**, don't merge into WoWRollback.

**Rationale**:
- Already stable and bug-fixed
- Clean separation of concerns (format reading)
- Can be updated independently
- Avoid re-breaking what's fixed

---

## Project Structure

### Root Level
```
src/gillijimproject-csharp/
├── gillijimproject-csharp.csproj   # .NET 9.0 library
├── Program.cs (8KB)                # CLI stub for testing
├── Utilities/                      # Helper functions
│   └── Utilities.cs
└── WowFiles/                       # Format readers (31 items)
    ├── Alpha/ (9 files)            # Alpha format readers
    ├── LichKing/ (3 files)         # LichKing format readers
    └── (shared chunk parsers)
```

### File Inventory (31 files, ~80KB)

#### **Alpha Format Readers** (9 files, 36KB)
```
WowFiles/Alpha/
├── WdtAlpha.cs (3.3KB)      # Combined WDT+ADT container
├── AdtAlpha.cs (8KB)        # Alpha tile reader
├── McnkAlpha.cs (12.8KB)    # Alpha chunk reader (largest)
├── McvtAlpha.cs (3.3KB)     # Alpha vertex heights
├── McnrAlpha.cs (3.7KB)     # Alpha normals
├── MainAlpha.cs (2.1KB)     # Alpha tile table
├── MphdAlpha.cs (1.2KB)     # Alpha WDT header
├── Mdnm.cs (817 bytes)      # M2 name table
└── Monm.cs (945 bytes)      # WMO name table
```

#### **LichKing Format Writers** (3 files, 37KB)
```
WowFiles/LichKing/
├── AdtLk.cs (20.8KB)        # LK ADT writer (largest)
├── McnkLk.cs (15.2KB)       # LK chunk writer
└── McnrLk.cs (721 bytes)    # LK normal writer
```

#### **Shared Chunk Parsers** (19 files, 47KB)
```
WowFiles/
├── Chunk.cs (4.4KB)         # Base chunk reader
├── ChunkHeaders.cs (3.4KB)  # FourCC constants
├── WowChunkedFormat.cs (1.3KB) # Base class
├── Wdt.cs (2.7KB)           # LK WDT
├── Main.cs (495 bytes)      # MAIN chunk
├── Mcin.cs (1KB)            # MCIN chunk index
├── Mcnk.cs (847 bytes)      # MCNK base
├── Mhdr.cs (2.7KB)          # MHDR header
├── Mh2o.cs (2.4KB)          # MH2O liquids
├── Mcal.cs (849 bytes)      # MCAL alpha maps
├── Mcrf.cs (4KB)            # MCRF references
├── Mddf.cs (4.4KB)          # MDDF doodad defs
├── Modf.cs (3KB)            # MODF object defs
├── Mmdx.cs (2.1KB)          # MMDX M2 names
├── Mwmo.cs (1.8KB)          # MWMO WMO names
├── Mmid.cs (1KB)            # MMID M2 indices
├── Mwid.cs (1KB)            # MWID WMO indices
├── Mphd.cs (848 bytes)      # MPHD WDT header
└── Wdl.cs (489 bytes)       # WDL low-res
```

---

## Core Functionality

### Primary Use Case: Alpha → LichKing Conversion

**CLI Usage**:
```bash
# Convert Alpha WDT to LK format
gillijimproject-csharp Azeroth.wdt -o output/

# Result:
# - output/Azeroth.wdt (LK format)
# - output/Azeroth_XX_YY.adt (all tiles, LK format)
```

**Programmatic Usage**:
```csharp
// Read Alpha combined WDT
var wdtAlpha = new WdtAlpha(alphaWdtPath);
var existingTiles = wdtAlpha.GetExistingAdtsNumbers();

// Convert WDT to LK
var wdtLk = wdtAlpha.ToWdt();
wdtLk.ToFile(outputDir);

// Convert each tile to LK
foreach (var idx in existingTiles)
{
    var adtAlpha = new AdtAlpha(alphaWdtPath, offset, idx);
    var adtLk = adtAlpha.ToAdtLk(mdnm, monm);
    adtLk.ToFile(outputDir);
}
```

---

## Architecture Patterns

### 1. **Porting Strategy** (from C++)
Every class has `[PORT]` comments documenting the C++ → C# mapping:

```csharp
/// <summary>
/// [PORT] Original C++ used raw pointers here.
/// Adapted to Span<T> for safety and performance.
/// </summary>
public byte[] ReadChunk(int offset) { ... }
```

### 2. **FourCC Handling** (Critical Detail)
On-disk chunk FourCCs are **reversed**:
- Disk: `RDHM` → Memory: `MHDR`
- Base `Chunk.cs` handles reversal automatically
- All derived classes use forward FourCC literals

```csharp
// Chunk.cs reverses on read/write
public Chunk(byte[] data) 
{
    FourCC = ReverseString(ReadFourCC(data));
}
```

### 3. **Alpha Specifics**

**Combined WDT File**:
- Single file contains WDT + all ADTs
- `MAIN` chunk has 4096 entries (64×64 grid)
- Each entry: offset to embedded ADT (0 = no tile)

**Alpha MCNK Differences**:
- Absolute heights (not relative)
- Different vertex/normal ordering
- No MCVT size field (fixed 580 bytes)
- Different hole encoding

### 4. **Memory Management**
- No manual memory management (C# handles it)
- Uses `Span<T>` for performance-sensitive code
- `IDisposable` not needed (no unmanaged resources)

---

## Key Bug Fixes (from C++ Original)

### 1. **Vertex Ordering**
**C++ Bug**: Incorrect vertex index mapping  
**C# Fix**: Correct Alpha → LK vertex remapping in `McvtAlpha.cs`

### 2. **MCLQ Water Handling**
**C++ Bug**: Wrong offset calculation for liquid data  
**C# Fix**: Proper offset tracking in `McnkAlpha.cs`

### 3. **Hole Encoding**
**C++ Bug**: Inverted hole bitmask  
**C# Fix**: Correct bitmask interpretation

### 4. **Normal Packing**
**C++ Bug**: Precision loss in normal compression  
**C# Fix**: Proper int8 → float conversion

---

## Dependencies

### External
- **.NET 9.0 SDK** (target framework)
- **System.IO** (file operations)
- **System.Linq** (LINQ queries)
- **System.Security.Cryptography** (SHA256 for validation)

### Internal
- No dependencies on other projects
- Self-contained format library
- Can be used standalone

---

## Testing & Validation

### Current Testing
- Manual CLI testing with known Alpha WDTs
- Output validated against legacy converter
- SHA256 hash comparison for regression

### Missing Testing
- No unit tests yet
- No automated regression suite
- No fuzzing for malformed inputs

### Validation Strategy
```bash
# Legacy converter output (C++)
legacy-tool Azeroth.wdt -o legacy/
sha256sum legacy/Azeroth_32_32.adt > legacy.hash

# New converter output (C#)
gillijimproject-csharp Azeroth.wdt -o new/
sha256sum new/Azeroth_32_32.adt > new.hash

# Must match byte-for-byte
diff legacy.hash new.hash
```

---

## Integration Points

### How AlphaWdtAnalyzer Uses It

**AlphaWdtAnalyzer.Core** depends on gillijimproject-csharp:

```csharp
// AlphaWdtAnalyzer.Core/AdtScanner.cs
using GillijimProject.WowFiles.Alpha;

public void ScanWdt(string path)
{
    var wdtAlpha = new WdtAlpha(path);
    var tiles = wdtAlpha.GetExistingAdtsNumbers();
    
    foreach (var idx in tiles)
    {
        var adt = new AdtAlpha(path, offset, idx);
        // Extract terrain data
        // Generate CSVs
        // Export to web viewer format
    }
}
```

### How WoWRollback Could Use It

**Option A: Direct Dependency** (Recommended)
```csharp
// WoWRollback.Core/Formats/AlphaWdtAdapter.cs
using GillijimProject.WowFiles.Alpha;

public class AlphaWdtAdapter : IWdtReader
{
    private readonly WdtAlpha _wdt;
    
    public AlphaWdtAdapter(string path)
    {
        _wdt = new WdtAlpha(path);
    }
    
    public IEnumerable<int> GetTiles() => _wdt.GetExistingAdtsNumbers();
}
```

**Option B: Wrap as Plugin**
```csharp
// WoWRollback.Plugins/AlphaFormatPlugin.cs
public class AlphaFormatPlugin : IFormatPlugin
{
    public string FormatId => "alpha-0.5.x";
    
    public IWdtReader CreateWdtReader(string path)
    {
        return new AlphaWdtAdapter(path);
    }
}
```

---

## Migration Strategy: Keep as Library

### Recommendation: **Library-First Approach**

**DON'T**:
- ❌ Merge code into WoWRollback.Core
- ❌ Rewrite from scratch
- ❌ Copy-paste code

**DO**:
- ✅ Keep gillijimproject-csharp as separate library
- ✅ Reference it from WoWRollback.Core
- ✅ Create thin adapter layers
- ✅ Wrap with plugin interfaces

### Benefits

1. **Don't Re-Break Fixed Bugs**
   - C# port already fixed C++ bugs
   - Merging = risk of reintroducing bugs
   - Separate library = proven stable

2. **Independent Updates**
   - Format changes update library only
   - WoWRollback doesn't need rebuild
   - Versioning independence

3. **Cleaner Architecture**
   - Separation of concerns
   - Format reading ≠ analysis logic
   - Easier to test

4. **Reusability**
   - Other tools can use library
   - AlphaWdtAnalyzer already uses it
   - Don't duplicate code

---

## Phase 6 Integration Plan

### Week 9: Audit Complete ✓
- This document
- Decision: Keep as library

### Week 10-11: Adapter Layer
```
WoWRollback.Core/
└── Formats/
    ├── IWdtReader.cs           # Interface
    ├── IAdtReader.cs           # Interface
    └── Alpha/
        ├── AlphaWdtAdapter.cs  # Wraps WdtAlpha
        └── AlphaAdtAdapter.cs  # Wraps AdtAlpha
```

### Week 12-13: Plugin Wrapper
```
WoWRollback.Plugins/
└── AlphaFormatPlugin.cs        # Registers Alpha readers
```

### Week 14-15: Testing
- Adapter unit tests
- Integration tests with AlphaWdtAnalyzer logic
- SHA256 validation (old vs new)

---

## Risks & Mitigations

### Risk 1: Version Skew
**Risk**: Library updates break WoWRollback  
**Mitigation**: Pin to specific NuGet version, test before upgrade

### Risk 2: Performance Overhead
**Risk**: Adapter layer adds overhead  
**Mitigation**: Make adapters thin (zero-cost abstraction)

### Risk 3: Missing Features
**Risk**: Library doesn't expose needed data  
**Mitigation**: Extend library, submit PR, or fork

---

## Success Criteria

### Phase 6 Complete When:
- [ ] WoWRollback.Core references gillijimproject-csharp
- [ ] Adapter interfaces defined
- [ ] Alpha plugin registered
- [ ] Unit tests pass
- [ ] SHA256 validation: old == new outputs
- [ ] AlphaWdtAnalyzer logic migrated to use adapters
- [ ] Zero code duplication

---

## Code Quality Assessment

### Strengths ✅
- Clean, modern C# (net9.0)
- Well-documented with XML comments
- [PORT] notes track C++ → C# changes
- Bug fixes documented
- Consistent naming conventions

### Weaknesses ⚠️
- No unit tests
- No CI/CD pipeline
- No NuGet package published
- CLI stub minimal (testing only)

### Technical Debt
- None significant
- Library is focused and stable
- Ready for production use

---

## Recommendations

### Short Term (Phase 6)
1. **Keep as library** - Don't merge
2. **Create adapter layer** - Wrap with interfaces
3. **Add unit tests** - Test adapters
4. **Reference from WoWRollback** - NuGet or project reference

### Long Term (Post-Phase 6)
1. **Publish NuGet package** - gillijimproject.wowfiles
2. **Add CI/CD** - Automated testing
3. **Expand test coverage** - Unit + integration tests
4. **Documentation** - API docs, examples

---

## Next Steps

1. **User Approval**: Review and approve this audit
2. **Phase 6 Week 10**: Create adapter interfaces
3. **Phase 6 Week 11**: Implement Alpha adapters
4. **Phase 6 Week 12**: Test and validate

---

**Status**: ✅ Complete, awaiting approval  
**Confidence**: High  
**Decision**: Keep as library (don't merge)  
**Estimated Integration**: 2 weeks (Phase 6, Weeks 10-11)
