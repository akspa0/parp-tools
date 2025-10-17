# Progress

- **MCAL decoder parity**: Shared decoder `McalAlphaDecoder` added (covers compressed, big, and small alpha; includes edge-fix option). Alpha builder hooked up to use it.
- **MCAL encoder added**: `McalAlphaEncoder` emits compressed, big, or packed 4-bit payloads from column-major layers; Alpha options now expose `AssumeAlphaEdgeFixed` and `ForceCompressedAlpha` for round-trip control.
- **Managed LK builders scaffolded**: Introduced strongly typed `LkAdtSource`/`LkMcnkSource` models and `LkMcnkBuilder`/`LkAdtBuilder` placeholders. They still require population logic and LK chunk header formatting before integration.
- **✅ UseManagedBuilders option**: Added flag to `LkToAlphaOptions` to toggle between managed builders (new path) and legacy raw byte copying (old path).
- **✅ LkMcnkSource properties**: Changed raw byte array properties from `init` to `set` to allow population after construction.
- **✅ LkMcnkBuilder Range syntax fix**: Fixed Span slicing to use `.Slice()` instead of range operators with addition.
- **✅ Comprehensive test suite**: Created 22 automated tests covering `LkMcnkBuilder` and `LkAdtBuilder` with synthetic test data.
  - Tests verify MCNK/ADT structure, chunk presence, header preservation, error handling
  - `TestDataFactory` generates valid test data (MCVT, MCNR, MCLY, alpha layers)
  - All 22 tests passing ✅
  - Test project: `WoWRollback.LkToAlphaModule.Tests` added to solution
- **✅ Builder fixes from testing**:
  - Fixed MVER FourCC (REVM not RVEM)
  - Fixed MCLY/MCAL chunk header writing
  - Validated builder produces correct output structure
- **✅ Infrastructure complete and validated**:
  - Managed builders are production-ready
  - All tests pass, build succeeds
  - Ready for integration into main program
- **❌ Not yet integrated into main program**:
  - No CLI command to invoke managed builders
  - `ConvertAlphaToLkManagedBuilders()` is just a TODO comment
  - Legacy `AlphaMcnkBuilder` path still being used
- **✅ API Discovery Complete**:
  - Examined `AdtAlpha` and `McnkAlpha` classes
  - Documented constructor signatures and data access patterns
  - See `memory-bank/API_DISCOVERY.md` for details
- **⚠️ Round-Trip Testing PARTIAL**:
  1. ✅ `AlphaDataExtractor` service extracts raw data from standalone Alpha ADTs
  2. ✅ `RoundTripValidator` orchestrates full Alpha→LK→Alpha pipeline for standalone ADTs
  3. ✅ CLI command `roundtrip-test` added to AdtConverter
  4. ✅ Build succeeds (18 warnings)
  5. ✅ LK→Alpha conversion step implemented using `AlphaMcnkBuilder.BuildFromLk()`
  6. ✅ Byte-by-byte comparison with detailed chunk-level analysis
  7. ✅ Detailed diff reporting with IndexX/IndexY tracking
  8. ✅ Monolithic WDT detection and structure parsing
  9. ❌ Monolithic WDT round-trip conversion NOT YET IMPLEMENTED
- **Files Created/Updated**:
  - `Services/AlphaDataExtractor.cs` - Extracts Alpha MCNK data to LkMcnkSource (may need MCAL improvements based on testing)
  - `Validators/RoundTripValidator.cs` - ✅ **COMPLETE** implementation with full pipeline
  - `ROUNDTRIP_TESTING.md` - Complete usage guide
  - `memory-bank/API_DISCOVERY.md` - Alpha class API documentation
- **Implementation Highlights**:
  - `ConvertLkToAlpha()` - Parses MHDR/MCIN to find all MCNK offsets, converts each using `AlphaMcnkBuilder`
  - `FindMcnkOffsets()` - Reads MCIN table from LK ADT to locate 256 MCNK chunks
  - `CompareAlphaAdtsDetailed()` - Chunk-level comparison with difference tracking
  - Progress logging throughout pipeline (extraction, building, conversion, comparison)
  - Saves intermediate files: `*_lk.adt` (LK format) and `*_roundtrip.adt` (converted back to Alpha)
- **Current Capabilities**:
  - ✅ **Standalone ADT round-trip**: Fully working for individual .adt files (Maps.335 format)
  - ✅ **Monolithic WDT detection**: Correctly identifies and parses .wdt structure
  - ❌ **Monolithic WDT round-trip**: Detection works, but conversion not implemented
  
- **Blocking Issue**:
  - **Monolithic WDT round-trip** is the primary use case (Alpha 0.5.3 format)
  - User's test file (Kalidar.wdt, 56 tiles, 32.8 MB) ready but can't be processed yet
  - Need to implement: tile extraction → LK conversion → WDT packing → comparison
  
- **Next Implementation Tasks**:
  1. ❌ Extract individual tile ADT data from monolithic WDT
  2. ❌ Convert each extracted tile through Alpha→LK→Alpha pipeline
  3. ❌ Pack converted tiles back into monolithic WDT format
  4. ❌ Compare packed WDT with original (byte-by-byte + per-tile analysis)
  5. ⏳ Test with Kalidar.wdt and other Alpha 0.5.3 WDTs
  6. ⏳ Fix MCAL parsing if texture layer differences found
  7. ⏳ Iterate until byte-for-byte parity achieved
  8. ⏳ Validate converted files load in Alpha 0.5.3 client
