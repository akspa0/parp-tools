# Active Context

- **Focus**: Round-trip testing infrastructure for Alpha→LK→Alpha validation
- **Current Status**: ⚠️ **PARTIAL** - Standalone ADT round-trip complete, monolithic WDT detection working but conversion not yet implemented
- **Completed This Session**:
  1. ✅ `LkAdtSource` and `LkMcnkSource` models with proper property types
  2. ✅ `LkAdtBuilder` and `LkMcnkBuilder` fully implemented with MCAL encoding
  3. ✅ `McalAlphaEncoder` integrated for texture layer encoding
  4. ✅ `UseManagedBuilders` option added to `LkToAlphaOptions`
  5. ✅ Comprehensive test suite: 22 automated tests, all passing
  6. ✅ `TestDataFactory` for generating synthetic test data
  7. ✅ Builder bugs fixed (MVER FourCC, MCLY/MCAL headers)
  8. ✅ Build succeeds, tests validate correct output structure
  9. ✅ API Discovery complete - documented Alpha class APIs
  10. ✅ `AlphaDataExtractor` - extracts Alpha ADT data to `LkMcnkSource`
  11. ✅ `RoundTripValidator` - **COMPLETE orchestration framework**
  12. ✅ CLI command `roundtrip-test` added
  13. ✅ `ROUNDTRIP_TESTING.md` documentation created
  14. ✅ **LK→Alpha conversion step implemented** (for standalone ADTs)
  15. ✅ **Byte-by-byte comparison with detailed reporting**
  16. ✅ **Chunk-level difference analysis**
  17. ✅ **Monolithic WDT detection** - detects .wdt files vs .adt files
  18. ✅ **AlphaWdtReader integration** - parses WDT structure, reads tile count
  19. ⚠️ **Monolithic WDT round-trip** - detection works, conversion TODO

- **What Works Now**:
  - **Standalone ADT files** (Maps.335 format):
    - Alpha ADT → LkMcnkSource extraction ✅
    - LkMcnkSource → LK ADT building ✅
    - LK ADT → Alpha ADT conversion ✅
    - Saves intermediate LK ADT file ✅
    - Saves final round-trip Alpha ADT ✅
    - Detailed comparison with original ✅
    - Chunk-level diff reporting ✅
  - **Monolithic WDT files** (0.5.3 format):
    - Detects WDT vs ADT ✅
    - Parses WDT structure ✅
    - Reports tile count and offsets ✅
    - ❌ Full round-trip conversion NOT YET IMPLEMENTED
  - CLI command runs without errors ✅
  - Build succeeds (18 warnings) ✅

- **Implementation Details**:
  - `RoundTripValidator.ValidateRoundTrip()` now performs complete Alpha→LK→Alpha pipeline
  - `ConvertLkToAlpha()` reads MCIN from LK ADT and converts each MCNK using `AlphaMcnkBuilder.BuildFromLk()`
  - `FindMcnkOffsets()` parses MHDR/MCIN to locate all MCNK chunks in LK ADT
  - `CompareAlphaAdtsDetailed()` provides chunk-level analysis with IndexX/IndexY tracking
  - Progress logging at each step with byte counts and chunk counts
  - Detailed difference reporting (up to 10 chunks shown, with totals)

- **What's Missing** (Critical for monolithic WDT support):
  1. ❌ **Extract individual tile ADT data from monolithic WDT**
     - Need to read each tile's MHDR + MCNKs from WDT
     - Parse MCIN to find all 256 MCNK offsets per tile
     - Extract raw ADT bytes for each tile
  2. ❌ **Convert extracted tiles to LK ADT format**
     - Use existing `AlphaDataExtractor` + `LkAdtBuilder` per tile
     - Save intermediate LK ADTs (optional, for debugging)
  3. ❌ **Pack LK ADTs back into monolithic Alpha WDT**
     - Use existing `LkToAlphaOrchestrator.PackMonolithicAlphaWdt()`
     - Or implement direct packing logic
  4. ❌ **Compare packed WDT with original**
     - Byte-by-byte comparison
     - Per-tile difference analysis
     - Report which tiles differ and by how much

- **Known Issues**:
  - `AlphaDataExtractor` MCAL parsing is simplified (assumes 4096-byte layers)
  - Doesn't handle compressed alpha maps properly yet
  - Doesn't parse MCLY flags to determine alpha format
  - May fail on complex texturing scenarios
  - **These issues will be revealed by round-trip testing**

- **Next Steps** (Priority Order):
  1. ⚠️ **Implement monolithic WDT round-trip** (current blocker)
     - Extract tile ADT data from WDT
     - Convert each tile Alpha→LK→Alpha
     - Pack back into monolithic WDT
     - Compare with original
  2. Test with real Alpha 0.5.3 monolithic WDTs (Kalidar.wdt ready)
  3. Analyze comparison results and identify systematic differences
  4. Fix MCAL parsing issues if texture layer differences found
  5. Iterate until byte-for-byte parity achieved
  6. Validate converted files load in Alpha 0.5.3 client

- **Key Files**:
  - `Validators/RoundTripValidator.cs` - ⚠️ Partial (standalone ADT complete, WDT TODO)
  - `Readers/AlphaWdtReader.cs` - ✅ Parses monolithic WDT structure
  - `Services/AlphaDataExtractor.cs` - ✅ Works for standalone ADTs, needs WDT tile extraction
  - `Builders/AlphaMcnkBuilder.cs` - ✅ LK→Alpha conversion (working)
  - `Builders/LkMcnkBuilder.cs` - ✅ Alpha→LK conversion (working)
  - `LkToAlphaOrchestrator.cs` - ✅ Has `PackMonolithicAlphaWdt()` for packing
  - `ROUNDTRIP_TESTING.md` - Usage guide

- **Test Results**:
  - **Kalidar.wdt** (32.8 MB, 56 tiles):
    - ✅ Detection works
    - ✅ Structure parsed correctly
    - ✅ First tile at offset 67280 identified
    - ❌ Round-trip conversion not yet implemented
