# Progress

## Completed Milestones

### MCNK Parity Implementation (Oct 2025)
- ✅ **Alpha MCNK audit**: Documented all missing Alpha subchunks (`MCRF`, `MCLQ`, `MCSE`) and header fields using `Alpha.md` / `ADT_v18.md` as references.
- ✅ **Header parity**: Implemented full LK header metadata mirroring (`flags`, `holes`, predictor textures, no-effect doodads).
- ✅ **MCRF implementation**: Built Alpha-native `MCRF` tables with correct doodad/WMO counts and proper index ordering.
- ✅ **Liquids implementation**: 
  - Implemented MH2O extraction from LK MCNK chunks
  - Created MH2O parser for per-chunk liquid data
  - Integrated `LiquidsConverter` for MH2O→MCLQ conversion
  - Implemented MCLQ serializer (469-byte Alpha format)
  - Wired `offsLiquid` header field (0x64)
- ✅ **Sound emitters implementation**:
  - Implemented MCSE extraction from LK chunks
  - Wired `offsSndEmitters` and `nSndEmitters` header fields (0x5C/0x60)
  - MCSE data passes through unchanged (format compatibility assumed)
- ✅ **Build validation**: Zero compilation errors, ready for runtime testing.

### Infrastructure Ready
- `AlphaDataExtractor`, `LiquidsConverter`, `AlphaMclqExtractor` all available and integrated
- `LkToAlphaOptions.SkipLiquids` defaults to `false` (liquids enabled)
- Error handling with graceful fallback (logs warnings, continues conversion)

## Current Status (2025-10-19)
- ✅ **DECISION MADE**: RoundTripValidator abandoned due to reader/writer confusion; reverting to working existing code from src/gillijimproject-csharp.
- ✅ **MCLY/MCAL Extraction Fixed**: Root cause identified - new tool introduced errors; original code handles correctly.
- ⏳ **Integration Priority**: Copy/adapt proven AlphaAdtReader/LkAdtWriter into WoWRollback.LkToAlphaModule.
- ⏳ **Texture Layer Fixes**: Ensure AlphaDataExtractor.cs uses existing MCLY/MCAL logic (headers for MCLY, raw for MCAL).

## Next Steps
1. **Integrate Working Code**:
   - Copy proven converters from src/gillijimproject-csharp into WoWRollback.
   - Ensure pure Alpha WDT → LK ADT and LK ADT → Alpha ADT pipelines.
2. **Fix Texture Layers**:
   - Update AlphaDataExtractor.cs to use correct MCLY ("YLCM" header) and MCAL (raw bytes) from existing patterns.
   - Test with real Alpha ADT to verify non-zero data preservation.
3. **Test Conversions**:
   - Run full round-trip with integrated code; achieve byte-level parity.
   - Add xUnit tests based on working examples.
4. **Clean Up**:
   - Remove or comment out broken new implementations.
   - Eliminate build warnings; ensure zero errors.
5. **Future Enhancements** (Deferred):
   - HeightUv/HeightUvDepth liquids, Alpha build detection, and MCSE structural validation remain on the backlog.

## Summary
Abandoned faulty roundTrip; refocused on integrating working existing code for reliable conversions and proper texture layer handling. No more circular development.
