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
- ✅ **MCLY/MCAL Extraction Fixed**: Root cause identified and corrected in `AlphaDataExtractor.cs`
  - MCLY now reads chunk header correctly ("YLCM" + size)
  - MCAL now reads raw bytes without header stripping
  - Build succeeds with fixes applied
- ⏳ **LK→Alpha conversion**: Next priority - implement missing pipeline step in `RoundTripValidator.cs`
- ⏳ **Round-trip testing**: Ready to test with real Alpha ADT once LK→Alpha writer is complete

## Next Steps
1. **Extractor Parity**:
   - Compare logged `MCLY`/`MCAL` buffers with `McnkAlpha` outputs to identify where zeroing occurs.
   - Adjust `AlphaDataExtractor`/`LkMcnkBuilder` to preserve Alpha bytes and correct offsets.
   - Add targeted xUnit tests to guard against regressions.
2. **Writers (deferred)**:
   - Implement `LkAdtWriter` and `AlphaWdtWriter` once extractor parity is restored.
   - Ensure final writers consume pass-through data without reintroducing zeroing issues.
3. **Validation**:
   - Re-run single-tile and full-map RoundTrip after extractor fix; confirm byte-parity in `debug_mcal` dumps.
   - Capture CLI logs in memory bank when parity achieved.
4. **Future Enhancements**:
   - HeightUv/HeightUvDepth liquids, Alpha build detection, and MCSE structural validation remain on the backlog.
