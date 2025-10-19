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

## Current Status
- ❗ RoundTrip not reliable: pipeline produces intermediate/partial ADTs (terrain-focused) instead of complete files. Alpha texture data (`MCLY/MCAL`) not strictly passed through in Alpha extraction.
- **LK→Alpha conversion**: MCNK subchunks assembly exists but needs integration into a complete Alpha writer (WDT/ADT as applicable).
- **Round-trip testing**: Blocked until full-file writers exist and Alpha texture pass-through is enforced.

## Next Steps
1. **Direct-write Writers**:
   - Implement `LkAdtWriter` for complete LK ADT (MHDR/MCIN/MCNK[256], MMDX/MMID, MWMO/MWID, MDDF/MODF, MH2O, optional MFBO/MTXF) and add `AdtLk.ValidateIntegrity()`.
   - Implement `AlphaWdtWriter` (and `AlphaWdtMonolithicWriter` if needed) to write `MVER/MPHD/MAIN/MDNM/MONM` (+ `MODF`), applying MONM trailing empty string rule.
2. **Texture Policy Enforcement**:
   - Alpha extraction: strict pass-through of `MCLY` table and `MCAL` blob using header offsets/size.
   - LK write: re-pack MCAL only when required; update `MCLY` offsets after packing.
3. **Orchestration**:
   - Replace intermediate emissions with in-memory assembly feeding final writers.
   - Add a single RoundTrip command that writes only final targets.
4. **Validation**:
   - One-tile smoke test: verify MHDR/MCIN/MCNK offsets and presence of all required chunks.
   - Dump `debug_mcal/YY_XX/mcly_raw.bin` and pre/post `mcal.bin` when `--verbose`.
5. **Future Enhancements**:
   - Implement HeightUv/HeightUvDepth liquid formats.
   - Alpha build detection (0.5.3 vs 0.5.5) for writer nuances.
   - Validate MCSE Alpha vs LK structural differences with real data.
