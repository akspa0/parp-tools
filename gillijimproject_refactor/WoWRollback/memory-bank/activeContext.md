# Active Context

- **Focus**: Abandon broken roundTrip implementation and refocus on pure Alpha WDT → LK ADT and LK ADT → Alpha ADT conversions using working existing code; fix texture layers (MCLY/MCAL) properly once and for all.
- **Current Status**: ✅ **DECISION MADE** - RoundTripValidator doesn't work due to reader/writer confusion; reverting to proven existing code from src/gillijimproject-csharp which handles conversions correctly.
- **Completed This Session (2025-10-19)**:
  1. ✅ **Identified Root Cause**: New tool introduced errors (e.g., MCLY/MCAL zeroing); original working code functions as intended.
  2. ✅ **Plan Updated**: Reuse working AlphaAdtReader/LkAdtWriter from existing codebase; abandon faulty new implementations.
  3. ✅ **Texture Layer Focus**: Prioritize fixing MCLY/MCAL using proven logic (MCLY with headers, MCAL as raw bytes).
- **What Works Now**:
  - Existing code correctly handles Alpha WDT → LK ADT and vice versa.
  - MCLY/MCAL extraction in original implementations preserves data without zeroing.
  - No need for mixed reader/writer logic.
- **Next Steps**:
  1. **Integrate Working Code**: Copy/adapt proven converters into WoWRollback.LkToAlphaModule.
  2. **Fix Texture Layers**: Ensure AlphaDataExtractor.cs uses correct MCLY ("YLCM" header) and MCAL (raw bytes) from existing patterns.
  3. **Test Conversions**: Run full round-trip with real data; achieve byte-level parity.
  4. **Add Tests**: Create xUnit tests based on working examples.
  5. **Clean Up**: Remove broken new code; update build to eliminate warnings.
- **Implementation Notes**:
  - Reuse src/gillijimproject-csharp AdtAlpha and AdtLk for conversions.
  - MCLY: Always read with chunk header; MCAL: Read as raw bytes.
  - Avoid hybrid paths; stick to pure Alpha/LK pipelines.
- **Known Limitations**:
  - New faulty implementations are abandoned; focus on integration.
  - Round-trip will use existing working code for reliability.
  - Texture layers fixed using proven methods; no new bugs introduced.
