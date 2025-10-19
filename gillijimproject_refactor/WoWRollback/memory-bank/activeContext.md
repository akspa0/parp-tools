# Active Context

- **Focus**: Fix RoundTrip by eliminating intermediates and implementing direct-write pipelines for full files (Alpha WDT/ADT and LK ADT). Ensure strict pass-through for Alpha `MCLY/MCAL` and correct LK re-pack when required. Target reliable Alpha↔LK round-trips with complete data (terrain, textures, placements, liquids, sounds).
- **Current Status**: MCNK subchunk assembly exists (`AlphaMcnkBuilder`), liquids and sounds logic implemented; however, the pipeline synthesizes intermediate partial ADTs and does not emit complete files. Alpha texture extraction (`MCLY/MCAL`) must be strict pass-through per `McnkAlpha` parity; LK write needs proper chunk order and offsets.
- **Completed This Session**:
  1. ✅ Diagnosed RoundTrip failure mode: intermediates only (terrain-focused), no end-to-end writers producing complete ADTs/WDTs.
  2. ✅ Adopted policy: direct-write final targets only; remove/suppress intermediate artifact emission.
  3. ✅ Defined texture policy: Alpha-side `MCLY/MCAL` pass-through; LK-side `MCAL` re-pack only when necessary and fix `MCLY` offsets accordingly.
  4. ✅ Updated memory bank to track the new plan and constraints.
- **What Works Now**:
  - Alpha MCNK subchunk builder exists and can assemble `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCSE`, `MCLQ` raw blocks in Alpha order.
  - MH2O→MCLQ conversion via `LiquidsConverter` with precedence.
  - Sound emitters extraction present.
- **Next Steps (implementation)**:
  1. Implement `LkAdtWriter` that writes complete LK terrain ADTs with `MHDR/MCIN/MCNK[256]`, `MMDX/MMID`, `MWMO/MWID`, `MDDF/MODF`, `MH2O`, and optional `MFBO/MTXF`. Add `AdtLk.ValidateIntegrity()`.
  2. Implement `AlphaWdtWriter` (and `AlphaWdtMonolithicWriter` if needed) to write `MVER/MPHD/MAIN/MDNM/MONM` (+ `MODF` when present) and embedded terrain when required (monolithic). Apply MONM trailing empty string rule.
  3. Replace any intermediate emissions with in-memory assembly feeding writers; orchestrate a single RoundTrip command.
  4. Normalize Alpha texture handling to strict pass-through on read; re-pack only on LK write and update `MCLY` offsets.
  5. Run a one-tile RoundTrip smoke test and record results in `progress.md`.
- **Implementation Notes**:
  - Alpha `MCNK` sub-blocks have no per-subchunk headers; offsets/sizes live in the 128-byte header. Preserve bytes verbatim for `MCLY/MCAL`.
  - LK `MCIN` must be 4096 bytes; `MCNK` GivenSize = 0x80 + Σ(8 + subchunk data+pad); FourCC reversal centralized in writer.
  - MONM stability: `MPHD.nMapObjNames = wmoNames.Count + 1` when any WMO names exist and append a trailing empty string.
- **Known Limitations**:
  - HeightUv/HeightUvDepth liquid formats still deferred.
  - MCSE Alpha vs LK structural differences unverified; keep pass-through with caution.
  - Alpha build detection (0.5.3 vs 0.5.5) TBD for writer nuances.
