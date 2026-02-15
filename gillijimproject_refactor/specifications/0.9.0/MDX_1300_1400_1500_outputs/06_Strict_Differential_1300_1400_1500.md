# Task 6 - Strict differential report

## A) Parser path differences by version
- Observed path for all tested version families in this binary:
  - chunk dispatch: `FUN_00784200`
  - `VERS` -> `FUN_007abf40`
  - `SEQS` -> `FUN_007a9c70`
  - `GEOS` -> `FUN_007a2a00`
- Explicit version compare found: only upper bound in `FUN_007abdd0` (`version > 1500` reject).

## B) SEQS record layout differences
- Parser assumes fixed `SEQS` entry size `0x8c` and fixed name block size `0x50`.
- No version-conditional stride/name offset branch exists in `FUN_007a9c70`.
- Differential implication:
  - 1300 works because layout matches hardcoded expectations.
  - 1400 breaks names if entry framing changed but parser still forces `0x8c/0x50`.

## C) GEOS/PVTX interpretation differences
- Proven core logic uses element-count semantics (not byte-count) for VRTX/NRMS/PVTX.
- No version split observed in these handlers.
- Differential implication:
  - 1500 invisibility can result from field provenance drift under unchanged decode assumptions.

## D) First render gate hit differences
- First suppress gate in scene path: `FUN_004349b0` checks `*(param_4 + 3 + geosetId*0x14) != 0`.
- If false, geoset is skipped from draw submission.

## E) Minimal code change candidates (ranked)
1. **High confidence (1400 names):** make SEQS stride/name framing version-aware in `FUN_007a9c70` equivalent parser code.
2. **Medium confidence (1500 invisible):** validate/fix provenance of per-geoset visibility byte used by `FUN_004349b0` gate.
3. **Medium confidence:** audit GEOS subchunk length interpretation for unproven tags (GNDX/MTGC/MATS/UV*) under 1500.
