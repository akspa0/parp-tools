# WMO Bounds & Deltas — 0.9.0 (vs 0.8.0)

## Why this doc
- 0.9.0 WMOs intermittently fail parsing when run through a 0.8.0 (v17) profile. The root cause is ordering/version deltas and added chunks that the 0.8.0 loader does not expect.
- This is a mitigation guide to update the loader and avoid bounds/assertion failures.

## Root-level changes (vs 0.8.0)
- MVER version: **0x11 (17)** in 0.9.x, not 0x10.
- Required root chunk order in 0.9.x (matches 0.9.1 evidence, observed in 0.9.0):
  1) MVER(0x11)
  2) MOHD
  3) MOTX
  4) MOMT
  5) MOGN
  6) MOGI
  7) **MOSB** (new vs 0.8.0)
  8) MOPV
  9) MOPT
  10) MOPR
  11) **MOVV** (new vs 0.8.0)
  12) **MOVB** (new vs 0.8.0)
  13) MOLT
  14) MODS
  15) MODN
  16) MODD
  17) MFOG
  18) optional MCVP (if present)
- Divisors confirmed from 0.9.1 (carry to 0.9.0): MOMT=0x40, MOGI=0x20, MOPT=0x14, MOPR=0x08, MOLT=0x30, MODS=0x20, MODD=0x28, MFOG=0x30. Treat MOSB/MOVV/MOVB as required passthrough chunks (no divisor change needed, but do not skip them).

### Action: update root parser
- Require MVER==0x11.
- Insert MOSB, MOVV, MOVB into the strict order above; treat any deviation as fatal.
- Maintain strict size accounting: total chunk walk must not exceed file size; each chunk must fit remaining bytes.

## Group-level changes (vs 0.8.0)
- Required group order (0.9.x): MVER(0x11) → MOGP → MOPY → MOVI → MOVT → MONR → MOTV → MOBA, then optional region.
- Optional flags (0.9.x, from 0.9.1 decompile; adopt in 0.9.0 unless disproved):
  - 0x0001: MOBN + MOBR
  - 0x0004: MOCV
  - 0x0200: MOLR
  - 0x0400: MPBV + MPBP + MPBI + MPBG
  - 0x0800: MODR
  - 0x1000: MLIQ
  - **0x20000: MORI + MORB (new vs 0.8.0)**
- MOTV appears exactly once, stride 0x08 (size >> 3).
- Divisors: MOPY size>>2, MOVI size>>1, MOVT size/0x0C, MONR size/0x0C, MOTV size>>3, MOBA size>>5.

### Action: update group parser
- Require MVER==0x11 before MOGP.
- Enforce the required sequence above.
- Honor new optional flag 0x20000 to accept MORI/MORB; treat unexpected optional tokens as fatal instead of skipping.
- Keep strict per-chunk bounds checks.

## Likely failure signatures you’re seeing
- Loader built for 0.8.0 rejects MOSB/MOVV/MOVB as “unknown” or overruns trying to skip them, breaking subsequent offsets.
- Version check mismatch on MVER 0x11 triggers an early fail or misaligns the cursor if ignored.
- Group files with MORI/MORB set (flags 0x20000) are rejected or misparsed by 0.8.0-based code, leading to bounds assertions in later chunks.

## Minimal patch to tooling
1) Root: accept MVER 0x11; include MOSB, MOVV, MOVB in strict order; keep MCVP optional trailing.
2) Group: accept MVER 0x11; enforce required chain through MOTV and MOBA; process optional flags including 0x20000 (MORI/MORB).
3) Bounds: keep size/divisor checks; fail fast on any unexpected token rather than skipping.
4) Tests: load a failing 0.9.0 WMO—if it previously died post-MOGI, expect success after adding MOSB/MOVV/MOVB. If it died in group optional parse, ensure 0x20000 gate is implemented.

## Notes on liquids (MLIQ)
- 0.9.x keeps MLIQ gated on flag 0x1000; header layout similar to 0.8.0 but includes dual per-vertex layouts; keep the 0x26 + (xVerts*yVerts*8) secondary offset rule from 0.8.0 as a baseline, but be prepared for endian-swapped uint16/uint32 variants per-vertex (see 0.9.1 notes).

## Next steps
- Verify on 0.9.0 samples: confirm MOSB/MOVV/MOVB presence; log any additional unexpected tokens.
- If a WMO still fails after this, dump the offending chunk tag/size at the failure cursor for further deltas.
