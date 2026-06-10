# WDT Chunks — 0.5.3.3368 (Deep-Dive)

This folder tracks deeper 0.5.3 WDT findings from parser assertion strings in the client binary.

## Confirmed Root WDT Tokens
- `MARE` (`0x0089FBE4`)
- `MAOF` (`0x0089FC00`)
- `MVER` (`0x0089FC2C`, `0x0089FC84`)
- `MAIN` (`0x0089FC54`)
- `MPHD` (`0x0089FC6C`)
- `MDNM` (`0x0089FC9C`)
- `MONM` (`0x0089FCB4`)

## Confirmed Embedded ADT-Like Tokens in Same Parse Domain
- `MHDR` (`0x008A2388`)
- `MCIN` (`0x008A236C`)
- `MTEX` (`0x008A2350`)
- `MCNK` (`0x008A126C`)
- `MCLY` (`0x008A1254`)
- `MCRF` (`0x008A12DC`)
- `MDDF` (`0x008A2334`)
- `MODF` (`0x008A2318`)

## Interpretation
- 0.5.3 appears to use a monolithic map parse path where WDT root parsing flows into ADT-like terrain/object chunk handling.
- This aligns with the early-alpha “WDT contains ADT-style content” behavior.

## Key Files
- `MONOLITHIC_LAYOUT.md` contains the stage model and confidence split.
- Per-chunk files document knowns and unresolved fields individually.

## Pending
- Recover parser entrypoint addresses and real decompiled loop bodies.
- Convert `???` fields into concrete offsets/types with decompiler + hex corroboration.
