# MONOLITHIC_LAYOUT — 0.5.3 WDT↔ADT Parse Domain

## Summary
0.5.3 map loading appears monolithic: WDT root parsing transitions into ADT-like terrain/object chunk parsing in the same operational domain.

## Confirmed token checks

### Root WDT stage
- `MARE` (`0x0089FBE4`)
- `MAOF` (`0x0089FC00`)
- `MVER` (`0x0089FC2C`, `0x0089FC84`)
- `MAIN` (`0x0089FC54`)
- `MPHD` (`0x0089FC6C`)
- `MDNM` (`0x0089FC9C`)
- `MONM` (`0x0089FCB4`)

### Embedded terrain/object stage
- `MHDR` (`0x008A2388`)
- `MCIN` (`0x008A236C`)
- `MTEX` (`0x008A2350`)
- `MCNK` (`0x008A126C`)
- `MCLY` (`0x008A1254`)
- `MCRF` (`0x008A12DC`)
- `MDDF` (`0x008A2334`)
- `MODF` (`0x008A2318`)

## Inferred parse flow
1. Build WDT path (`%s\\%s.wdt`, `wdtFile`) and open file.
2. Validate root WDT chunk cluster (`MVER/MPHD/MAIN/...`).
3. Continue into ADT-like chunk handling under same map-loading path.
4. Parse terrain chunks and placement chunks for runtime structures.

## Why it is high-risk/complex
- Early-alpha custom chunks (`MARE`, `MAOF`) coexist with later-familiar chunk names.
- Some terrain subchunks may be checked via constants rather than assertion strings, making coverage incomplete unless loop bodies are decompiled.
- Layout/field offsets likely drift from later stable docs despite shared FourCC names.

## Confidence
- Monolithic split interpretation: **High**
- Exact parser function boundaries: **Medium**
- Full field maps for alpha-only root chunks: **Low**
