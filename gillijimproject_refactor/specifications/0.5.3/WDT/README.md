# WDT Chunks — 0.5.3.3368

This folder tracks confirmed WDT parsing evidence from the 0.5.3 client binary.

## Confirmed
- `MVER` assertion path exists (`iffChunk.token == 'MVER'`)
- Build-specific marker `WDT500` exists
- Loader references include `wdtFile` and `%s\\%s.wdt`

## Pending
- Resolve parser function entrypoint addresses
- Recover full chunk dispatch and offsets for `MPHD`, `MAIN`, and related early-WDT structures
