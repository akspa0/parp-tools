# MOMT — WMO Root Material Records

## Summary
Material record table for root WMO data.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c11a0` after `MOTX`.
- Count formula: `count = chunkSize >> 6` (64-byte stride).

## Structure
- Fixed-size records, 64 bytes each.

## Confidence
- Presence/order/stride: **High**
- Field semantics: **Medium**
