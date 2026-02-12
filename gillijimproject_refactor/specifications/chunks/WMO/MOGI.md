# MOGI — WMO Group Info Records

## Summary
Per-group metadata table used to instantiate root-side group objects.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c11a0` after `MOGN`.
- Count formula: `count = chunkSize >> 5` (32-byte stride).

## Structure
- Fixed-size records, 32 bytes each.

## Confidence
- Presence/order/stride: **High**
- Per-field names: **Medium**
