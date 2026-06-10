# MCVP — WMO Optional Convex/Collision Planes

## Summary
Optional root-level table parsed only if present at end of root sequence.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed via optional lookup `FUN_006c1160` in `FUN_006c11a0`.
- Count formula when present: `count = chunkSize >> 4` (16-byte stride).

## Structure
- Fixed-size records, 16 bytes each.

## Confidence
- Presence gate/stride: **High**
