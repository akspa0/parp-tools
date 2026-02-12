# MOCV — Group Vertex Colors

## Summary
Optional per-vertex color data.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` when `MOGP.flags & 0x4`.
- Count formula: `count = chunkSize >> 2`.

## Confidence
- Presence gate/stride: **High**