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

## Runtime behavior (0.7.0.3694)

- When present, `FUN_006c1c60` stores vertex-color stream pointer/count at `group+0xDC` / `group+0x12C`.
- This optional stream augments group render data and enables color-influenced shading paths where the pipeline consumes per-vertex colors.
- Absence of `MOCV` keeps group rendering on non-vertex-color path.