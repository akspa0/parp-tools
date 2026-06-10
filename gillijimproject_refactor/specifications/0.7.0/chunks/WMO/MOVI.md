# MOVI — Group Triangle Indices

## Summary
Triangle index array for group geometry.

## Parent Chunk
`MOGP` (group file).

## Build 0.7.0.3694 evidence
- Required second subchunk in `FUN_006c1a10`.
- Count formula: `count = chunkSize >> 1` (16-bit indices).

## Confidence
- Presence/order/stride: **High**

## Runtime behavior (0.7.0.3694)

- `FUN_006c1a10` wires `MOVI` to `group+0xB8` with count at `group+0x110`.
- Triangle index data is consumed by group intersection/render traversal paths (through `FUN_006a2fb0` callers).
- Together with `MOVT`, it defines the final triangle topology used in culling, hit tests, and drawing.
