# MONR — Group Vertex Normals

## Summary
Normal vectors paired with `MOVT` vertices.

## Parent Chunk
`MOGP` (group file).

## Build 0.7.0.3694 evidence
- Required fourth subchunk in `FUN_006c1a10`.
- Count formula: `count = chunkSize / 12`.

## Structure
- `float[3]` entries.

## Confidence
- Presence/order/stride: **High**

## Runtime behavior (0.7.0.3694)

- `FUN_006c1a10` wires `MONR` to `group+0xC0` with count at `group+0x118`.
- Normals are consumed alongside vertices in lighting and geometric evaluation paths used by renderer/query code.
