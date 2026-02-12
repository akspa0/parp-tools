# MOBN — Group BSP Nodes

## Summary
Optional BSP node records for group spatial partitioning.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` when `MOGP.flags & 0x1`.
- Must be immediately followed by `MOBR`.
- Node count passed as `chunkSize >> 4` to `FUN_006c0050`.

## Confidence
- Presence/order/stride: **High**
