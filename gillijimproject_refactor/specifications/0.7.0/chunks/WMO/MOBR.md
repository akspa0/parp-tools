# MOBR — Group BSP Face/Reference Indices

## Summary
Optional companion table to `MOBN` for BSP references.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` immediately after `MOBN` when `MOGP.flags & 0x1`.
- Count passed as `chunkSize >> 1` to `FUN_006c0050`.

## Confidence
- Presence/order/stride: **High**
