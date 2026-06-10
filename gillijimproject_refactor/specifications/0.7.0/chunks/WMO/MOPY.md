# MOPY — Group Triangle Material/Flags

## Summary
Per-face material/flag records for a group.

## Parent Chunk
`MOGP` (group file).

## Build 0.7.0.3694 evidence
- Required first subchunk in `FUN_006c1a10`.
- Count formula: `count = chunkSize >> 2` (4-byte stride).

## Confidence
- Presence/order/stride: **High**

## Runtime behavior (0.7.0.3694)

- `FUN_006c1a10` stores `MOPY` pointer/count at `group+0xB4` / `group+0x10C`.
- In intersection/query flow (`FUN_006a9b00`), selected polygon index is validated against `group+0x10C` and used to index `group+0xB4` (`polyList + polyIndex*4`).
- This means `MOPY` participates directly in per-polygon material/flag behavior used by gameplay queries, not only rendering.
