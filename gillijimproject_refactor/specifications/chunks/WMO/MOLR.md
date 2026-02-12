# MOLR — Group Light References

## Summary
Optional list of light references for a group.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` when `MOGP.flags & 0x200`.
- Count formula: `count = chunkSize >> 1`.

## Confidence
- Presence gate/stride: **High**

## Runtime behavior (0.7.0.3694)

- In visibility processing, `FUN_00699fa0` iterates `MOLR` refs (`group+0xCC`, count `group+0x124`).
- Missing light runtime objects are instantiated and initialized from map-object light tables.
- Light objects are linked into group runtime lists and group state is marked as light-processed.
