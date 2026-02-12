# MODR — Group Doodad References

## Summary
Optional doodad reference list for a group.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` when `MOGP.flags & 0x800`.
- Count formula: `count = chunkSize >> 1`.

## Confidence
- Presence gate/stride: **High**

## Runtime behavior (0.7.0.3694)

- In visibility processing, `FUN_00699de0` iterates `MODR` refs (`group+0xD0`, count `group+0x128`).
- Each ref resolves to a doodad instance and links it into runtime draw/update lists.
- Group state bits are updated so doodad linkage work is tracked and not repeated unnecessarily each frame.
