# MPBV — Group Optional Portal/Batch Vertices Block

## Summary
Optional block in a 4-chunk sequence gated by group flags.

## Parent Chunk
`MOGP` (group file), optional.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c1c60` when `MOGP.flags & 0x400`.
- Required sequence: `MPBV -> MPBP -> MPBI -> MPBG`.

## Confidence
- Presence/order: **High**
- Semantics: **Low-Medium**
