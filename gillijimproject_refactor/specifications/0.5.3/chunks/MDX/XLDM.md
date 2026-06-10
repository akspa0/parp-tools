# XLDM — MDLX Magic Check (Little-Endian)

## Summary
0.5.3 model loader validates MDLX magic using little-endian `'XLDM'` comparison.

## Parent Chunk
Root-level MDX/MDL file header.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Assertion `*((ULONG *) (fileData)) == 'XLDM'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | magic | MDLX signature read as `'XLDM'` in LE |
| 0x04 | ... | section stream | Section-driven parse follows |

## Confidence
- Magic check: **High**
