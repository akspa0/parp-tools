# MDNM — Doodad Name Table

## Summary
Root-level name table chunk for M2/MDX doodad model references.

## Parent Chunk
Root-level WDT

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Parser assertion string `iffChunk.token=='MDNM'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | char[] | stringBlock | Null-terminated model path strings |

## Ghidra Notes
- **Function address**: `???`
- **Evidence**: assertion string at `0x0089FC9C`
- **Parser pattern**: paired with `MONM`, then referenced by placement chunks (`MDDF`)

## Confidence
- **Medium**

## Unknowns
- Encoding/path normalization behavior in this build
