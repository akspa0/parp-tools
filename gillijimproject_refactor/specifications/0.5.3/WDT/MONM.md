# MONM — World Model Name Table

## Summary
Root-level name table chunk for WMO references.

## Parent Chunk
Root-level WDT

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Parser assertion string `iffChunk.token=='MONM'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | char[] | stringBlock | Null-terminated WMO path strings |

## Ghidra Notes
- **Function address**: `???`
- **Evidence**: assertion string at `0x0089FCB4`
- **Parser pattern**: root-table stage before placement data (`MODF`)

## Confidence
- **Medium**

## Unknowns
- Any alpha-specific path rewrite logic
