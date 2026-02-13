# MARE — Alpha-Era Root Chunk (Unresolved)

## Summary
Root-level chunk explicitly validated by the 0.5.3 parser; likely alpha-era-specific metadata.

## Parent Chunk
Root-level WDT

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `iffChunk.token == 'MARE'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | ??? | unresolved; no field-level decode yet |

## Ghidra Notes
- **Function address**: `???`
- **Evidence**: string at `0x0089FBE4`
- **Key observations**: appears in same root-assertion cluster as `MVER/MAIN/MPHD/MDNM/MONM`

## Confidence
- **Low-Medium** (presence confirmed, meaning unresolved)
