# MOPY — WMO Material/Polygon Flags

## Summary
Per-polygon material/flag chunk confirmed in 0.5.3 parser assertions.

## Parent Chunk
WMO group (`MOGP`) stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `pIffChunk->token=='MOPY'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | polyFlags | Expected per-face material/flag entries; exact bit layout pending |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A2914`)
- **Parser pattern**: in-group IFF chunk validation
- **Key observations**: confirms MOPY support in pre-release client.

## Confidence
- **Medium-Low**

## References
- internal string evidence at `0x008A2914`
