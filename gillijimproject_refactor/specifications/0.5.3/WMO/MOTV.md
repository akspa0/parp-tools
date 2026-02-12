# MOTV — WMO Texture Coordinates

## Summary
WMO UV chunk referenced directly by 0.5.3 parser checks.

## Parent Chunk
WMO group (`MOGP`) content stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `pIffChunk->token=='MOTV'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | float[2] * N | uv | Standard UV stream expected; dual-MOTV support not yet confirmed for this build |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A28C0`)
- **Parser pattern**: strict token check per chunk
- **Key observations**: confirms explicit MOTV parsing path in alpha-era client.

## Confidence
- **Medium**

## References
- internal string evidence at `0x008A28C0`
