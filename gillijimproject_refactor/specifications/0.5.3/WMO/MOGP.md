# MOGP — WMO Group Header

## Summary
Per-group WMO chunk validated in 0.5.3 parser logic.

## Parent Chunk
WMO root/object stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `iffChunk->token=='MOGP'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | groupHeader | Group header exists; detailed field map pending decompile recovery |

## Version Differences
- **0.5.3 → 0.6.0**: unresolved for now.

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A2854`)
- **Parser pattern**: per-chunk token validation in IFF-style reader
- **Key observations**: MOGP is explicitly validated in-line, confirming active WMO group parsing in this build.

## Confidence
- **Medium**

## References
- internal string evidence at `0x008A2854`
