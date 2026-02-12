# MCVT — Height Values

## Summary
Stores terrain heights for one MCNK using the classic 9x9 + 8x8 vertex pattern.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Confirmed non-interleaved sequential float reads |
| 0.7.0.3694 | Inferred continuity from 0.6.0 transitional implementation |

## Structure — Build 0.7.0.3694 (inferred, high confidence)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[145] | heights | 81 outer + 64 inner terrain heights |

## Size
- Expected payload size: `145 * 4 = 580` bytes.

## Ghidra Notes
- 0.6.0 processing function: `FUN_006a7d20`.
- Sequential float iteration confirms non-interleaved layout.

## Confidence
- **High**
