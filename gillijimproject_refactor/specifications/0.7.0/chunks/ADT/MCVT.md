# MCVT — Height Values

## Summary
Stores terrain heights for one MCNK using the classic 9x9 + 8x8 vertex pattern.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed non-interleaved 9x9 + 8x8 decode in `FUN_006b0770` |

## Structure — Build 0.7.0.3694 (confirmed)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[145] | heights | 81 outer + 64 inner terrain heights |

## Size
- Expected payload size: `145 * 4 = 580` bytes.

## Ghidra Notes
- 0.7 function `FUN_006b0770` iterates:
	- outer grid: 9x9
	- inner grid: 8x8
	- source pointer advances linearly over `float` heights.

## Confidence
- **High**
