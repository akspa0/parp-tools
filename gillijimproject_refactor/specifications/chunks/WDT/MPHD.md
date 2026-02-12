# MPHD — WDT Header Flags

## Summary
Global WDT header flags and reserved metadata.

## Parent Chunk
Root-level WDT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_006987e0`: MPHD payload read size is `0x20` |

## Structure — Build 0.7.0.3694 (confirmed)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | WDT flags (`???` complete bit map for 0.7) |
| 0x04 | uint32[7] | reserved | Remaining MPHD fields in this build |

Expected payload size: `0x20` (32) bytes.

## Confidence
- Layout size: **High**
- Flag semantics: **Medium**
