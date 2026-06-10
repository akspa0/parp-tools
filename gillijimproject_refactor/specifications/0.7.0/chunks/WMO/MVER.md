# MVER — WMO Version

## Summary
Version chunk for WMO file.

## Parent Chunk
Root-level WMO chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Explicit check for version `0x10` (v14 path) |
| 0.7.0.3694 | Confirmed in root parser `FUN_006c11a0` and group parser `FUN_006c17f0` |

## Structure — Build 0.7.0.3694 (confirmed)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MVER` |
| 0x04 | uint32 | size | Standard chunk size field (expected to be 4 in valid files) |
| 0x08 | uint32 | version | Checked as `0x10` |

## Notes
- This build checks `MVER` + version `0x10` in both root and per-group WMO files.
- Version value alone does not indicate monolithic layout in this build; group split is handled separately.

## Confidence
- **High (direct decompile evidence)**
