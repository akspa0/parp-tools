# VERS — MDX Version

## Summary
Version chunk for MDX model files.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | `MDLX` magic confirmed in `FUN_004220e0`; no explicit `VERS` token query found in traced load chain |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | magic | `MDLX` file signature |
| 0x04 | ... | ... | Subsequent chunks (`MODL`, etc.) |

## Confidence
- Magic presence: **High**
- Standalone `VERS` chunk use in this build: **Low/Unknown**
