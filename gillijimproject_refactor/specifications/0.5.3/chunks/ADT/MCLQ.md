# MCLQ — Legacy Terrain Liquid

## Summary
High-priority liquid chunk in alpha-era terrain; not directly observed by assertion strings in this pass.

## Parent Chunk
Likely `MCNK`.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | **Not directly observed** by token assertion string in current scan |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ??? | liquidHeader | Known to vary heavily across early builds |
| 0x?? | ??? | vertex/tile data | Requires dedicated decompile + hex validation |

## Notes
- This remains the top crash-risk terrain chunk and should be first target for field-level recovery.

## Confidence
- Presence in 0.5.3 specifically: **Low-Medium** (lineage-based expectation)
