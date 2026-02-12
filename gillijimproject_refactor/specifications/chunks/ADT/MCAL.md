# MCAL — Alpha Maps

## Summary
Contains per-layer blend masks for terrain texturing.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | 4-bit packed alpha decoding documented |
| 0.6.0.3592 | MCAL presence and offset table confirmed |
| 0.7.0.3694 | Inferred from transitional continuity |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | byte[] | alphaPayload | Layer alpha bit/byte packed data; decoding depends on layer flags |

## Notes
- 0.5.3 analysis confirms row-major alpha interpretation for packed 4-bit paths.
- Exact compression mode matrix for 0.7 requires direct decompile/asset checks.

## Confidence
- Presence and role: **High**
- Exact mode map in 0.7: **Medium/Unknown**
