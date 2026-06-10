# MCAL — Alpha Maps

## Summary
Contains per-layer blend masks for terrain texturing.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | MCAL token validation confirmed in `FUN_006af6f0` |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | byte[] | alphaPayload | Layer alpha bit/byte packed data; decoding depends on layer flags |

## Notes
- MCAL base pointer is established from MCNK header offset `+0x24` in `FUN_006af6f0`.
- Exact decode/compression branches are outside this function and still require dedicated tracing.

## Confidence
- Presence and role: **High**
- Exact mode map in 0.7: **Medium/Unknown**
