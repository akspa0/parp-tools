# MOGP — WMO Group Header and Container

## Summary
Defines one WMO group and anchors required/optional geometry subchunks.

## Parent Chunk
WMO group parse domain.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk->token=='MOGP'` |

## Notes
- Exact header field map unresolved in this pass.
- Related group token assertions in same cluster: `MOBA`, `MOTV`, `MONR`, `MOVT`, `MOPY`, `MLIQ`, `MOBN`, `MOBR`, `MOLR`, `MODR`, `MOCV`, `MPB*`.

## Confidence
- Presence and central role: **High**
- Header offsets: **Low-Medium**
