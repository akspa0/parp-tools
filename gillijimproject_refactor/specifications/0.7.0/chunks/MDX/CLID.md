# CLID — Collision Data

## Summary
Collision primitives consumed by model collision subsystem.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_004459c0` token `0x44494c43` (`CLID`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | subchunk | `VRTX` | Collision vertices (`count = header[2]`) |
| ... | subchunk | `TRI ` | Triangle indices (`0x20495254`) |
| ... | subchunk | `NRMS` | Collision normals |

## Confidence
- Subchunk framing/order: **High**
- Full semantic mapping: **Medium**
