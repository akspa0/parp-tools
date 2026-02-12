# MTLS — Material Definitions

## Summary
Material records controlling render blend/state and texture bindings.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_0044d100` token `0x534c544d` (`MTLS`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total MTLS payload bytes |
| 0x04 | uint32 | materialCount | Number of material records |
| 0x08 | byte[] | records | Variable-size records; each starts with `bytesThisMaterial` |

## Confidence
- Presence and framing: **High**
- Per-record internals: **Medium**
