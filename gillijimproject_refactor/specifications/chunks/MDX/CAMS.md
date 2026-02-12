# CAMS — Camera Definitions

## Summary
Camera records for model cinematic/preview behavior.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_00448b20` token `0x534d4143` (`CAMS`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total CAMS payload bytes |
| 0x04 | uint32 | cameraCount | Number of camera records |
| 0x08 | byte[] | cameras | Variable-size records; each starts with `bytesThisCamera` |

## Confidence
- Framing: **High**
- Full per-field semantics: **Medium**
