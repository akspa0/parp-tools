# ATCH — Attachment Records

## Summary
Attachment points and associated metadata.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_0044e7e0` queries token `0x48435441` (`ATCH`).

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total ATCH payload bytes |
| 0x04 | uint32 | attachmentCount | Number of attachment records |
| 0x08 | byte[] | attachments | Variable-size records; each starts with `bytesThisAttachment` |

## Confidence
- Framing and count: **High**
- Per-record semantics: **Medium**
