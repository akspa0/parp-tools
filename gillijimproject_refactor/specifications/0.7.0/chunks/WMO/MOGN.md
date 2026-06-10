# MOGN — WMO Group Name Blob

## Summary
Raw group-name string blob used by group metadata.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c11a0` after `MOMT`.
- Stored as pointer + byte-size (`param_1[0x4a]`, `param_1[0x56]`).

## Structure
- Payload treated as opaque bytes in this pass.

## Confidence
- Presence/order: **High**
- Internal offsets/encoding: **Medium**
