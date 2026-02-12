# MOTX — WMO Root Texture Name Blob

## Summary
Raw texture-name string blob used by root material records.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c11a0` immediately after `MOHD`.
- Stored as pointer + byte-size (`param_1[0x49]`, `param_1[0x55]`).

## Structure
- Payload is treated as opaque bytes in this pass.

## Confidence
- Presence/order: **High**
- Internal string table semantics: **Medium**
