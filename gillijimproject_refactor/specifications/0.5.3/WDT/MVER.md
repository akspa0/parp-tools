# MVER — Version Chunk

## Summary
Root-level version chunk validated by the 0.5.3 client parser before WDT body parsing.

## Parent Chunk
Root-level (WDT file)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | 4 bytes (expected) | Assertion string references `iffChunk.token == 'MVER'` and `iffChunk.token=='MVER'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32 | version | WDT format version value (exact numeric value not yet confirmed from decompiler path) |

## Version Differences
- **0.5.3 → 0.6.0**: pending deep-diff pass.

## Ghidra Notes
- **Function address**: `???` (string-backed assertion located at `0x0089FC2C` and `0x0089FC84`)
- **Parser pattern**: `iffChunk.token == 'MVER'` style validation before consuming body
- **Key observations**: this is one of the few chunk checks explicitly emitted as assertion text in 0.5.3.

## Confidence
- **Medium** (chunk identity and validation path confirmed; exact function body still unresolved)

## References
- wowdev.wiki (general MVER semantics)
