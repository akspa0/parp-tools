# Task 2 - SEQS 1400 name corruption

## Concrete parser evidence
- Function: `FUN_007a9c70` (SEQS chunk handler)
- Entry check:
  - Reads `numSeq = FUN_007f40a0()`
  - Requires `param_2 - 4 == numSeq * 0x8c` (compare at `0x007a9cc5`/`0x007a9cc7`)
  - Rejects with `Invalid SEQX section` on mismatch.
- Per-record decode:
  - Raw copy `0x50` bytes from stream at `0x007a9d85` (`FUN_007f3ed0(..., 0x50)`) into record base.
  - Then reads typed fields at fixed offsets:
    - `+0x50`, `+0x54` (int)
    - `+0x58` (float)
    - `+0x5c` (uint)
    - `+0x60` (vec3)
    - `+0x6c` (vec3)
    - `+0x78` (float)
    - `+0x80`, `+0x84` (int)
    - `+0x88` (uint)
  - Record stride increment is fixed: `iVar5 += 0x8c`.

## SEQS field provenance map
| Version | stride | name_offset | interval_offset | flags_offset | bounds_offset | Evidence |
|---|---:|---:|---:|---:|---:|---|
| 1300 | 0x8c | 0x00 (size 0x50) | 0x50/0x54 | 0x5c | 0x60 and 0x6c | `FUN_007a9c70` |
| 1400 | **parser still enforces 0x8c** | **parser still uses 0x00..0x4f** | same | same | same | `CMP param_2-4, num*0x8c` |
| 1500 | **parser still enforces 0x8c** | **parser still uses 0x00..0x4f** | same | same | same | same |

## First divergence hypothesis (minimal, concrete)
- There is no in-handler version split for 1400.
- If v1400 SEQS entry size/name placement changed from the hardcoded `0x8c`/`0x50`, name bytes become misframed and decode as garbage.
- Address-level mismatch causing this: `0x007a9cbc` (`IMUL ECX, ECX, 0x8C`) + `0x007a9d85` (`copy 0x50 name bytes`) under all versions.
