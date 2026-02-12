# ADT MCLQ Analysis — WoW 0.8.0.3734

## Summary
In build `0.8.0.3734`, `MCLQ` is still present in `MCNK` and is parsed via fixed-offset references from the MCNK header structure. The parser supports up to 4 liquid entries per chunk controlled by MCNK flags.

## Build
- **Build**: `0.8.0.3734`
- **Source confidence**: High for offsets/control flow, Medium for exact field semantics

---

## MCNK Relationship to MCLQ

### MCNK parser
- **Function**: `FUN_006b8f90`
- Validates MCNK subchunks by offsets in the MCNK header struct.
- Uses MCNK header field at offset `+0x60` as `MCLQ` relative offset.
- Assertion: token at target must be `MCLQ` (`0x4d434c51`).
- Stores resolved pointer in object field `param + 0xf20`.

### Related offsets in same parser
- `MCVT`: header offset field `+0x1c`
- `MCNR`: `+0x18`
- `MCLY`: `+0x1c` (from base pointer path shown in decompile)
- `MCAL`: `+0x24`
- `MCLQ`: `+0x60`
- `MCSE`: `+0x58`

(Exact header struct member naming pending full MCNK struct reconstruction.)

---

## MCLQ Parse Usage

### Liquid object fill routine
- **Function**: `FUN_006b8be0`
- Input source: pointer from `param + 0xf20` (resolved MCLQ payload).
- Checks MCNK flags and conditionally creates up to **4** liquid records.

### Flag gating
- Loop uses masks: `0x04`, `0x08`, `0x10`, `0x20`.
- If flag absent: corresponding liquid slot is freed/zeroed.
- If present: slot is allocated/updated from current MCLQ record.

### Per-record stride and inferred layout
From pointer math in `FUN_006b8be0`:
- Start of record: `puVar4`
- Assigned scalar header values:
  - `record->field0 = puVar4[0]`
  - `record->field1 = puVar4[1]`
- Pointers assigned:
  - `record->pHeights = puVar4 + 2`
  - `record->pTileFlagsOrAux = puVar4 + 0xA4`
  - `record->field2 = *(puVar4 + 0xB4)`
  - `record->pExtra = puVar4 + 0xB5`
- Next record increment:
  - `puVar4 += 0xB5` dwords (`0x2D4` bytes)

So each liquid record is parsed as a fixed-size block of approximately **724 bytes**.

---

## Interpretation vs Later Expansions
- This is **pre-MH2O** style parsing (classic `MCLQ` path is active).
- Parser behavior suggests packed, fixed-stride liquid records rather than later flexible MH2O tile schemas.
- Coexistence with `MCSE` (sound emitter chunk) is explicit in same MCNK parser.

---

## What This Answers
- `MCLQ` exists and is required when MCNK header points to it.
- Up to 4 per-chunk liquid layers are supported in this build.
- Layer activation is controlled by MCNK liquid-related flag bits (`0x04..0x20`).
- Record layout is fixed-stride (`0x2D4` bytes/record) in this implementation.

## Unknowns / Follow-up
- Exact semantic names for each field in the `0x2D4` record (e.g., height grid dimensions, flow, tile mask meaning).
- Whether this exact in-memory interpretation maps 1:1 to on-disk MCLQ without transformation in helper routines.

## Confidence
- **High** for offsets, record stride, and flag-gated 4-slot handling.
- **Medium** for field semantics (needs hex cross-check against sample ADTs).