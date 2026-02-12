# ADT Unknown Field Resolution — 0.9.1.3810 (macOS Ghidra)

## Goal
Resolve previously unknown ADT fields by tracing real consumer code paths in the 0.9.1.3810 mac client.

## Executive Summary
Most ADT unknowns in the current 0.9.1 notes are now explainable with direct usage evidence:
- `MCNK` count fields (`0x10`, `0x38`, `0x5C`) are confirmed runtime counters.
- `MCNK+0x34` behaves like `areaId` (high-confidence inference).
- `MCLQ` flow/tile fields now map to concrete query/render logic.
- `MHDR` first dword is present but not consumed in this build path (likely flags/reserved).

---

## A) MHDR fields (root ADT)
From `CMapArea::Create()` (`0x00295c7c`):

- `MHDR + 0x04` (`relative +0x14` in file view due chunk header) -> used as `ofsMCIN`.
- `MHDR + 0x08` -> `ofsMTEX`.
- `MHDR + 0x0C` -> `ofsMMDX`.
- `MHDR + 0x10` -> `ofsMMID`.
- `MHDR + 0x14` -> `ofsMWMO`.
- `MHDR + 0x18` -> `ofsMWID`.
- `MHDR + 0x1C` -> `ofsMDDF`.
- `MHDR + 0x20` -> `ofsMODF`.

### Notable unknown
- `MHDR + 0x00` is byte-swapped but not used by this code path.
- Interpretation: likely flags/reserved field for this era.

---

## B) MCNK header unknowns
From `CMapChunk::CreatePtrs`/`Create`/`CreateSoundEmitters`:

- `MCNK+0x10` = `doodadRefCount` (**confirmed**).
- `MCNK+0x38` = `mapObjRefCount` (**confirmed**).
- `MCNK+0x5C` = `soundEmitterCount` (**confirmed** via `CreateSoundEmitters` loop bound).
- `MCNK+0x34` copied to chunk member used as a persistent chunk attribute.
  - Inference: `areaId` (matches historical MCNK layout and placement).

### Count-field takeaway (re your 0.5.3 note)
The 0.9.1 loader clearly treats several previously opaque fields as counters. This strongly supports your 0.5.3 observation that unknown fields can be count fields in earlier layouts.

---

## C) MCLQ unknowns resolved
From `CMapChunk::CreateLiquids`, `CChunkLiquid::GetAaBox`, `CMap::QueryLiquidStatus`, and `FUN_00023884`:

- `MCLQ block +0x000/+0x004` -> min/max liquid Z bounds (fed to liquid AABB).
- `MCLQ block +0x290` -> pointer to 8×8 per-tile flags.
  - Runtime uses low nibble (`&0xF`) for empty/type checks, low bits (`&0x3`) for mode, high bit (`>>7`) as extra flag.
- `MCLQ block +0x2D0` -> flow mode/count selector (`0`, `1`, `2` observed).
- `MCLQ block +0x2D4` -> `SWFlowv[2]` (flow vectors/params) used when flow mode requires it.

### Still unknown
- `MCLQ block +0x288`, `+0x28C` remain unreferenced in observed runtime paths (likely reserved/padding in this build).

---

## D) Confidence
- **High**: all count-field claims, tile/flow field usage, and MHDR offset mapping.
- **Medium-High**: `MCNK+0x34` as `areaId` (structural fit + usage pattern, no explicit symbol label).
- **Medium**: exact semantics of `MCLQ +0x288/+0x28C`.
