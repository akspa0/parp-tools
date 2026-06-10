# Alpha ADT Alpha Map Rules (v18) — Consolidated Guide

This document consolidates the definitive rules we follow for Alpha ADT terrain chunks (MCNK) with a focus on texture layers (MCLY) and alpha maps (MCAL). It captures the on-disk format and the exact implementation choices used in our LK→Alpha converter.

## Scope
- Alpha ADT (aka v18) terrain tiles.
- MCNK sub-structure only: MCVT, MCNR, MCLY, MCRF, MCAL, MCSH, MCSE.
- What the client expects on disk and how we emit it.

## Terminology and Flags
- **MCLY.flags**
  - `0x100` use_alpha_map — layer has an MCAL slice (layers > 0 only).
  - `0x200` alpha_map_compressed — indicates RLE compression (only valid with 8-bit alpha mode).
- **WDT MPHD flags**
  - `0x4` or `0x80` → map alpha bit depth is 8 bits. Without these, the map uses 4-bit alpha.
- **MCNK.flags**
  - `0x8000` → if bit depth is 8, client multiplies alpha by 0.7 where shadows are present.

## MCNK Sub-blocks (Alpha v18)
- **With headers (FourCC + size + data):** MCLY, MCRF.
- **Raw payload only (NO headers):** MCVT, MCNR, MCAL, MCSH, MCSE.
- The MCNK header stores relative offsets/sizes into these payloads:
  - offsHeight (MCVT), offsNormal (MCNR), offsLayer (MCLY header), offsRefs (MCRF header), offsAlpha (MCAL raw), sizeAlpha, offsShadow (MCSH raw), sizeShadow, offsSndEmitters, nSndEmitters.

## MCLY Table (texture layers)
- 16 bytes per entry (little-endian):
  - `[0x00]` uint32 textureId
  - `[0x04]` uint32 flags (bit 8 = 0x100 use_alpha_map, bit 9 = 0x200 alpha_map_compressed)
  - `[0x08]` uint32 offsetInMCAL (start index into MCAL raw payload for this layer)
  - `[0x0C]` uint32 effectId
- **Layer 0 (base):**
  - Must NOT use an alpha map. Clear `0x100` and `0x200`. Set `offsetInMCAL = 0`.
- **Layers > 0:**
  - If emitting 4-bit alpha: set `0x100`, clear `0x200`.
  - `offsetInMCAL` points into this chunk’s MCAL raw payload (NOT a chunk header; there is none) and increments cumulatively per layer by the size of the previous layer’s MCAL slice.

## MCAL (alpha maps)
There are three modes in the client, determined by WDT MPHD bit-depth and MCLY 0x200:

- **Uncompressed 2048 (4-bit)**
  - Used when MPHD does NOT set 0x4 or 0x80.
  - 64×64 alpha map stored in 2048 bytes (2 pixels per byte, LSB-first: low nibble = first pixel, high nibble = second pixel).
  - Client expects the "63×63" layout: last column/row duplicate previous values.
    - Duplicate rule:
      - `alpha[x][63] = alpha[x][62]`
      - `alpha[63][x] = alpha[62][x]`
      - `alpha[63][63] = alpha[62][62]`
  - Read/write order: left-to-right, top-to-bottom.
  - No chunk header for MCAL; it is a raw payload inside MCNK.

- **Uncompressed 4096 (8-bit)**
  - Used when MPHD sets 0x4 or 0x80 (bit depth 8).
  - 64×64 alpha map in 4096 bytes, 1 byte per pixel.
  - If `MCNK.flags` has `0x8000`, the client multiplies alpha by 0.7 where shadows are present.

- **Compressed (RLE) 8-bit**
  - Only valid with 8-bit bit depth. Signal with `MCLY.flags 0x200`.
  - Per-row runs; decompress to exactly 4096 bytes (stop at 4096 if Blizzard data overflows due to historical corruption).
  - Values are stored line by line; runs may not span rows.

## Our Implementation (LK→Alpha)
- We currently emit the 4-bit uncompressed mode (2048 bytes per alpha slice):
  - WDT MPHD depth flags (0x4/0x80): NOT set.
  - `MCLY.flags` for layers > 0: set `0x100`, clear `0x200`.
  - `MCAL` is a raw concatenation of 4bpp slices per layer (>0) with:
    - LSB-first nibble order.
    - 63×63 duplication (last row/column duplicate the previous values).
  - `offsetInMCAL` for each layer points into this concatenation.
- Base layer (0) does not have an alpha map; flags cleared; offset 0.
- `MCLY` and `MCRF` are emitted with headers; `MCAL`, `MCSH`, `MCSE` are raw.

## Edge Cases and Validations
- **Offsets:** Ensure `offsetInMCAL` is within `sizeAlpha` for this MCNK.
- **Monotonicity:** Per-layer `offsetInMCAL` must be monotonic increasing as slices are appended.
- **Sizes:** Each 4-bit layer slice is exactly 2048 bytes (after 63×63 duplication and packing). 8-bit uncompressed is 4096. Compressed varies; decompress must end at 4096.
- **Orientation:** Rows are written top-to-bottom; columns left-to-right. No vertical flip is applied in our 4-bit packer. If a source appears vertically mirrored, verify source orientation before packing.

## Rationale for Choices
- We avoid 8-bit modes (and thus shadow attenuation and RLE) to keep the pipeline deterministic and identical to many Alpha-era assets.
- Clearing `0x200` prevents the client from attempting RLE decompression on our 4-bit data.
- 63×63 duplication prevents cross-tile bleeding and inverted-edge artifacts.

## Pseudocode for 4-bit 63×63 Pack
- For each row `y = 0..63`:
  - `yy = y == 63 ? 62 : y`
  - For each pair `(x0, x1) = (0,1), (2,3), ... (62,63)`:
    - Clamp to `(62,62)` when `x1 == 63`.
    - Convert 8-bit source to 4-bit: `(value + 8) >> 4`.
    - Store as `dstByte = lo | (hi << 4)` (LSB-first nibble).
- Result is 2048 bytes per layer slice.

## File/Code Pointers
- Builder: `WoWRollback/WoWRollback.LkToAlphaModule/Builders/AlphaMcnkBuilder.cs`
  - `Pack8To4_63x63` implements packing.
  - MCLY flags/offset logic ensures 4bpp mode (0x100 set, 0x200 cleared).
- Packer: `WoWRollback/WoWRollback.LkToAlphaModule/Writers/AlphaWdtMonolithicWriter.cs`
  - Emits Alpha WDT/ADT scaffolding and writes MCNK blobs.

## Future Work (Optional)
- Add a guarded 8-bit path:
  - Set WDT MPHD bit depth (0x4/0x80).
  - Emit 4096-byte slices (uncompressed) or enable RLE and set `0x200`.
  - Apply MCNK 0x8000 shadow attenuation.

---
This document aims to be the single source of truth for Alpha ADT alpha maps in this codebase. If implementation diverges, update this file in the same change.
