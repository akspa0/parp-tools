# Ghidra MH2O Deep-Dive Guide (Cross-Version)

Goal: replicate the MH2O liquid reverse-engineering process on other WoW builds and detect version deltas.

## 1) Establish the MH2O entry point

1. Find the ADT parser and MHDR offsets.
   - Look for code that resolves `MHDR+0x28` into an MH2O pointer.
   - Confirm it adds `+8` to skip the FourCC+size header.
2. Locate the per-liquid struct entry list and count.
   - In 3.3.5 this is read in `FUN_007d4ab0` via `param_1 + 0x10` and `param_1 + 0x14`.

## 2) Map the per-liquid struct by usage

1. Find functions that read `+0x34..+0x40` for min/max bounds.
2. Find the mask pointer at `+0x54` and how it is indexed.
3. Find the vertex list base at `+0x78` (float3 per vertex).
4. Identify the vtable pointer at `+0x44` and its methods:
   - vtable+4: height sample by index
   - vtable+8: per-vertex extra
   - vtable+0xc: UV data

## 3) Decode the mask correctly

1. Locate the mask decoder (`FUN_007ce180` in 3.3.5).
2. Confirm the byte layout:
   - low nibble (bits 0..3): 4-bit cell value, `0xF` means no-liquid
   - bit 6 and bit 7: extra flags (often used in queries)
3. Confirm addressing: `index = x + y * 8` for 8x8 local cells.

## 4) Verify height solver and collision

1. Find the height solver (`FUN_007ce0b0` in 3.3.5).
   - Ensure bilinear interpolation is used.
2. Find the triangle test (`FUN_009836b0` in 3.3.5).
   - Confirm ray origin+direction layout and outputs (distance, barycentric).
3. Cross-check the player query functions (`FUN_007a0820`, `FUN_007a3570`).

## 5) Rendering pipeline checks

1. Find buffer creation (`CChunkBuf_Vertex`, `CChunkBuf_Index`).
2. Find chunk pool name (`WCHUNKLIQUID`).
3. Check shader strings for liquid variants:
   - `vsLiquidWater`, `psLiquidWater`
   - `vsLiquidProcWater%s`, `psLiquidProcWater%s`

## 6) DBC linkage checks

1. LiquidType.dbc and LiquidMaterial.dbc used for material/settings.
2. Identify the type id source (often at struct offset `+0x144`).
3. Check for fallback logs: "Liquid type [%d] not found, defaulting to water!"

## 7) Diffing between versions

1. Compare MH2O entry and offset usage.
   - If MH2O is missing from MHDR, confirm alternate lookup or scan.
2. Compare mask decode and bounds.
   - Changes in mask layout usually show up in the mask decoder.
3. Compare vtable methods.
   - If signatures or call patterns change, the data layout likely changed too.
4. Track constant changes:
   - Grid scale constants (cell size, chunk size) and interpolation constants.
5. Track shader name changes or additional liquid shader variants.

## 8) Suggested Ghidra workflow

1. Start with string anchors:
   - `WCHUNKLIQUID`
   - `MapChunkLiquid.cpp`
   - `Liquid type [%d] not found`
2. Use xrefs from these strings to find the core liquid functions.
3. Expand the call tree from the liquid buffer builder and query functions.
4. Build a per-liquid struct map and validate it by reading live fields.

## 9) Validation checklist

- MH2O located via MHDR offset and data pointer = `offset + 8`.
- Per-liquid bounds correct and used in loops.
- Mask decode produces `0xF` as no-liquid.
- Heights use bilinear interpolation.
- UVs and extra per-vertex data retrieved via vtable.
- Material/settings derived from LiquidType/LiquidMaterial DBCs.

## 10) Version delta notes

Keep a change log with:
- Field offset shifts in the per-liquid struct.
- Mask byte layout changes.
- New vtable methods or removed ones.
- Changes in query logic or collision tolerances.
- Shader variants or feature flags.
