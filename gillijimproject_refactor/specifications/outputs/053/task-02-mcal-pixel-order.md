# Task 2: MCAL Alpha Map Pixel Order Analysis

## Overview
Analysis of MCAL 4-bit alpha map reading order in WoW Alpha 0.5.3 (build 3368).

## Key Findings

### Function Analyzed: CMapChunk::UnpackAlphaBits
**Address**: 0x0069a621

### Decompiled Code
```c
void __fastcall CMapChunk::UnpackAlphaBits(ulong *param_1, uchar *param_2)
{
  int iVar1, iVar2, iVar4, iVar6;
  uint uVar3, uVar5;
  int local_8;
  
  // Null pointer checks
  if (param_1 == NULL) {
    _SErrDisplayError_24(0x85100000, sourceFile, 0x638, "pixels", NULL, 1);
  }
  if (param_2 == NULL) {
    _SErrDisplayError_24(0x85100000, sourceFile, 0x639, "alphaPixels", NULL, 1);
  }
  
  iVar1 = 0;
  
  // MIP LEVEL 1 PATH (32x32)
  if (CWorld::alphaMipLevel == 1) {
    iVar4 = 0;
    local_8 = 0x20;  // 32 rows
    do {
      iVar6 = 0x20;  // 32 columns
      do {
        iVar2 = iVar1;
        param_1[iVar4] = (uint)param_2[iVar2] << 0x1c | 0xffffff;
        iVar4 = iVar4 + 1;
        iVar6 = iVar6 + -1;
        iVar1 = iVar2 + 1;
      } while (iVar6 != 0);
      iVar1 = iVar2 + 0x21;  // Skip to next row (+33)
      local_8 = local_8 + -1;
    } while (local_8 != 0);
    return;
  }
  
  // MIP LEVEL 0 PATH (64x64) - 4-bit packed
  uVar5 = 0;
  do {
    if ((uVar5 & 1) == 0) {
      // Even index: low nibble
      uVar3 = (uint)*param_2 << 0x1c;
    }
    else {
      // Odd index: high nibble
      uVar3 = (*param_2 & 0xfffffff0) << 0x18;
      param_2 = param_2 + 1;  // Advance to next byte
    }
    param_1[uVar5] = uVar3 | 0xffffff;
    uVar5 = uVar5 + 1;
  } while (uVar5 < 0x1000);  // 4096 pixels (64x64)
  return;
}
```

### Analysis

#### Output Array
- **param_1**: Output pixel array (ARGB format with alpha in high byte)
- **Size**: 0x1000 = 4096 pixels = 64×64

#### Input Data (4-bit packed)
- **param_2**: Input alpha data (packed 4-bit values)
- **Size**: 2048 bytes (4096 nibbles)

#### Pixel Order

The critical loop structure is:
```c
uVar5 = 0;
do {
  // Process pixel at index uVar5
  // Extract nibble from param_2
  uVar5 = uVar5 + 1;
} while (uVar5 < 0x1000);
```

**This is a SINGLE LINEAR LOOP** from 0 to 4095.

There are **NO nested row/column loops** in the main path. This means:

### ✅ **ANSWER: ROW-MAJOR ORDER**

The alpha map is stored in **linear row-major order**:
```
for (pixel = 0; pixel < 4096; pixel++)
    read_nibble_at(pixel)
```

Which is equivalent to:
```
for (y = 0; y < 64; y++)      // Outer loop: rows
    for (x = 0; x < 64; x++)  // Inner loop: columns
        read_nibble_at(y*64 + x)
```

This is **row-major** because pixels are stored sequentially left-to-right, then top-to-bottom.

### Nibble Packing

The 4-bit values are packed as:
```
Byte 0: [pixel 1 high nibble][pixel 0 low nibble]
Byte 1: [pixel 3 high nibble][pixel 2 low nibble]
...
```

- **Even indices (0, 2, 4...)**: Extract low nibble (bits 0-3), shift left by 28
- **Odd indices (1, 3, 5...)**: Extract high nibble (bits 4-7), shift left by 24, advance byte pointer

### Mip Level 1 (32×32)

For mip level 1, the code uses explicit nested loops:
```c
local_8 = 0x20;  // 32 rows
do {
  iVar6 = 0x20;  // 32 columns
  do {
    // Process pixel
    iVar4 = iVar4 + 1;
    iVar6 = iVar6 + -1;
    iVar1 = iVar2 + 1;
  } while (iVar6 != 0);
  iVar1 = iVar2 + 0x21;  // +33 to skip to next row
  local_8 = local_8 + -1;
} while (local_8 != 0);
```

This **also confirms row-major**: outer loop = rows (0x20), inner loop = columns (0x20).

The `+0x21` (33) stride shows it's reading from a 33-wide source buffer (likely a different mip format).

## Conclusion

**MCAL Alpha Map Pixel Order: ROW-MAJOR**

The 4-bit alpha values are stored in linear row-major order:
- Pixels 0-63: Row 0 (top)
- Pixels 64-127: Row 1
- ...
- Pixels 4032-4095: Row 63 (bottom)

This matches standard texture layouts used in most graphics systems.

## Cross-References

Functions using alpha data:
- `CMapChunk::UnpackAlphaBits` @ 0x0069a621 (alpha unpacking)
- String reference "alphaPixels" @ 0x008a13a8

## Confidence Level

**High** - The decompiled code clearly shows:
- ✅ Single linear loop processing pixels 0-4095 sequentially
- ✅ No column-major indexing (would require `x*64+y` calculation)
- ✅ Row-major confirmed by mip level 1 nested loop structure (rows outer, columns inner)
- ✅ Standard nibble packing (2 pixels per byte)

## Differences from Later WoW Versions

This matches the row-major format used in later WoW versions. No differences detected in pixel ordering for MCAL.
