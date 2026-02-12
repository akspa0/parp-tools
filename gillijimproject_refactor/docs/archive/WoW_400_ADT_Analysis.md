# WoW 4.0.0.11927 ADT Format Analysis

**Analysis Date**: Jan 18 2026  
**Binary**: `wow.exe` (4.0.0.11927 Cataclysm Beta)  
**Tool**: Ghidra

## 1. Key Functions Found

### 1.1 Alpha Map Unpacking
| Function | Address | Description |
|----------|---------|-------------|
| `CMapChunk::UnpackAlphaBits` | `0x00674b70` | Main alpha unpacking dispatcher |
| `CMapChunk::UnpackAlphaShadowBits` | `0x00674560` | Shadow+alpha combined unpacker |
| `RLE_Decompress` | `0x00673230` | RLE decompression routine |

### 1.2 String References
- `"CMapChunk::UnpackAlphaBits(): Bad genformat."` at `0x00a2402c`
- `"CMapChunk::UnpackAlphaShadowBits(): Bad genformat."` at `0x00a23ff8`
- `"terrainAlphaBitDepth"` at `0x00a24bfc` (CVar)
- `"Alpha map bit depth set to %dbit on restart."` at `0x00a24430`

## 2. Dispatch Logic

The `UnpackAlphaBits` function dispatches based on two flags:

### 2.1 MCNK Flag Check
```c
if ((*(byte *)(this + 10) & 8) == 0)  // MCNK.flags.do_not_fix_alpha_map
```
- **Flag 0x8 clear**: Use "fixed" 63×63 alpha (expand to 64×64)
- **Flag 0x8 set**: Use direct 64×64 alpha

### 2.2 GenFormat Parameter
- **genformat = 2**: 8-bit alpha (uncompressed or compressed)
- **genformat = 3**: 4-bit alpha (2048 bytes)

### 2.3 MCLY Compression Flag
```c
if ((*param_3 & 0x200) == 0)  // MCLY.flags.alpha_map_compressed
```
- **Flag 0x200 clear**: Read raw bytes directly
- **Flag 0x200 set**: Call RLE decompression

## 3. RLE Decompression Algorithm

Decompiled from `FUN_00673230`:

```c
int RLE_Decompress(byte* src, byte* dest, int maxSize) {
    int iRead = 0;
    int iWrite = 0;
    
    while (iWrite < maxSize) {
        byte ctrl = src[iRead++];
        
        if (ctrl & 0x80) {  // High bit set = FILL mode
            byte fillValue = src[iRead++];
            int count = ctrl & 0x7F;
            memset(&dest[iWrite], fillValue, count);
            iWrite += count;
        } else {  // COPY mode
            int count = ctrl;
            for (int i = 0; i < count; i++) {
                dest[iWrite++] = src[iRead++];
            }
        }
    }
    return iRead;
}
```

### Key Points:
- Control byte: `bit 7 = mode`, `bits 0-6 = count`
- **FILL (0x80+)**: Next byte repeated `count` times
- **COPY (0x00-0x7F)**: Copy `count` bytes directly
- Decompresses to exactly 4096 bytes (64×64)

## 4. Shadow Multiplier

When shadow exists at a position:
```c
bVar4 = (byte)((uint)bVar4 * 0xb2 >> 8);  // ≈ 0.695 multiplier
```
Alpha is multiplied by `178/256 ≈ 0.695` (close to the documented 0.7).

## 5. Implementation Recommendations

### For LKMapService (C#):
1. Check `WDT.MPHD` flags 0x4/0x80 for bit depth
2. Check `MCLY.flags` bit 0x200 for compression
3. Implement RLE decompression matching above algorithm
4. Handle 63×63 → 64×64 expansion based on MCNK flag 0x8

### Helper Functions by Format:
| Address | Description |
|---------|-------------|
| `0x006734f0` | 4-bit unfixed |
| `0x006735c0` | 4-bit fixed |
| `0x006748d0` | 8-bit uncompressed unfixed |
| `0x006749c0` | 8-bit uncompressed fixed |
| `0x00674640` | 8-bit compressed unfixed |
| `0x00674720` | 8-bit compressed fixed |

## 6. Version Notes

4.0.0.11927 uses the **pre-split ADT format** (same as 3.3.5). Split ADTs (`_tex0.adt`, `_obj0.adt`) were introduced in 4.0.1+.
