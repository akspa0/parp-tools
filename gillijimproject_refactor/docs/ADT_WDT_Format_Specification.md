# ADT/WDT Format Specification

**Complete terrain and map format documentation across all WoW versions.**

---

## Version History

| Client | Map Format | ADT Format | Notes |
|:---|:---|:---|:---|
| Alpha 0.5.3 | Monolithic WDT | None | All terrain in WDT |
| Alpha 0.5.5 | Monolithic WDT | Prototype (latent) | ADT loader exists but gated |
| Alpha 0.6.0 | Split WDT+ADT | Standard | First production ADT |
| WotLK 3.3.5a | Split WDT+ADT | Standard | Identical to 0.6.0 |

---

# Part 1: WDT (World Data Table)

## Monolithic WDT (0.5.3/0.5.5)

The entire map is contained in a single `.wdt` file with embedded terrain chunks.

### Chunk Sequence
```
MVER → MPHD → MAIN → [terrain data] → MODF (optional)
```

### MAIN Chunk
Contains terrain info for all 64x64 tiles (4096 entries). Each entry indicates if terrain exists at that position.

## Split WDT (0.6.0+)

The WDT references external `.adt` files for terrain.

### Chunk Sequence
```
MVER → MPHD → MAIN → MWMO → MODF
```

### File Naming
- Root: `MapName.wdt`
- Tiles: `MapName_XX_YY.adt` (where XX, YY are 0-63)

### MPHD Chunk (WDT Header)

> [!IMPORTANT]
> Ghidra analysis of WoW 4.0.0.11927 confirmed these flags control ADT parsing behavior.

```c
struct MPHD {
  uint32_t flags;        // 0x00
  uint32_t lgtFileDataId; // 0x04 (Legion+)
  uint32_t occFileDataId; // 0x08 (Legion+)
  // ... more fields in newer versions
};
```

| Flag | Value | Description | Ghidra Reference |
|:---|:---|:---|:---|
| `wdt_uses_global_map_obj` | 0x0001 | WMO only (no terrain) | - |
| `adt_has_mccv` | 0x0002 | ADT has MCCV (vertex colors) | - |
| **`adt_has_big_alpha`** | **0x0004** | **4096-byte alpha maps (8-bit)** | `0x00674640` |
| `adt_has_doodad_ref_sort` | 0x0008 | Sorted doodad refs | - |
| `adt_has_mclv` | 0x0010 | ADT has MCLV | - |
| `adt_has_upside_down_ground` | 0x0020 | Inverted ground | - |
| `unk_0x0040` | 0x0040 | Unknown | - |
| **`adt_has_height_texturing`** | **0x0080** | Combined with 0x0004 for texturing | `0x00674720` |

# Part 2: ADT (Area Data Tile)

## Standard ADT Structure (0.6.0+)

### Top-Level Chunks

| Chunk | Token | Description |
|:---|:---|:---|
| MVER | 0x4D564552 | Version |
| MHDR | 0x4D484452 | Header with offsets |
| MCIN | 0x4D43494E | 16x16 chunk index |
| MTEX | 0x4D544558 | Texture names |
| MMDX | 0x4D4D4458 | M2 model names |
| MMID | 0x4D4D4944 | M2 name offsets |
| MWMO | 0x4D574D4F | WMO names |
| MWID | 0x4D574944 | WMO name offsets |
| MDDF | 0x4D444446 | M2 placements (36 bytes) |
| MODF | 0x4D4F4446 | WMO placements (64 bytes) |
| MCNK | 0x4D434E4B | Map chunks (256 total) |

### MHDR Offsets
```c
struct MHDR {
  uint32_t flags;       // 0x00
  uint32_t mcin_ofs;    // 0x04
  uint32_t mtex_ofs;    // 0x08
  uint32_t mmdx_ofs;    // 0x0C
  uint32_t mmid_ofs;    // 0x10
  uint32_t mwmo_ofs;    // 0x14
  uint32_t mwid_ofs;    // 0x18
  uint32_t mddf_ofs;    // 0x1C
  uint32_t modf_ofs;    // 0x20
  // ...more in later versions
};
```

---

## Prototype ADT (0.5.5 Latent)

> [!WARNING]
> This format exists in 0.5.5 code but is gated by a hardcoded flag (`0x008ab3e4`).

### Key Differences from Standard
1. **No MVER chunk** - File starts directly with MHDR
2. **8-byte padding** before most chunks
3. **MHDR offsets point to padding**, not chunk headers

### File Structure
```
MHDR
[8 bytes padding]
MCIN
[8 bytes padding]
MTEX
[8 bytes padding]
MDDF
[8 bytes padding]
MODF
MCNK...
```

### Modified MHDR
Offsets point to the padding before each chunk, with unused gaps:
```c
struct MHDR_055 {
  uint32_t flags;       // 0x00
  uint32_t mcin_ofs;    // 0x04 → points to padding
  uint32_t mtex_ofs;    // 0x08 → points to padding
  uint32_t unused1;     // 0x0C
  uint32_t mddf_ofs;    // 0x10 → points to padding
  uint32_t unused2;     // 0x14
  uint32_t modf_ofs;    // 0x18 → points to padding
  uint32_t unused3;     // 0x1C
};
```

---

# Part 3: MCNK (Map Chunk)

Each ADT contains 256 MCNK chunks (16x16 grid).

## Sub-Chunks

| Chunk | Token | Description |
|:---|:---|:---|
| MCVT | 0x4D435654 | Height map (145 floats) |
| MCNR | 0x4D434E52 | Normals (435 bytes packed) |
| MCLY | 0x4D434C59 | Texture layers |
| MCRF | 0x4D435246 | Object references |
| MCSH | 0x4D435348 | Shadow map |
| MCAL | 0x4D43414C | Alpha maps |
| MCLQ | 0x4D434C51 | Liquid (old format) |
| MCSE | 0x4D435345 | Sound emitters |

### 0.5.5 Prototype MCNK Layout
Fixed offsets with explicit padding:

| Offset | Size | Content |
|:---|:---|:---|
| +0 | 128 | MCNK Header |
| +128 | 8 | **Padding** |
| +136 | 580 | MCVT (heights) |
| +716 | 435 | MCNR (normals) |
| +1151 | 13 | **Padding** |
| +1164 | Var | MCLY (layers) |

## MCAL Alpha Map Formats

> [!IMPORTANT]
> Decompiled from WoW 4.0.0.11927 `CMapChunk::UnpackAlphaBits()` at `0x00674b70`.

### Format Selection Logic

| MCLY Flag | WDT MPHD Flag | Format | Size |
|:---|:---|:---|:---|
| - | - | Uncompressed 4-bit | 2048 bytes |
| - | 0x4 or 0x80 | Uncompressed 8-bit | 4096 bytes |
| 0x200 | 0x4 or 0x80 | **RLE Compressed** | Variable |

### MCLY Layer Flags (Ghidra: `0x00674640`)

| Flag | Value | Description |
|:---|:---|:---|
| `animation_x` | 0x001 | Texture animation |
| `animation_y` | 0x002 | |
| `animation_45deg` | 0x004 | |
| `animation_90deg` | 0x008 | |
| `animation_speed` | 0x010 | |
| `animation_faster` | 0x020 | |
| `animation_fastest` | 0x040 | |
| `animation_mask` | 0x07F | All animation flags |
| `overbright` | 0x080 | 2x brightness |
| `use_alpha_map` | 0x100 | Layer has alpha |
| **`alpha_map_compressed`** | **0x200** | **RLE compressed alpha** |
| `use_cube_map_reflection` | 0x400 | Cubemap reflection |

### RLE Decompression Algorithm

**Decompiled from `FUN_00673230`:**

```c
// Control byte: bit 7 = mode, bits 0-6 = count
// Mode 1 (0x80+): FILL - repeat next byte 'count' times
// Mode 0 (0x00-0x7F): COPY - copy 'count' bytes directly

int RLE_Decompress(byte* src, byte* dest, int maxSize) {
    int iRead = 0, iWrite = 0;
    
    while (iWrite < maxSize) {
        byte ctrl = src[iRead++];
        
        if (ctrl & 0x80) {  // FILL mode
            byte value = src[iRead++];
            int count = ctrl & 0x7F;
            memset(&dest[iWrite], value, count);
            iWrite += count;
        } else {  // COPY mode
            for (int i = 0; i < ctrl; i++)
                dest[iWrite++] = src[iRead++];
        }
    }
    return iRead;
}
```

### MCNK Flag 0x8000 (Shadow Multiply)

When set, alpha values are multiplied by `178/256 ≈ 0.695` at positions with shadow:
```c
if (hasShadow) alpha = (alpha * 0xB2) >> 8;  // Ghidra: 0x00674720
```

---

# Part 4: Placement Structures

## Coordinate System

> [!IMPORTANT]
> WoW uses **XZY** order where Z is height (up).

- **Position**: X, Z (height), Y
- **Extents**: MinX, MinZ, MinY, MaxX, MaxZ, MaxY
- **Rotation**: Euler angles in degrees

## MDDF (M2 Placement) - 36 bytes

```c
struct SMDoodadDef {
  /*0x00*/ uint32_t nameId;      // MMDX index
  /*0x04*/ uint32_t uniqueId;
  /*0x08*/ float posX;
  /*0x0C*/ float posZ;           // Height
  /*0x10*/ float posY;
  /*0x14*/ float rotX;
  /*0x18*/ float rotY;
  /*0x1C*/ float rotZ;
  /*0x20*/ uint16_t scale;       // 1024 = 1.0
  /*0x22*/ uint16_t flags;
};
```

## MODF (WMO Placement) - 64 bytes

```c
struct SMMapObjDef {
  /*0x00*/ uint32_t nameId;      // MWMO index
  /*0x04*/ uint32_t uniqueId;
  /*0x08*/ float posX;
  /*0x0C*/ float posZ;           // Height
  /*0x10*/ float posY;
  /*0x14*/ float rotX;
  /*0x18*/ float rotY;
  /*0x1C*/ float rotZ;
  /*0x20*/ float extents[6];     // MinX, MinZ, MinY, MaxX, MaxZ, MaxY
  /*0x38*/ uint16_t flags;
  /*0x3A*/ uint16_t doodadSet;
  /*0x3C*/ uint16_t nameSet;
  /*0x3E*/ uint16_t scale;
};
```

---

# Part 5: Converter Requirements

## Generating 0.5.3/0.5.5 Maps
1. Use **Monolithic WDT** format
2. Embed all terrain in WDT
3. No external ADT files

## Generating 0.6.0+ Maps
1. Use **Split WDT+ADT** format
2. MDDF: 36-byte entries
3. MODF: 64-byte entries
4. MCSH (shadows) should be included
5. Coordinates: Z in position 2 (height)

## Generating 0.5.5 Prototype ADT
1. **No MVER** chunk
2. Add 8-byte padding before MCIN, MTEX, MDDF, MODF
3. MHDR offsets point to padding
4. MCNK: 8-byte padding after header, MCLY starts at offset 1164
