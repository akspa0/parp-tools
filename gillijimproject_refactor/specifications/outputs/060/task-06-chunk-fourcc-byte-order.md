# Task 6: Chunk FourCC Byte Order

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Determine if FourCCs are forward or reversed on disk.

---

## Key Findings

### Format Determination: **Forward Byte Order (LK-Style)**

---

## Evidence

### 1. FourCC Comparison Values

From the disassembly of [`FUN_006a6d00`](0x006a6d00):

```asm
006a6d09: CMP dword ptr [EAX], 0x4d434e4b    ; 'MCNK'
006a6d40: CMP dword ptr [EAX + ECX*0x1], 0x4d435654    ; 'MCVT'
006a6d82: CMP dword ptr [ECX + EAX*0x1], 0x4d434e52    ; 'MCNR'
006a6dc4: CMP dword ptr [ECX + EAX*0x1], 0x4d434c59    ; 'MCLY'
006a6e06: CMP dword ptr [ECX + EAX*0x1], 0x4d435246    ; 'MCRF'
006a6e48: CMP dword ptr [ECX + EAX*0x1], 0x4d435348    ; 'MCSH'
006a6e8a: CMP dword ptr [ECX + EAX*0x1], 0x4d43414c    ; 'MCAL'
006a6ecc: CMP dword ptr [ECX + EAX*0x1], 0x4d434c51    ; 'MCLQ'
006a6f0e: CMP dword ptr [ECX + EAX*0x1], 0x4d435345    ; 'MCSE'
```

### 2. FourCC Byte Analysis

| Hex Value | Byte 0 | Byte 1 | Byte 2 | Byte 3 | FourCC |
|-----------|--------|--------|--------|--------|--------|
| 0x4d434e4b | 0x4d | 0x43 | 0x4e | 0x4b | 'MCNK' |
| 0x4d435654 | 0x4d | 0x43 | 0x56 | 0x54 | 'MCVT' |
| 0x4d434e52 | 0x4d | 0x43 | 0x4e | 0x52 | 'MCNR' |
| 0x4d434c59 | 0x4d | 0x43 | 0x4c | 0x59 | 'MCLY' |
| 0x4d435246 | 0x4d | 0x43 | 0x52 | 0x46 | 'MCRF' |
| 0x4d435348 | 0x4d | 0x43 | 0x53 | 0x48 | 'MCSH' |
| 0x4d43414c | 0x4d | 0x43 | 0x41 | 0x4c | 'MCAL' |
| 0x4d434c51 | 0x4d | 0x43 | 0x4c | 0x51 | 'MCLQ' |
| 0x4d435345 | 0x4d | 0x43 | 0x53 | 0x45 | 'MCSE' |

### 3. WDT Chunk FourCCs

From [`FUN_00690530`](0x00690530):

| Hex Value | FourCC | Purpose |
|-----------|--------|---------|
| 0x4d564552 | 'MVER' | Version |
| 0x4d504844 | 'MPHD' | Header |
| 0x4d41494e | 'MAIN' | Tile Index |
| 0x4d574d4f | 'MWMO' | WMO Filenames |
| 0x4d4f4446 | 'MODF' | WMO Placement |

### 4. WMO Chunk FourCCs

From [`FUN_006b7a50`](0x006b7a50):

| Hex Value | FourCC | Purpose |
|-----------|--------|---------|
| 0x4d564552 | 'MVER' | Version |
| 0x4d4f4844 | 'MOHD' | Header |
| 0x4d4f5458 | 'MOTX' | Textures |
| 0x4d4f4d54 | 'MOMT' | Materials |
| 0x4d4f474e | 'MOGN' | Group Names |
| 0x4d4f4749 | 'MOGI' | Group Info |
| 0x4d4f5056 | 'MOVP' | Vertices |
| 0x4d4f5054 | 'MOPT' | Portal Vertices |
| 0x4d4f5052 | 'MOPR' | Portal References |
| 0x4d4f4c54 | 'MOLT' | Lights |
| 0x4d4f4453 | 'MODS' | Doodad Sets |
| 0x4d4f444e | 'MODN' | Doodad Names |
| 0x4d4f4444 | 'MODD' | Doodad Data |
| 0x4d4f4750 | 'MOGP' | Group |
| 0x4d435650 | 'MVFP' | Visibility Flags |

### 5. Model Magic

From [`FUN_00421250`](0x00421250):

| Hex Value | FourCC | Purpose |
|-----------|--------|---------|
| 0x584c444d | 'MDLX' | Model Magic |

### 6. Byte Order Interpretation

On x86 (little-endian), when reading 4 bytes from memory into a 32-bit register:
- Memory: `[0x4d, 0x43, 0x4e, 0x4b]` (M, C, N, K)
- Register: `0x4d434e4b`

This means the bytes are stored in **forward order** on disk:
- First byte = 'M' (0x4d)
- Second byte = 'C' (0x43)
- Third byte = 'N' (0x4e)
- Fourth byte = 'K' (0x4b)

If the bytes were reversed (Alpha-style), we would see:
- Memory: `[0x4b, 0x4e, 0x43, 0x4d]` (K, N, C, M)
- Register: `0x4b4e434d`

---

## Conclusion

**0.6.0 uses forward byte order for FourCCs (LK-style)**:
- FourCCs stored as 'MCNK', 'MCVT', etc. on disk
- NOT reversed ('KNCM', 'TVCM', etc.)
- Same byte order as later versions (1.x+)

This is a **transitional change** from 0.5.x Alpha format (if Alpha used reversed byte order).

---

## Confidence Level: **HIGH**

- Direct comparison values from disassembly
- All FourCCs follow forward byte order pattern
- Consistent across WDT, ADT, WMO, and model formats
