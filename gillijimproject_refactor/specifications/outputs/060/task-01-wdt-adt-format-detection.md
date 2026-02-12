# Task 1: WDT/ADT Format Detection

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Determine if 0.6.0 uses monolithic WDT or separate ADT files.

---

## Key Findings

### Format Determination: **LK-Style (Separate ADT Files)**

---

## Evidence

### 1. ADT File Path Format String

**Address**: `0x008af7f4`
**String**: `"%s\%s_%d_%d.adt"`

This format string is used to construct ADT file paths in the pattern:
```
Path\MapName_XX_YY.adt
```

This is the LK-style naming convention for separate ADT files per tile.

### 2. WDT Loading Function

**Function**: [`FUN_00690530`](0x00690530)
**Address**: `0x00690530`

The WDT loading function reads:
1. MVER chunk (version)
2. MPHD chunk (header flags)  
3. MAIN chunk (64x64 tile index = 0x8000 bytes)
4. Optionally MWMO and MODF chunks (if WMO world)

**Key Code Pattern**:
```c
// Read MAIN chunk - 0x8000 bytes (64x64 * 16 bytes per entry)
FUN_00646d70(local_8, &DAT_00e6a8f0, 0x8000, 0, 0, 0);
```

The MAIN chunk is read as 0x8000 (32768) bytes, which corresponds to 64x64 entries of 8 bytes each (Alpha-style MAIN entry size).

### 3. ADT Loading Function

**Function**: [`FUN_006b5010`](0x006b5010)
**Address**: `0x006b5010`

This function:
1. Constructs the ADT filename using the format string
2. Checks if the tile exists via `areaInfo[index].flags & FLAG_EXISTS`
3. Opens and loads the separate ADT file

**Key Code Pattern**:
```c
FUN_006428d0(local_110, 0x100, "%s\%s_%d_%d.adt", &DAT_00e728f8, &DAT_00e76b10, param_1, param_2);
```

### 4. MAIN Entry Structure

The MAIN chunk entries appear to be 8 bytes (Alpha-style) rather than the later 16-byte entries:
- Alpha: 8 bytes per entry (flags + offset/size info)
- LK+: 16 bytes per entry (flags + unused padding)

---

## Corresponding 0.5.3 Functions

| 0.6.0 Address | Function Purpose | 0.5.3 Reference |
|---------------|------------------|-----------------|
| 0x00690530 | WDT Loading | `WorldMap::LoadWDT` |
| 0x006b5010 | ADT Tile Loading | `WorldMap::LoadADT` |
| 0x006b3f70 | ADT File Reading | `CMap::LoadADTFile` |

---

## Conclusion

**0.6.0 uses separate ADT files (LK-style)**, not monolithic WDT files. The WDT contains only the MAIN index that references which tiles exist, and each tile is stored in a separate `MapName_XX_YY.adt` file.

This confirms that the WDT/ADT split had already occurred by build 0.6.0.

---

## Confidence Level: **HIGH**

- Direct string evidence for ADT file naming
- Clear function structure for separate file loading
- MAIN chunk size consistent with Alpha-style entries
