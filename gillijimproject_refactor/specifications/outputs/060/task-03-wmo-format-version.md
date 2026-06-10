# Task 3: WMO Format Version

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Determine WMO version used and check for group file support.

---

## Key Findings

### Format Determination: **WMO v14 (Alpha-Style, Monolithic)**

---

## Evidence

### 1. WMO Root File Loading

**Function**: [`FUN_006b7a50`](0x006b7a50)
**Address**: `0x006b7a50`

This function loads the WMO root file and parses its chunks:

```c
// Version check
if (*piVar1 != 0x4d564552) {  // 'MVER'
    // Error: expected MVER
}
if (piVar1[2] != 0x10) {  // Version must be 0x10 (16 decimal)
    // Error: version == 0x0010
}

// Parse chunks in order:
iVar2 = FUN_006b7940(&local_8, 0x4d4f4844);  // MOHD - Header
iVar2 = FUN_006b7940(&local_8, 0x4d4f5458);  // MOTX - Textures
iVar2 = FUN_006b7940(&local_8, 0x4d4f4d54);  // MOMT - Materials
iVar2 = FUN_006b7940(&local_8, 0x4d4f474e);  // MOGN - Group Names
iVar2 = FUN_006b7940(&local_8, 0x4d4f4749);  // MOGI - Group Info
iVar2 = FUN_006b7940(&local_8, 0x4d4f5056);  // MOVP - Vertices
iVar2 = FUN_006b7940(&local_8, 0x4d4f5054);  // MOPT - Portal Vertices
iVar2 = FUN_006b7940(&local_8, 0x4d4f5052);  // MOPR - Portal References
iVar2 = FUN_006b7940(&local_8, 0x4d4f4c54);  // MOLT - Lights
iVar2 = FUN_006b7940(&local_8, 0x4d4f4453);  // MODS - Doodad Sets
iVar2 = FUN_006b7940(&local_8, 0x4d4f444e);  // MODN - Doodad Names
iVar2 = FUN_006b7940(&local_8, 0x4d4f4444);  // MODD - Doodad Data
iVar2 = FUN_006b7a10(&local_8, 0x4d435650);  // MVFP (optional)
```

### 2. WMO Version Check

**String Reference**: `"*version == 0x0010"` at `0x008afacc`

The version is checked against `0x10` (16 decimal), which corresponds to **WMO v14** format.

### 3. WMO Group Parsing

**Function**: [`FUN_006b8080`](0x006b8080)
**Address**: `0x006b8080`

```c
// Version validation
if (*piVar1 != 0x4d564552) {  // 'MVER'
    // Error
}
if (piVar1[2] != 0x10) {  // Version 0x10
    // Error: *version == 0x0010
}
if (piVar1[3] != 0x4d4f4750) {  // 'MOGP' - Group chunk
    // Error: expected MOGP
}
```

### 4. Chunk Order Analysis

The chunk parsing order in `FUN_006b7a50` reveals the file structure:

| Chunk | FourCC | Purpose |
|-------|--------|---------|
| MVER | 0x4d564552 | Version |
| MOHD | 0x4d4f4844 | Header |
| MOTX | 0x4d4f5458 | Texture filenames |
| MOMT | 0x4d4f4d54 | Materials |
| MOGN | 0x4d4f474e | Group names |
| MOGI | 0x4d4f4749 | Group info |
| MOVP | 0x4d4f5056 | Vertices |
| MOPT | 0x4d4f5054 | Portal vertices |
| MOPR | 0x4d4f5052 | Portal references |
| MOLT | 0x4d4f4c54 | Lights |
| MODS | 0x4d4f4453 | Doodad sets |
| MODN | 0x4d4f444e | Doodad names |
| MODD | 0x4d4f4444 | Doodad data |
| MVFP | 0x4d435650 | Visibility flags (optional) |

### 5. No Group File Loading Detected

The code shows **no evidence** of loading separate group files (`*_000.wmo`, `*_001.wmo`, etc.). All group data appears to be embedded within the single WMO file.

This is consistent with **WMO v14 (monolithic)** format where:
- All groups are stored in a single file
- No separate root/group file split
- MOGP chunks are embedded within the root file

---

## Corresponding 0.5.3 Functions

| 0.6.0 Address | Function Purpose | 0.5.3 Reference |
|---------------|------------------|-----------------|
| 0x006b7a50 | WMO Root Loading | `CMapObj::Load` |
| 0x006b8080 | WMO Group Parsing | `CMapObjGroup::Load` |
| 0x006b7940 | Chunk Finder | `FindChunk` |

---

## Conclusion

**0.6.0 uses WMO v14 (Alpha-style, monolithic format)**:
- Version: 0x10 (16 decimal) = WMO v14
- No separate group files - all data in single WMO
- Chunk structure matches 0.5.3 Alpha format

The WMO v17 split format (separate root + group files) had **NOT** been adopted yet.

---

## Confidence Level: **HIGH**

- Explicit version check against 0x10
- Complete chunk parsing order documented
- No group file path construction found in binary
