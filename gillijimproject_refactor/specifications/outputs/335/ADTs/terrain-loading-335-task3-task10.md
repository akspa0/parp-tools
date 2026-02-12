# Wow.exe 3.3.5.12340 Terrain Loading - Tasks 3-10

Scope: MCNK header and subchunks, MCIN offset table, heights, world position, iteration order, normals, vertex grid layout.

## Task 3 - MCNK Header Layout (evidence-based)

### Function addresses
- FUN_007c64b0: MCNK init and field reads
- FUN_007c3a10: MCNK subchunk scan

### Decompiled evidence

MCNK data pointer and flags:

```c
// FUN_007c3a10
iVar5 = *(int *)(param_1 + 0x10c);        // MCNK chunk header pointer
*(int *)(param_1 + 0x110) = iVar5 + 8;    // MCNK data start
```

```c
// FUN_007c64b0
pbVar2 = *(byte **)(param_1 + 0x110);     // MCNK data start
*(undefined4 *)(param_1 + 0xc) = 0;
if ((*pbVar2 & 2) != 0) {
  *(undefined4 *)(param_1 + 0xc) = 0x40;  // MCNK.flags bit 1
}
```

MCNK fields used by name/offset:

```c
// FUN_007c64b0
*(undefined4 *)(param_1 + 0xb0) = *(undefined4 *)(*(int *)(param_1 + 0x110) + 0x34); // areaId
*(undefined4 *)(param_1 + 0x84) = *(undefined4 *)(*(int *)(param_1 + 0x110) + 0x70); // position.z
```

```c
// FUN_007c64b0
FUN_007c6150(iVar1, *(undefined4 *)(param_1 + 0x134),
             *(undefined4 *)(pbVar2 + 0x10),
             *(undefined4 *)(pbVar2 + 0x38));
// pbVar2+0x10 = nDoodadRefs, pbVar2+0x38 = nMapObjRefs
```

### Definitive answer
- MCNK data start is treated as `mcnk_header + 8`.
- Observed field offsets:
  - `0x00`: `flags` (bit 1 tested)
  - `0x10`: `nDoodadRefs`
  - `0x34`: `areaId`
  - `0x38`: `nMapObjRefs`
  - `0x70`: `position.z` (used as base height)
- Other MCNK fields exist but were not referenced in the decompiled paths shown here.

### Confidence
Medium (confirmed by field access; full 128-byte layout not fully referenced in available functions).

---

## Task 4 - MCNK Subchunk Offset Convention

### Function addresses
- FUN_007c3a10: MCNK subchunk scan

### Decompiled evidence

```c
// FUN_007c3a10
*(int *)(param_1 + 0x110) = iVar5 + 8;    // MCNK data start
puVar4 = (uint *)(iVar5 + 0x88);          // start of subchunks (0x80 header + 8)
for (iVar5 = *(int *)(iVar5 + 4) + -0x80; 0 < iVar5; iVar5 = (iVar5 + -8) - *puVar1) {
  uVar2 = *puVar4;                        // FourCC
  puVar3 = puVar4 + 2;                    // data start
  ...
  puVar1 = puVar4 + 1;                    // size
  puVar4 = (uint *)((int)puVar3 + *puVar1); // next chunk
}
```

### Definitive answer
- Subchunks are located by scanning the MCNK payload after the 0x80-byte header.
- The loader does not rely on `ofsHeight` / `ofsNormal` to seek; it walks subchunk headers sequentially.
- When used, the base is the MCNK chunk header pointer; subchunk data is at `chunk_start + 0x88`.

### Confidence
High for behavior (scan-based). Low for offset fields since they are not used in this path.

---

## Task 5 - MCVT Height Interpretation

### Function addresses
- FUN_007c3a10: captures MCVT pointer
- FUN_007c5220: height min/max and base height
- FUN_007c64b0: base height assignment

### Decompiled evidence

MCVT pointer capture:

```c
// FUN_007c3a10
if (uVar2 == 0x4d435654) { // MCVT
  *(uint **)(param_1 + 0x11c) = puVar3;  // MCVT data
}
```

Base height from MCNK header:

```c
// FUN_007c64b0
*(undefined4 *)(param_1 + 0x84) = *(undefined4 *)(*(int *)(param_1 + 0x110) + 0x70);
```

MCVT heights adjusted by base height:

```c
// FUN_007c5220
pfVar6 = (float *)(*(int *)(param_1 + 0x11c) + 8); // MCVT floats
...
*(float *)(param_1 + 0x54) = *(float *)(param_1 + 0x84) + *(float *)(param_1 + 0x54);
*(float *)(param_1 + 0x60) = *(float *)(param_1 + 0x84) + *(float *)(param_1 + 0x60);
```

### Definitive answer
- MCVT heights are treated as **relative deltas** and are offset by `position.z` from the MCNK header.

### Confidence
High.

---

## Task 6 - World Position Computation

### Function addresses
- FUN_007d9a70: tile world origin
- FUN_007d6b30: chunk index assignment
- FUN_007c64b0: chunk world origin

### Decompiled evidence

Tile origin (tile indices to world space):

```c
// FUN_007d9a70
*(int *)(iVar4 + 0x48) = param_1;          // tile x
*(int *)(iVar4 + 0x4c) = param_2;          // tile y
fVar1 = (float)(param_2 << 4) * _DAT_00a3e554;
...
fVar2 = (float)(param_1 << 4) * _DAT_00a3fdb0 + _DAT_009e2acc;
fVar1 = -fVar1 + _DAT_009e2acc;
*(float *)(iVar4 + 0x3c) = fVar1;          // tile world Y
*(float *)(iVar4 + 0x40) = fVar2;          // tile world X
```

Chunk indices (local to tile) and global chunk indices:

```c
// FUN_007d6b30
*(int *)(iVar4 + 0x24) = param_2;          // chunk x (0..15)
*(int *)(iVar4 + 0x28) = param_3;          // chunk y (0..15)
*(int *)(iVar4 + 0x34) = *(int *)(param_1 + 0x50) + param_2; // global chunk x
*(int *)(iVar4 + 0x38) = *(int *)(param_1 + 0x54) + param_3; // global chunk y
```

Chunk world origin:

```c
// FUN_007c64b0
fVar3 = (float)*(int *)(param_1 + 0x34) * _DAT_00a3e554; // chunk size
*(float *)(param_1 + 0x7c) = -(_DAT_00a3e554 * (float)*(int *)(param_1 + 0x38)) + _DAT_009e2acc;
*(float *)(param_1 + 0x80) = -fVar3 + _DAT_009e2acc;
```

### Definitive answer
- Tile world origin uses `MapOrigin = _DAT_009e2acc` and tile indices scaled by `(_DAT_00a3e554 * 16)`.
- Chunk world origin uses `MapOrigin` and **global chunk indices** scaled by `_DAT_00a3e554` (chunk size).
- Base height for the chunk is `MCNK.position.z` (offset 0x70) and is added to MCVT heights.

### Constants found
- `_DAT_009e2acc` (MapOrigin, 17066.666)
- `_DAT_00a3e554` (chunk size, 33.333...)

### Confidence
Medium (vertex local-to-world mapping not fully observed in this snippet).

---

## Task 7 - MCNK Chunk Iteration Order

### Function addresses
- FUN_007d6bf0: chunk load loop
- FUN_007d6b30: per-chunk loader

### Decompiled evidence

```c
// FUN_007d6bf0
uVar5 = uVar4 & 0xf;   // y
uVar1 = uVar3 & 0xf;   // x
...
FUN_007d6b30(uVar1,uVar5);
```

```c
// FUN_007d6b30
iVar3 = (param_3 * 0x10 + param_2) * 0x10; // (y * 16 + x) * 16 bytes
```

### Definitive answer
- Chunk order is row-major: `index = y * 16 + x` (y outer, x inner).
- `IndexX = x`, `IndexY = y` in the loader.

### Confidence
High.

---

## Task 8 - MCNR Normal Format

### Function addresses
- FUN_007c3a10: subchunk identification

### Decompiled evidence

```c
// FUN_007c3a10
if (uVar2 == 0x4d434e52) {               // MCNR
  *(uint **)(param_1 + 0x124) = puVar3;  // MCNR data
  if (param_2 != 0) {
    puVar4[1] = 0x1c0;                   // size override
  }
}
```

### Definitive answer
- MCNR is captured as a subchunk within MCNK; size is treated as 0x1c0 bytes when forced.
- This matches the expected 145 normals with padding (size 448 bytes).

### Confidence
Medium (byte-to-float conversion not observed here).

---

## Task 9 - Terrain Vertex Grid Layout

### Evidence status
- Direct vertex ordering (9-8-9-8) and per-vertex local coordinate computation were not observed in the functions traced so far.
- MCVT data is read for height min/max in FUN_007c5220, but mesh construction logic was not in the surfaced call chain.

### Confidence
Low.

---

## Task 10 - MCIN Chunk Offset Table

### Function addresses
- FUN_007d6ef0: sets MCIN pointer
- FUN_007d6b30: uses MCIN entries

### Decompiled evidence

MCIN table pointer setup:

```c
// FUN_007d6ef0
*(byte **)(param_1 + 0x88) = pbVar1 + *(int *)(pbVar1 + 4) + 8; // MCIN chunk header
```

MCIN entry use (16-byte entries):

```c
// FUN_007d6b30
iVar3 = (param_3 * 0x10 + param_2) * 0x10; // entry = (y*16 + x) * 16
bVar2 = *(byte *)(*(int *)(param_1 + 0x88) + 8 + iVar3);
puVar1 = (uint *)(*(int *)(param_1 + 0x88) + 8 + iVar3);
*puVar1 = *puVar1 | 1;
...
FUN_007c64b0(*(int *)(*(int *)(param_1 + 0x88) + iVar3) + *(int *)(param_1 + 0x80),
             (bVar2 & 1) == 0);
```

### Definitive answer
- MCIN entries are 16 bytes each, indexed by `y * 16 + x`.
- The first 4 bytes of the entry are treated as the MCNK chunk offset.
- The offset is added to the tile file base (`param_1 + 0x80`), indicating **absolute file offset**.

### Confidence
Medium (entry access uses both header and +8 data start; decompiler shows slight inconsistency but base+offset behavior is clear).
