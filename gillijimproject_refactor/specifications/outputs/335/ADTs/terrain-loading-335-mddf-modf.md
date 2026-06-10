# Wow.exe 3.3.5.12340 - MDDF/MODF Evidence (Ghidra)

Scope: How MDDF/MODF chunks are located and how name indices map to MMDX/MMID and MWMO/MWID.

## Chunk discovery: MHDR offsets (not linear scan)

### Function addresses
- FUN_007d6ef0: ADT load/parse (uses MHDR offsets)

### Decompiled evidence

```c
// FUN_007d6ef0
*(byte **)(param_1 + 0x68) = pbVar1; // MHDR data base
*(byte **)(param_1 + 0x88) = pbVar1 + *(int *)(pbVar1 + 4) + 8;   // MCIN
*(byte **)(param_1 + 0xa0) = pbVar1 + *(int *)(pbVar1 + 0xc) + 8; // MMDX
*(byte **)(param_1 + 0xa8) = pbVar1 + *(int *)(pbVar1 + 0x10) + 8; // MMID
*(byte **)(param_1 + 0xa4) = pbVar1 + *(int *)(pbVar1 + 0x14) + 8; // MWMO
*(byte **)(param_1 + 0xac) = pbVar1 + *(int *)(pbVar1 + 0x18) + 8; // MWID
*(byte **)(param_1 + 0x90) = pbVar1 + *(int *)(pbVar1 + 0x1c) + 8; // MDDF
*(byte **)(param_1 + 0x94) = pbVar1 + *(int *)(pbVar1 + 0x20) + 8; // MODF
if ((*pbVar1 & 1) != 0) {
  *(byte **)(param_1 + 0xb0) = pbVar1 + *(int *)(pbVar1 + 0x24) + 8; // MFBO
}
if (*(int *)(pbVar1 + 0x28) != 0) {
  FUN_007d4f10(pbVar1 + *(int *)(*(int *)(param_1 + 0x68) + 0x28) + 8); // MH2O
}
if (*(int *)(*(int *)(param_1 + 0x68) + 0x2c) != 0) {
  *(byte **)(param_1 + 0xb4) = pbVar1 + *(int *)(*(int *)(param_1 + 0x68) + 0x2c) + 8; // MTXF
}
```

### Definitive answer
- The client does **not** scan for MDDF/MODF linearly. It uses **MHDR offsets**.
- Offsets are **relative to MHDR data start** (`pbVar1`) and the code adds `+8` to point to chunk data (skipping FourCC+size).

## MDDF name index mapping (M2)

### Function addresses
- FUN_007c6150: spawns M2 placements from MDDF

### Decompiled evidence

```c
// FUN_007c6150
// MDDF entry size = 0x24
iVar7 = FUN_007becd0(
  *(int *)(*(int *)(param_2 + 0xa8) +
    *(int *)(*(int *)(param_2 + 0x90) + iVar4 * 0x24) * 4) +
  *(int *)(param_2 + 0xa0),
  *(int *)(param_2 + 0x90) + iVar4 * 0x24,
  &local_1c);
```

### Definitive answer
- `MDDF.nameId` (first uint32 in the 0x24 entry) is used as an **index into MMID**.
- `MMID[index]` is a **uint32 offset into MMDX**, then `MMDX + offset` is the filename.
- This is **not** a direct index into a packed string list unless MMID happens to be 0..N-1 with contiguous offsets.

## MODF name index mapping (WMO)

### Function addresses
- FUN_007c6150: spawns WMO placements (MODF + MWID/MWMO)

### Decompiled evidence

```c
// FUN_007c6150
piVar10 = (int *)(*(int *)(local_10 + local_8 * 4) * 0x40 + *(int *)(param_2 + 0x94));
...
FUN_007bf460(
  *(int *)(*(int *)(param_2 + 0xac) + *piVar10 * 4) +
  *(int *)(param_2 + 0xa4),
  piVar10,
  &local_1c,
  1);
```

### Definitive answer
- MODF uses the same pattern: `MWID[index]` is a uint32 offset into `MWMO`, and the filename is `MWMO + offset`.

## Practical implications for your code

1. Use MHDR offsets to locate MDDF/MODF (do not rely on linear scans alone).
2. Resolve MDDF names via `MMID -> MMDX`, not direct index into `m2Names` unless you built `m2Names` from MMID order.
3. Resolve MODF names via `MWID -> MWMO`.

### Confidence
High (direct pointer arithmetic and usage in loader).
