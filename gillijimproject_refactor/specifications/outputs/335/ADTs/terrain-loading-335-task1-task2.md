# Wow.exe 3.3.5.12340 Terrain Loading - Tasks 1-2

Scope: WDT MAIN indexing and ADT filename construction.

## Task 1 - WDT MAIN Chunk: Tile Enumeration and Indexing

### Function addresses
- FUN_007bf8b0: CMap::LoadWdt (reads WDT MAIN data)
- FUN_007b5950: tile enumeration and load loop
- FUN_007d9a70: tile object init (stores tile coords)

### Decompiled evidence

WDT MAIN read (entry table size 0x8000 bytes = 4096 entries * 8 bytes):

```c
// FUN_007bf8b0
FUN_00422530(local_8,&DAT_00ce88d0,0x8000,0,0,0);
```

Tile enumeration and index math (flat index = y * 64 + x):

```c
// FUN_007b5950
iVar1 = iVar7 * 0x40 + iVar9;
pbVar8 = (byte *)(&DAT_00ce88d0 + iVar1 * 2);
local_18 = &DAT_00ce48d0 + iVar1;
...
if (((*pbVar8 & 1) != 0) && (*local_10 == 0)) {
  iVar7 = FUN_007d9a70(iVar9,iVar7);
  ...
}
...
pbVar8 = pbVar8 + 8;     // next entry
...
pbVar8 = local_1c + 0x200; // next row (64 * 8)
```

Tile coordinate storage in the tile object (x, y):

```c
// FUN_007d9a70
*(int *)(iVar4 + 0x48) = param_1; // x
*(int *)(iVar4 + 0x4c) = param_2; // y
(&DAT_00ce48d0)[param_2 * 0x40 + param_1] = iVar4;
```

### Definitive answer
- Entry size is 8 bytes per tile (0x8000 bytes for 64 * 64 entries).
- The flat index is decomposed as:
  - x = i % 64
  - y = i / 64
- Indexing is row-major by y (outer loop on y, inner loop on x).
- The existence check uses the low byte of the 8-byte entry: `(*entry & 1) != 0`.
- No coordinate swapping is applied between MAIN indexing and tile storage.

### Constants
- 0x8000 bytes total for MAIN table
- 0x40 (64) for grid dimension
- 0x200 stride between rows (64 * 8)

### Confidence
High.

---

## Task 2 - ADT Filename Construction

### Function addresses
- FUN_007d9a20: ADT filename formatting
- FUN_007d9a70: tile coordinate storage (source for %d args)

### Decompiled evidence

```c
// FUN_007d9a20
FUN_0076f070(local_104,0x100,
  "%s\\%s_%d_%d.adt",
  &DAT_00ce07d0,&DAT_00ce06d0,
  *(undefined4 *)(param_1 + 0x48),
  *(undefined4 *)(param_1 + 0x4c));
```

```c
// FUN_007d9a70
*(int *)(iVar4 + 0x48) = param_1; // x
*(int *)(iVar4 + 0x4c) = param_2; // y
```

### Definitive answer
- Format string: `%s\%s_%d_%d.adt`.
- The first `%d` is the tile x (column) coordinate.
- The second `%d` is the tile y (row) coordinate.
- The coordinates are passed through directly; no swapping or transformation between MAIN index and filename.

### Confidence
High.
