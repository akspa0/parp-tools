# Wow.exe 3.3.5.12340 - MCNR Decoding and Vertex Layout

Scope: MCNR normal decoding and the 9-8-9-8 vertex layout used when building terrain geometry.

## MCNR decoding

### Function addresses
- FUN_007c4f10
- FUN_007c3f30
- FUN_007c4620
- FUN_007c4960

### Decompiled evidence

```c
// FUN_007c4f10
pcVar9 = *(char **)(param_1 + 0x124); // MCNR data pointer
...
pfVar8[3] = (float)(int)*pcVar9 * fVar5;
pfVar8[4] = (float)(int)pcVar9[1] * fVar5;
pfVar8[5] = (float)(int)pcVar9[2] * fVar5;
...
pcVar9 = pcVar9 + 3;
```

```c
// FUN_007c3f30
pcVar4 = *(char **)(param_1 + 0x124); // MCNR data pointer
...
param_2[3] = (float)(int)*pcVar4 * fVar2;
param_2[4] = (float)(int)pcVar4[1] * fVar2;
param_2[5] = (float)(int)pcVar4[2] * fVar2;
...
pcVar4 = pcVar4 + 3;
```

### Definitive answer
- MCNR normals are stored as **signed bytes**, 3 bytes per vertex.
- The client converts them to float by **casting to int and scaling** by `_DAT_00a40360`.
- The MCNR pointer is read directly from the MCNK subchunk (`param_1 + 0x124`).

### Confidence
High.

---

## Vertex grid layout (9-8-9-8)

### Function addresses
- FUN_007c4f10: generic buffer build
- FUN_007c3f30 / FUN_007c4620 / FUN_007c4960: alternate buffer formats

### Decompiled evidence

```c
// FUN_007c4f10
local_18 = 0;
...
do {
  param_2 = 0;
  do {               // 9 outer vertices
    ...
    local_8 = local_8 + 1; // MCVT
    pcVar9 = pcVar9 + 3;   // MCNR
  } while ((int)param_2 < 9);

  if (local_18 < 8) {
    param_2 = 0;
    do {             // 8 inner vertices
      ...
      local_8 = local_8 + 1;
      pcVar9 = pcVar9 + 3;
    } while ((int)param_2 < 8);
  }
  local_18 = local_18 + 1;
} while (local_18 < 9);
```

### Definitive answer
- The client emits **9 outer vertices per row** for 9 rows, and **8 inner vertices** for the first 8 rows.
- This yields **17 rows** (9 outer + 8 inner) and **145 vertices** total.
- The ordering is row-major, alternating outer (9) then inner (8) rows.

### Confidence
High.

---

## Local position derivation

### Evidence highlights
- Vertex XY are derived from row/column multiplied by per-chunk step and adjusted by fixed offsets.
- Heights come from **MCVT + base height** (`param_1 + 0x84`).

```c
// FUN_007c3f30
param_2[2] = *(float *)(param_1 + 0x84) + *pfVar6; // base + MCVT
```

### Confidence
Medium (exact constant names vary by build, but formulas are explicit).
