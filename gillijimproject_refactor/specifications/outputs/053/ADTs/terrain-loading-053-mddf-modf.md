# Terrain Loading 0.5.3 MDDF and MODF Placements

## MDDF placements

### Function addresses
- `Create` (ADT load/parse) @ `0x006aad30`
- `CMap::CreateDoodadDef` @ `0x006a6cf0`
- `CMap::LoadDoodadNames` @ `0x00680c80`

### Decompiled evidence

```c
// 0x006aad30 (disassembly translated to C-like flow)
if (*(uint *)EDI != 'MDDF') { _SErrDisplayError_24(..., "mIffChunk->token=='MDDF'"); }
// count = (size * 0x38e38e39) >> 35  (magic divide)
MUL dword ptr [EDI + 0x4]
SHR EDI, 0x3
...
// copy MDDF payload into per-area storage
MOVSD/MOVSB from (MDDF + 8) to [this + 0x280]
```

```c
// CMap::CreateDoodadDef
NTempest::C44Matrix::Translate(&doodadDef->mat, position);
axis = {0.0, 0.0, 1.0};
NTempest::C44Matrix::Rotate(&doodadDef->mat, angle, &axis, true);
LoadDoodadModel(doodadDef, loadNow);
```

### Definitive answer
- The ADT MDDF chunk is validated and its entry count is derived by a magic divide on chunk size; payload bytes are copied into per-area storage for later instantiation.
- Doodad definitions are built from a position vector and a Z-axis rotation, then the model is loaded via `LoadDoodadModel`.

---

## MODF placements

### Function addresses
- `Create` (ADT load/parse) @ `0x006aad30`
- `CMap::CreateMapObjDef` @ `0x00681250`
- `CMap::LoadMapObjNames` @ `0x00687270`

### Decompiled evidence

```c
// 0x006aad30 (disassembly translated to C-like flow)
if (*(uint *)EDI != 'MODF') { _SErrDisplayError_24(..., "mIffChunk->token=='MODF'"); }
count = size >> 6; // divide by 0x40
// copy MODF payload into per-area storage
MOVSD/MOVSB from (MODF + 8) to [this + 0x294]
```

```c
// CMap::CreateMapObjDef
NTempest::C44Matrix::Translate(&mapObjDef->mat, position);
axis = {0.0, 0.0, 1.0};
NTempest::C44Matrix::Rotate(&mapObjDef->mat, angle, &axis, true);
mapObjDef->invMat = AffineInverse(mapObjDef->mat);
```

### Definitive answer
- The ADT MODF chunk is validated, its entry count is computed as size / 0x40, and entries are copied into per-area storage for later creation.
- Map object defs are built with a translate+Z-rotate transform and maintain an inverse matrix for queries.
