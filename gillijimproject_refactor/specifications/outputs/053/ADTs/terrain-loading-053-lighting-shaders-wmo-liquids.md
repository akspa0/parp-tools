# Terrain Loading 0.5.3 Lighting, Shaders, WMO, Liquids

## Lighting pipeline

### Function addresses
- `CMap::PrepareMapObjDefs` @ `0x00684170`

### Decompiled evidence

```c
// CMap::PrepareMapObjDefs
if ((pCVar6->field_0x75 & 2) == 0) {
	PrepareMapObjDef(pCVar6, mapObj);
}
...
if ((pCVar3->field_0x74 & 0x40) == 0) {
	CreateMapObjDefLights(mapObj, group, pCVar6, pCVar3);
}
```

### Definitive answer
- Map object lighting is prepared during map object definition processing; lights are created per group when the light flag is unset.

---

## Shader selection and binding

### Function addresses
- `CMapChunk::CreateAlphaShadow` @ `0x0069a040`
- `CMapChunk::CreateChunkShaderTex` @ `0x0069a390`
- `CMapChunk::AllocAlphaGxTex` @ `0x00697d90`
- `CMapChunk::UnpackAlphaShadowBits` @ `0x0069a430`

### Decompiled evidence

```c
// CMapChunk::CreateAlphaShadow
if (CWorld::shadowMipLevel == 1) { w = h = 0x20; }
else { w = h = 0x40; }
this->shaderGxTexture = AllocShadowGxTex(this, UpdateShaderGxTexture);
CreateChunkShaderTex(this);
GxTexUpdate(this->shaderGxTexture, 0, 0, w, h, 1);
```

```c
// CMapChunk::CreateChunkShaderTex
for (i = 0; i < 4; i++) {
	alpha[i] = (i < nLayers && (layer[i]->props & 0x100)) ? layer[i]->offsAlpha : 0;
}
UnpackAlphaShadowBits(shaderTex->pixels, shadowBits, alpha, shadowOffs);
```

### Definitive answer
- Terrain shader textures are sized to 32x32 or 64x64 based on `CWorld::shadowMipLevel` and are updated via GX after `CreateChunkShaderTex` builds the combined alpha+shadow buffer.

---

## WMO rendering and placement hooks

### Function addresses
- `CMap::LoadMapObjNames` @ `0x00687270`
- `CMap::CreateMapObjDef` @ `0x00681250`
- `CMap::PrepareMapObjDefs` @ `0x00684170`

### Decompiled evidence

```c
// CMap::LoadMapObjNames
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token != 'MONM') { _SErrDisplayError_24(..., "iffChunk.token=='MONM'"); }
SFile::Read(wdtFile, mapObjNames.data, iffChunk.size, &bRead, 0, 0);
// build name index table
```

```c
// CMap::CreateMapObjDef
NTempest::C44Matrix::Translate(&mapObjDef->mat, position);
local_1c = {0.0, 0.0, 1.0};
NTempest::C44Matrix::Rotate(&mapObjDef->mat, angle, &local_1c, true);
```

```c
// CMap::PrepareMapObjDefs
if (mapObj->bLoaded == 0) { CMapObj::WaitLoad(mapObj); }
if ((mapObjDef->field_0x75 & 2) == 0) { PrepareMapObjDef(mapObjDef, mapObj); }
```

### Definitive answer
- WMO names are loaded from `MONM`, map object defs are built with a translate+Z-rotate matrix, and definitions are prepared once the backing WMO has finished loading.

---

## Liquid rendering pipeline

### Function addresses
- `CWorldScene::PrepareRenderLiquid` @ `0x0066a590`
- `CWorldScene::AddChunkLiquid` @ `0x0066b120`
- `CMap::AllocChunkLiquid` @ `0x00691860`

### Decompiled evidence

```c
// CWorldScene::PrepareRenderLiquid
CWorld::QueryLiquidStatus(&camQueryPos, &newLiquid, &level, &dir);
...
if ((newLiquid != camLiquid) && ((CWorld::enables & 0x2000000) != 0)) {
	Particulate::InitParticles(CWorld::particulate, newLiquid);
}
```

```c
// CWorldScene::AddChunkLiquid
if (param_2 > 3) { _SErrDisplayError_24(..., "type < LQ_LAST"); }
sortTable.table[dist].liquidList[param_2].Link(param_1);
```

### Definitive answer
- Liquid rendering is driven by a per-frame query of liquid status at the camera, with liquids sorted into render buckets by type and distance before rendering.
