# Terrain Loading 0.5.3 WDT and ADT Deep Dive

## WDT top-level chunk scan

### Function addresses
- `CMap::LoadWdt` @ `0x0067fde0`

### Decompiled evidence

```c
// CMap::LoadWdt
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token != 'MVER') { _SErrDisplayError_24(..., "iffChunk.token=='MVER'"); }
SFile::Read(wdtFile, &version, 4, 0, 0, 0);
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token != 'MPHD') { _SErrDisplayError_24(..., "iffChunk.token=='MPHD'"); }
SFile::Read(wdtFile, &header, 0x80, 0, 0, 0);
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token != 'MAIN') { _SErrDisplayError_24(..., "iffChunk.token=='MAIN'"); }
SFile::Read(wdtFile, &areaInfo, 0x10000, 0, 0, 0);
LoadDoodadNames();
LoadMapObjNames();
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token == 'MODF') { ... SFile::Read(wdtFile, &smMapObjDef, 0x40, 0, 0, 0); ... }
```

### Definitive answer
- WDT parsing asserts `MVER`, `MPHD`, then `MAIN`, reads `header` (0x80) and `areaInfo` (0x10000), and conditionally reads `MODF`.
- Name tables for MDX/WMO are loaded immediately after `MAIN` via `LoadDoodadNames()` and `LoadMapObjNames()`.

---

## WDT MAIN grid indexing and ADT offsets

### Function addresses
- `CMap::PrepareArea` @ `0x00684a30`

### Decompiled evidence

```c
// CMap::PrepareArea
uVar6 = param_2 * 0x40 + param_1;
if ((&areaTable)[uVar6] != 0) { _SErrDisplayError_24(..., "areaTable[index] == 0"); }
...
CMapArea::Load(this, (SMAreaInfo *)(&areaInfo + uVar6 * 4));
```

### Definitive answer
- The WDT MAIN grid is treated as a 64x64 table (`index = param_2 * 0x40 + param_1`).
- Each MAIN entry is treated as a 4-byte `SMAreaInfo` record and passed into `CMapArea::Load` as the per-tile ADT entrypoint.

---

## WDT MDNM and MONM name tables

### Function addresses
- `CMap::LoadWdt` @ `0x0067fde0`

### Decompiled evidence

```c
// CMap::LoadWdt
SFile::Read(wdtFile, &areaInfo, 0x10000, 0, 0, 0);
LoadDoodadNames();
LoadMapObjNames();
```

### Definitive answer
- MDX (`MDNM`) and WMO (`MONM`) name tables are read immediately after `MAIN` in the WDT load sequence.

---

## WDT MODF (WMO-only maps)

### Function addresses
- `CMap::LoadWdt` @ `0x0067fde0`

### Decompiled evidence

```c
// CMap::LoadWdt
SFile::Read(wdtFile, &iffChunk, 8, 0, 0, 0);
if (iffChunk.token == 'MODF') {
	SFile::Read(wdtFile, &smMapObjDef, 0x40, 0, 0, 0);
	smMapObjDef.uniqueId = uniqueId;
	uniqueId = uniqueId - 1;
	pCVar1 = CreateMapObjDef(&smMapObjDef, &pos);
	...
}
```

### Definitive answer
- The WDT-level `MODF` chunk is optional; when present, 0x40-byte entries are read and converted into map object defs.

---

## Embedded ADT entrypoint from WDT

### Function addresses
- `CMap::PrepareArea` @ `0x00684a30`

### Decompiled evidence

```c
// CMap::PrepareArea
uVar6 = param_2 * 0x40 + param_1;
CMapArea::Load(this, (SMAreaInfo *)(&areaInfo + uVar6 * 4));
```

### Definitive answer
- The ADT entrypoint is derived from the WDT `MAIN` grid entry (`areaInfo[index]`) and passed to `CMapArea::Load` for tile-level parsing.

---

## ADT top-level chunk scan

### Function addresses
- `Create` (ADT load/parse) @ `0x006aad30`

### Decompiled evidence

```c
// 0x006aad30 (disassembly translated to C-like flow)
if (*(uint *)EBX != 'MHDR') { _SErrDisplayError_24(..., "mIffChunk->token=='MHDR'"); }
...
if (*(uint *)ESI != 'MCIN') { _SErrDisplayError_24(..., "mIffChunk->token=='MCIN'"); }
...
if (*(uint *)ESI != 'MTEX') { _SErrDisplayError_24(..., "mIffChunk->token=='MTEX'"); }
...
if (*(uint *)EDI != 'MDDF') { _SErrDisplayError_24(..., "mIffChunk->token=='MDDF'"); }
...
if (*(uint *)EDI != 'MODF') { _SErrDisplayError_24(..., "mIffChunk->token=='MODF'"); }
```

### Definitive answer
- The ADT load/parse path asserts `MHDR`, `MCIN`, `MTEX`, `MDDF`, and `MODF` in order, indicating the expected top-level ADT chunk sequence for 0.5.3.

---

## ADT MCIN table and MCNK offsets

### Function addresses
- `Create` (ADT load/parse) @ `0x006aad30`

### Decompiled evidence

```c
// 0x006aad30
if (*(uint *)ESI != 'MCIN') { _SErrDisplayError_24(..., "mIffChunk->token=='MCIN'"); }
// memcpy MCIN payload into object storage (+0x29c)
memcpy(this + 0x29c, ESI + 8, *(uint *)(ESI + 4));
```

### Definitive answer
- The MCIN payload is copied into per-area storage, and subsequent chunk loads reference this table for MCNK offsets.

---

## MCNK chunk base and scan

### Function addresses
- `CMapChunk::SyncLoad` @ `0x00698d20`
- `CMap::PrepareChunk` @ `0x00684be0`

### Decompiled evidence

```c
// CMapChunk::SyncLoad
SFile::SetFilePointer(CMap::wdtFile, this->fileOffset, 0, 0);
SFile::Read(CMap::wdtFile, &DAT_00e6e5e4, this->fileSize, 0, 0, 0);
if (DAT_00e6e5e4 != 'MCNK') { _SErrDisplayError_24(..., "iffChunk->token=='MCNK'"); }
*param_1 = (SMChunk *)&DAT_00e6e5ec;
if (DAT_00e6ea70 != 'MCLY') { _SErrDisplayError_24(..., "iffChunk->token=='MCLY'"); }
*param_2 = (SMLayer *)&DAT_00e6ea78;
*param_3 = (uchar *)(DAT_00e6ea74 + 0xe6ea80 + *(int *)(&DAT_00e6ea7c + DAT_00e6ea74));
*param_4 = *param_3 + (*param_1)->sizeShadow;
```

```c
// CMap::PrepareChunk
(this->aIndex).x = param_2;
(this->aIndex).y = param_3;
this->infoIndex = param_3 * 0x10 + param_2;
CMapChunk::Load(this, param_1->chunkInfo + this->infoIndex);
```

### Definitive answer
- Each MCNK is read from the WDT file using the per-chunk `fileOffset` and `fileSize` values, then validated by token.
- `CMapChunk::Load` receives the MCNK offset/size entry from the MCIN table (`chunkInfo[index]`).

---

## MCNK header layout (observed vs not observed)

### Function addresses
- `CMapChunk::SyncLoad` @ `0x00698d20`
- `CMapChunk::Load` (MCNK parse path) @ `0x00698e10`

### Decompiled evidence

```c
// CMapChunk::SyncLoad
*param_1 = (SMChunk *)&DAT_00e6e5ec;
*param_4 = *param_3 + (*param_1)->sizeShadow;
```

```c
// 0x00698e10 (disassembly translated to C-like flow)
if (*(uint *)EBX != 'MCNK') { ... }
...
// header fields used later in the parse
if ((*(byte *)EBX & 0x1) != 0 && (DAT_00e4046c & 0x40) != 0) {
	shadowPtr = ptrAfterMCRF;
	shadowParam = *(uint *)(EBX + 0x34);
	CreateShadow(shadowPtr, shadowParam);
}
...
layerCount = *(uint *)(EBX + 0x10);
refCountA = *(uint *)(EBX + 0x14);
refCountB = *(uint *)(EBX + 0x3c);
CreateRefs(refCountA, refCountB, ...);
```

### Definitive answer
- The MCNK header is treated as `SMChunk`, and the `sizeShadow` field is read directly from it to advance into subsequent subchunk data.
- Multiple MCNK header fields are read directly in the MCNK parse path, including a flags byte at offset 0x00 (bit 0 gating shadow work), a layer count at 0x10, and additional parameters at offsets 0x14, 0x34, and 0x3c.

### Observed field usage (0x00698e10)

| Offset | Usage observed | Notes |
| --- | --- | --- |
| 0x00..0x7F | Header block | Parser skips 0x80 bytes after MCNK token+size. |
| 0x00 | Bit test (`& 0x1`) | Gates shadow path in MCNK parse loop. |
| 0x10 | Layer count | Used to iterate MCLY entries. |
| 0x14 | Doodad ref count | Passed into `0x0069a0c0` and used as loop count for `CreateDoodadDef`. |
| 0x2C | Size parameter | Used to advance pointer to the liquid block. |
| 0x34 | Shadow size parameter | Used to advance pointer and stored to `this->field_0x130` before `CreateShadow`. |
| 0x3C | Map obj ref count | Passed into `0x0069a0c0` and used as loop count for `CreateMapObjDef`. |
| 0x44..0x50 | 4 dwords copied | Copied into `this->field_0x158` block. |
| 0x54 | Copied to `this->field_0x168` | Single dword. |
| 0x58 | Copied to `this->field_0x16c` | Single dword. |

---

## MCNK subchunk mapping (by FourCC)

### Function addresses
- `CMapChunk::SyncLoad` @ `0x00698d20`
- `Create` (MCNK validation path) @ `0x00698e10`
- `0x006997e0` (height/vertex build)
- `0x00699b60` (normal decode)
- `0x00699df0` (layer + alpha setup)
- `0x00699fb0` (shadow texture setup)
- `0x0069a040` (alpha shadow setup)
- `CMapChunk::UnpackAlphaBits` @ `0x0069a5f0`
- `CMapChunk::UnpackShadowBits` @ `0x0069a6b0`
- `CMapChunk::UnpackAlphaShadowBits` @ `0x0069a430`
- `CMapChunk::CreateChunkLayerTex` @ `0x0069a320`
- `CMapChunk::CreateChunkShaderTex` @ `0x0069a390`
- `CMap::AllocChunkLiquid` @ `0x00691860`

### Decompiled evidence

```c
// CMapChunk::SyncLoad
if (DAT_00e6e5e4 != 'MCNK') { _SErrDisplayError_24(..., "iffChunk->token=='MCNK'"); }
if (DAT_00e6ea70 != 'MCLY') { _SErrDisplayError_24(..., "iffChunk->token=='MCLY'"); }
```

```c
// 0x00698e10 (disassembly translated to C-like flow)
if (*(uint *)EBX != 'MCNK') { _SErrDisplayError_24(..., "iffChunk->token=='MCNK'"); }
...
if (*(uint *)(EBX + 0x1c0) != 'MCLY') { _SErrDisplayError_24(..., "iffChunk->token=='MCLY'"); }
...
if (*(uint *)(EBX + 0x??) != 'MCRF') { _SErrDisplayError_24(..., "iffChunk->token=='MCRF'"); }
```

```c
// 0x00698e10 (disassembly translated to C-like flow)
// Fixed-size jumps before MCLY
EBX += 0x8;   // skip MCNK token+size
EBX += 0x80;  // MCNK header size
EBX += 0x244; // MCVT payload
EBX += 0x1c0; // MCNR payload (with padding)
// EBX now points at MCLY chunk header
// Later, liquid data pointer is computed as:
//   liquidPtr = (end of MCRF) + header[0x34] + header[0x2c]
```

```c
// 0x00698e10 -> 0x006997e0
// Height/vertex build called with pointer derived from MCNK data block
CALL 0x006997e0
```

```c
// 0x006997e0 (height/vertex build)
// Reads 9x9 heights, then 8x8 inner heights (sequential float reads)
for (row = 0; row < 9; row++) {
	for (col = 0; col < 9; col++) {
		height = *heights++;
		...
	}
}
for (row = 0; row < 8; row++) {
	for (col = 0; col < 8; col++) {
		height = *heights++;
		...
	}
}
```

```c
// 0x00699b60 (normal decode - reads 3 signed bytes per normal)
for (row = 0; row < 9; row++) {
	for (col = 0; col < 9; col++) {
		nx = (sbyte)src[0] * kNormScale;
		ny = (sbyte)src[1] * kNormScaleY;
		nz = (sbyte)src[2] * kNormScale;
		dst->x = nx; dst->y = ny; dst->z = nz;
		src += 3; dst++;
	}
	for (col = 0; col < 8; col++) {
		nx = (sbyte)src[0] * kNormScale;
		ny = (sbyte)src[1] * kNormScaleY;
		nz = (sbyte)src[2] * kNormScale;
		dst->x = nx; dst->y = ny; dst->z = nz;
		src += 3; dst++;
	}
}
```

```c
// 0x00699df0 (layer + alpha setup)
if (this->nLayers > 4) { _SErrDisplayError_24(...); }
layer = AllocLayer();
layer->owner = this;
layer->texId = mclay->textureIndex;
layer->flags = mclay->flags;
layer->alphaOffset = mclay->alphaOffset + mcalBase;
layer->effectId = mclay->effectId;
if ((layer->flags & 1) != 0) {
	layer->alphaTex = CreateAlphaTexture(...);  // 0x00697d90
	BuildAlpha(layer);                           // 0x0069a320
}
```

```c
// CMapChunk::CreateChunkLayerTex
layer->tex = CMap::GetTex();
UnpackAlphaBits(layer->tex->pixels, layer->offsAlpha);
```

```c
// CMapChunk::CreateChunkShaderTex
for (i = 0; i < 4; i++) {
	alpha[i] = (i < nLayers && (layer[i]->props & 0x100)) ? layer[i]->offsAlpha : 0;
}
UnpackAlphaShadowBits(shaderTex->pixels, shadowBits, alpha, shadowOffs);
```

```c
// 0x00699fb0 (shadow texture setup)
this->shadowData = shadowBytes;
if (shadowBytes != 0) {
	this->shadowTex = CreateShadowTexture(...);  // 0x00697fe0
	BuildShadow(this);                           // 0x0069a2d0
}
```

```c
// 0x0069a040 (alpha shadow setup)
if (CWorld::shadowMipLevel == 1) { w = h = 0x20; }
else { w = h = 0x40; }
this->shaderGxTexture = AllocShadowGxTex(this, UpdateShaderGxTexture);
CreateChunkShaderTex(this);
GxTexUpdate(this->shaderGxTexture, 0, 0, w, h, 1);
```

```c
// 0x00698e10 (liquid allocation path)
if ((chunkHeader->flags & mask) != 0) {
	liquid = CMap::AllocChunkLiquid();
	liquid->height = *(float *)EBX;
	...
}
```

```c
// 0x00698e10 (disassembly translated to C-like flow)
// per-liquid slot loop (4 slots)
if ((*(uint *)headerFlags & bit) != 0) {
	if (liquidSlot == 0) { liquidSlot = CMap::AllocChunkLiquid(); }
	// copy MCLQ-like payload into liquid slot
	memcpy(liquidSlot + 0x8,  EBX,     0x288);
	memcpy(liquidSlot + 0x290,EBX+0x288,0x40);
	memcpy(liquidSlot + 0x2d4,EBX+0x2c8,0x50);
	EBX += 0x318;
}
bit <<= 1; // 0x4, 0x8, 0x10, 0x20 across 4 slots
```

```c
// 0x0069a0c0 (CreateRefs)
// doodad refs: count = *(uint *)(EBX + 0x14)
for (i = 0; i < refCountDoodad; i++) {
	refId = refTable[i];
	CreateDoodadDef(...);   // 0x006805e0
}

// map obj refs: count = *(uint *)(EBX + 0x3c)
for (i = 0; i < refCountMapObj; i++) {
	refId = refTable[i];
	CreateMapObjDef(...);   // 0x00681250
}
```

```c
// CMapChunk::UnpackAlphaBits
if (CWorld::alphaMipLevel == 1) {
	// 32x32: read byte-per-pixel, with +0x21 stride per row
	for (y = 0; y < 32; y++) {
		for (x = 0; x < 32; x++) {
			pixels[idx++] = (alpha[src++] << 28) | 0x00ffffff;
		}
		src += 0x21;
	}
} else {
	// 64x64: 4-bit packed alpha, two pixels per byte
	for (i = 0; i < 0x1000; i++) {
		if ((i & 1) == 0) value = (alphaByte << 28);
		else { value = (alphaByte & 0xf0) << 24; alphaByte++; }
		pixels[i] = value | 0x00ffffff;
	}
}
```

```c
// CMapChunk::UnpackShadowBits
if (CWorld::shadowMipLevel != 1) {
	for (y = 0; y < 0x40; y++) {
		for (x = 0; x < 0x40; x++) {
			pixels[idx] = (shadowBits[idx >> 3] & bitMask) ? 0xffffffff : 0x0;
		}
		if ((y & 1) == 0) {
			// build 32x32 bitmask row
			if (shadowBits[idx >> 3] & bitMask) shadowMask[bitIdx] |= bit;
		}
	}
} else {
	// 32x32 shadow path
	for (y = 0; y < 0x20; y++) {
		for (x = 0; x < 0x20; x++) {
			pixels[idx] = (shadowBits[idx >> 3] & bitMask) ? 0xffffffff : 0x0;
			shadowMask[idx >> 5] |= bit;
		}
	}
}
```

```c
// CMapChunk::CreateAlphaShadow
if (CWorld::shadowMipLevel == 1) { w = h = 0x20; }
else { w = h = 0x40; }
this->shaderGxTexture = AllocShadowGxTex(this, UpdateShaderGxTexture);
CreateChunkShaderTex(this);
GxTexUpdate(this->shaderGxTexture, 0, 0, w, h, 1);
```

```c
// CMapChunk::UnpackAlphaShadowBits
// Combines per-layer alpha (up to 4 sources) with shadow bits into RGBA output.
// Uses bit masks and per-mip stride rules driven by CWorld::shadowMipLevel.
```

### Definitive answer
- The MCNK loader explicitly validates `MCNK`, `MCLY`, and `MCRF` tokens in the parse path shown above.
- The MCNK loader advances by fixed sizes for MCVT (0x244) and MCNR (0x1c0) before validating the MCLY token, indicating fixed-size alpha-era layouts.
- MCNR normals are decoded from signed bytes into float normals with two nested loops (9x9 outer + 8x8 inner), matching the Alpha non-interleaved layout.
- MCVT heights are consumed as 9x9 outer + 8x8 inner float sequences to build the chunk vertex grid and bounds.
- MCLY drives per-layer setup, with alpha data referenced by `alphaOffset` (MCAL base + per-layer offset). Alpha textures are only created when the layer flag is set.
- Shadow data is treated as a separate subchunk payload and converted into a texture via a dedicated build path.
- Alpha+shadow shader textures are built via `CreateAlphaShadow`, which allocates a shadow GX texture and updates it at 32x32 or 64x64 based on `CWorld::shadowMipLevel`.
- MCAL alpha decoding uses a 4-bit packed 64x64 path or a 32x32 path with row stride 0x21 when `CWorld::alphaMipLevel == 1`.
- MCSH shadow decoding uses packed bitmasks and produces both a full pixel buffer and a compact 32x32 mask; shader shadow textures are allocated at 32x32 or 64x64 depending on `CWorld::shadowMipLevel`.
- Alpha+shadow combined decode exists (`UnpackAlphaShadowBits`) and merges multiple alpha sources plus shadow into RGBA output.
- Per-layer alpha textures are created via `CreateChunkLayerTex`, which directly feeds MCAL data to `UnpackAlphaBits`.
- Shader texture generation combines up to four layer alpha sources (flag `0x100`) with shadow bits via `UnpackAlphaShadowBits`.
- Liquid allocation is gated by MCNK flags; the allocation path calls `CMap::AllocChunkLiquid` and seeds liquid state from chunk data, consistent with a legacy liquid subchunk (likely MCLQ).
- Liquid payload parsing iterates 4 slots and bulk-copies a fixed-size block (~0x318 bytes) into each liquid slot, matching a legacy MCLQ-style packed structure.
- Liquid payload parsing iterates 4 slots gated by flags bits 0x4, 0x8, 0x10, 0x20 and bulk-copies a fixed-size block (~0x318 bytes) into each liquid slot.
- Doodad ref count is read from header offset 0x14 and used to call `CreateDoodadDef` for each entry.
- Map object ref count is read from header offset 0x3c and used to call `CreateMapObjDef` for each entry.
