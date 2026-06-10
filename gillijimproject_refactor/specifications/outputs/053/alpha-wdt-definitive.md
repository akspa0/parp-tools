# Alpha WDT Definitive Reference (0.5.3.3368)

This document is the single, detailed reference for the Alpha WDT (monolithic terrain) format in WoW Alpha 0.5.3.3368. It consolidates all Ghidra-backed findings from the reverse engineering work and replaces the previously scattered notes.

## Scope, build, and evidence
- Target build: WoW Alpha 0.5.3 (3368), WoWClient.exe x86 with PDB symbols.
- Primary evidence from binary analysis:
  - `CMap::LoadWdt` (0x0067fde0)
  - `CMap::PrepareArea` (0x00684a30)
  - `Create` ADT load/parse (0x006aad30)
  - `CMapChunk::SyncLoad` (0x00698d20)
  - `CMapChunk::Create` (0x00698e99)
  - `CreateRefs` (0x0069a0c0)
  - `CMap::CreateDoodadDef` (0x006a6cf0)
  - `CMap::CreateMapObjDef` (0x00681250)
  - `CMapChunk::UnpackAlphaBits` (0x0069a5f0)
  - `CMapChunk::UnpackShadowBits` (0x0069a6b0)
  - `CMapChunk::UnpackAlphaShadowBits` (0x0069a430)
- FourCC order: forward byte order ("MCNK" = 0x4D434E4B).

## Alpha-specific model (monolithic WDT)
Alpha 0.5.3 stores all terrain data in a single WDT file. The WDT MAIN table contains offsets and sizes for per-tile embedded ADT blocks, and each tile block includes 256 MCNK chunks. There are no separate .ADT files in this build. The client always reads chunk data from the WDT file handle and does not open per-tile ADT files in the 0.5.3 code path.

## High-level runtime load flow (as executed)
This section describes the exact call sequence the client uses to go from WDT file to renderable MCNKs. It is verbose on purpose to match the in-binary behavior.

1. `CMap::LoadWdt` reads MVER, MPHD, MAIN in order and validates each token.
2. `CMap::LoadWdt` loads name tables by calling `LoadDoodadNames()` (MDNM) and `LoadMapObjNames()` (MONM) immediately after MAIN.
3. `CMap::LoadWdt` optionally reads one WDT-level MODF entry (dungeon case).
4. `CMap::PrepareArea` indexes MAIN using `index = tileY * 64 + tileX` and calls `CMapArea::Load` with that entry.
5. `Create` (ADT load/parse) validates MHDR, MCIN, MTEX, MDDF, MODF for the embedded tile block.
6. `CMapChunk::SyncLoad` reads each MCNK from the WDT using the per-chunk offset and size from MCIN.
7. `CMapChunk::Create` parses MCNK, builds vertices and normals, creates layers, instantiates references, applies alpha/shadow, and allocates liquids if present.

## WDT top-level structure
The loader enforces this exact sequence and validates each token. All FourCC values are checked in forward byte order and will hard-error if mismatched.
1. MVER
2. MPHD (0x80 bytes)
3. MAIN (0x10000 bytes)
4. MDNM (doodad name table)
5. MONM (WMO name table)
6. Optional MODF (single WMO placement for dungeon-style maps)

### MVER
```c
struct MVER {
    uint32 version; // read but not otherwise validated
};
```

Loader behavior:
- Reads the 8-byte chunk header, checks token == "MVER".
- Reads 4 bytes of version payload.
- No value gate beyond the token check.

### MPHD
```c
struct MPHD {
    uint32 flags;     // bit 0: has global WMO (dungeon path)
    uint32 reserved[31];
};
// Size: 0x80 bytes
```

Loader behavior:
- Reads 0x80 bytes unconditionally.
- Uses bit 0 to switch to a "global WMO" dungeon path.
- Other flags are currently unknown (no direct references located in this path).

### MAIN (tile table)
```c
struct WdtMainEntry {
    uint32 flags;    // bit 0: tile present
    uint32 asyncId;  // runtime/unused in 0.5.3
    uint32 offset;   // file offset to per-tile block
    uint32 size;     // size of per-tile block
};
// 4096 entries (64x64), total 0x10000 bytes
```

#### MAIN indexing
The loader computes `index = tileY * 0x40 + tileX` and uses that index directly in the in-memory MAIN array. This means Y varies fastest in the on-disk table.

Runtime behavior detail:
- If the entry is already marked in the area table, the client errors out.
- Otherwise it passes the MAIN entry to `CMapArea::Load` as the tile entrypoint.

### MDNM / MONM (name tables)
- Loaded immediately after MAIN.
- Null-terminated full paths.
- Alpha uses direct string indices in MCRF; there is no MMID/MWID indirection.

Token order and behavior:
- MDNM and MONM are parsed by `LoadDoodadNames` and `LoadMapObjNames` respectively.
- Each loader validates its expected token and reads the full chunk payload in one call.
- The name index table is built after the payload is read.

### WDT-level MODF (optional)
- If present, entries are 0x40 bytes each (WMO placement definition).
- This marks the map as a dungeon and switches to absolute positioning for lights.

```c
struct MODF {
    uint32 nameId;       // index into MONM
    uint32 uniqueId;
    float pos[3];
    float rot[3];
    float aaBoxMin[3];
    float aaBoxMax[3];
    uint16 flags;
    uint16 doodadSet;
    uint16 nameSet;
    uint16 padding;
};
// Size: 0x40 bytes
```

Runtime behavior:
- A single MODF entry is read and immediately converted into a map object definition.
- The WMO placement is translated and rotated around the Z axis.
- The map is marked as a dungeon (used by lighting logic).

## Per-tile embedded ADT block (inside WDT)
Each MAIN entry points at an embedded ADT block. The parser expects these chunks in strict order and will hard-fail on a mismatch:
1. MHDR
2. MCIN
3. MTEX
4. MDDF
5. MODF
6. MCNK (256 entries referenced by MCIN)

### MHDR
```c
struct MHDR {
    uint32 offsInfo;   // MCIN
    uint32 offsTex;    // MTEX
    uint32 sizeTex;
    uint32 offsDoo;    // MDDF
    uint32 sizeDoo;
    uint32 offsMob;    // MODF
    uint32 sizeMob;
    uint8  pad[36];
};
```

Behavior:
- MHDR is validated by token check.
- Offsets are used as internal relative pointers within the embedded tile block.

### MCIN
- 256 entries for a 16x16 chunk grid.
- The payload is copied directly into per-area storage and used to locate MCNK offsets and sizes.
- Each entry contains an MCNK offset and size (exact per-entry layout is not needed to parse when using the client logic).

Runtime behavior:
- The MCIN payload is copied in one `memcpy` into the per-area structure.
- Later, `CMapChunk::Load` receives `chunkInfo[index]` to supply the MCNK offset and size.

### MTEX
- Texture filename table (Alpha layout matches later versions).
- Parsed by the ADT load path after MCIN.

Behavior:
- MTEX is validated by token.
- The client stores it for later lookup by MCLY entries.

### MDDF (doodad placements)
```c
struct MDDF {
    uint32 nameId;     // index into MDNM
    uint32 uniqueId;
    float pos[3];
    float rot[3];
    uint16 scale;      // scale / 1024.0f
    uint16 flags;
};
// Size: 0x24 bytes
```

Creation flow:
- Chunk token validated, count derived from size / 0x24.
- Payload copied into per-area storage.
- Placement built with Translate + Z-rotate, then model load.

Runtime details:
- `CreateDoodadDef` builds a `C44Matrix` by translating to the placement position.
- Rotation is applied around the Z axis using a fixed axis vector (0, 0, 1).
- The model is loaded immediately or deferred based on load mode.

### MODF (map object placements)
```c
struct MODF {
    uint32 nameId;     // index into MONM
    uint32 uniqueId;
    float pos[3];
    float rot[3];
    float aaBoxMin[3];
    float aaBoxMax[3];
    uint16 flags;
    uint16 doodadSet;
    uint16 nameSet;
    uint16 padding;
};
// Size: 0x40 bytes
```

Creation flow:
- Chunk token validated, count derived from size / 0x40.
- Payload copied into per-area storage.
- Placement built with Translate + Z-rotate, and an inverse matrix cached.

Runtime details:
- `CreateMapObjDef` uses the same translate + Z-rotate flow as doodads.
- The inverse matrix is computed and cached for spatial queries.

## MCNK (map chunk) definitive layout
Each MCNK is read from the monolithic WDT using MCIN offsets and sizes. The header is 0x88 bytes (136) after the IFF header. The MCNK chunk begins with a standard 8-byte IFF header (token + size).

```c
struct MCNK_Header {
    uint32 magic;         // 0x00 = "MCNK"
    uint32 size;          // 0x04 = payload size
    uint32 flags;         // 0x08
    uint32 indexX;        // 0x0C
    uint32 indexY;        // 0x10
    uint32 nLayers;       // 0x14
    uint32 nDoodadRefs;   // 0x18
    uint32 ofsHeight;     // 0x1C (MCVT)
    uint32 ofsNormal;     // 0x20 (MCNR)
    uint32 ofsLayer;      // 0x24 (MCLY)
    uint32 ofsRefs;       // 0x28 (MCRF)
    uint32 ofsAlpha;      // 0x2C (MCAL)
    uint32 sizeAlpha;     // 0x30
    uint32 ofsShadow;     // 0x34 (MCSH)
    uint32 sizeShadow;    // 0x38
    uint32 areaid;        // 0x3C
    uint32 nMapObjRefs;   // 0x40
    uint32 holes;         // 0x44 (16-bit mask)
    uint16 predTex[8];    // 0x48
    uint8  noEffectDoodad[8]; // 0x58
    uint32 ofsSndEmitters; // 0x60 (MCSE)
    uint32 nSndEmitters;   // 0x64
    uint32 ofsLiquid;      // 0x68 (inline liquid block)
    uint32 sizeLiquid;     // 0x6C
    float  position[3];    // 0x70 (X, Z, Y)
    uint32 ofsMCCV;        // 0x7C (unused in 0.5.3)
    uint32 padding[2];     // 0x80
    // header ends at 0x88
};
```

### Header usage highlights
- All offsets are relative to MCNK start (not relative to end of header).
- The header pointer begins at MCNK + 0x08 (after token + size).
- `nDoodadRefs` and `nMapObjRefs` drive `CreateRefs` loops for doodad and WMO creation.
- `sizeShadow` is used to advance the pointer to alpha/shadow merge and liquid data.
- `position[3]` is stored in X, Z, Y order (this is how `CreateVertices` receives it).

### MCNK flags
```
0x01 = has shadow map
0x02 = impassable
0x04 = liquid type 0 (water)
0x08 = liquid type 1 (ocean)
0x10 = liquid type 2 (magma)
0x20 = liquid type 3 (slime)
```

### Fixed layout before MCLY
The parser uses fixed-size jumps to reach MCLY:
- IFF header: 0x08
- MCNK header: 0x80
- MCVT payload: 0x244 bytes (145 floats)
- MCNR payload: 0x1C0 bytes (145 normals + padding)
- MCLY header is at offset 0x48C from MCNK start

This means MCVT and MCNR are stored without their own IFF headers in Alpha 0.5.3.

## MCNK subchunks and payloads

### MCVT (heights)
- No chunk header in Alpha.
- 145 floats total, ordered as 81 outer (9x9) then 64 inner (8x8).
- Heights are absolute (no base height bias).

Runtime detail:
- The vertex build reads 9x9 outer heights first, then 8x8 inner heights sequentially.

### MCNR (normals)
- No chunk header in Alpha.
- 145 normals, each 3 signed bytes.
- Ordered like MCVT (outer then inner).

Runtime detail:
- Normals are scaled by constants during unpack and stored as float vectors.

### MCLY (layers)
```c
struct MCLY_Entry {
    uint32 textureId;
    uint32 props;      // alpha usage and flags
    uint32 offsAlpha;  // MCAL offset
    uint16 effectId;
    uint16 pad;
};
```

Behavior:
- `nLayers` must be <= 4 (hard check).
- `props & 0x100` controls whether a layer contributes to the combined shader texture.

Runtime detail:
- Each layer references a texture by index into MTEX.
- Alpha offsets are resolved by adding the MCAL base pointer.

### MCRF (references)
- Direct indices into MDNM and MONM (no MMID/MWID indirection).
- `nDoodadRefs` and `nMapObjRefs` drive two loops in `CreateRefs`:
  - Doodad refs use `CMap::CreateDoodadDef`.
  - Map object refs use `CMap::CreateMapObjDef`.

Runtime detail:
- The doodad loop uses the count at header offset 0x18.
- The WMO loop uses the count at header offset 0x40.

### MCAL (alpha)
- 4-bit packed alpha, 64x64 (4096 pixels), row-major order.
- Two pixels per byte, low nibble first then high nibble.
- If `CWorld::alphaMipLevel == 1`, a 32x32 path is used.

Unpack detail:
- The 64x64 path is a linear loop over 4096 pixels.
- Even indices read low nibble, odd indices read high nibble, then advance source.
- Output is ARGB with alpha in the high byte and RGB forced to 0xFFFFFF.

### MCSH (shadow)
- Data size is `sizeShadow`.
- Shadow mip size is driven by `CWorld::shadowMipLevel` (32 or 64).
- Shadow is unpacked before shader texture combine.

Runtime detail:
- The shadow data is stored in a chunk-local buffer before texture build.
- The combined alpha+shadow texture is updated via GX after construction.

### MCSE (sound emitters)
- Entry stride: 0x34 bytes (52 bytes).
- Count is `nSndEmitters`, offset is `ofsSndEmitters`.

Emitter layout (from parsing loop):
```c
struct SoundEmitterData {
    uint32 soundPointID;
    uint32 soundNameID;
    float pos[3];
    float minDistance;
    float maxDistance;
    float cutoffDistance;
    uint16 startTime;
    uint16 endTime;
    uint16 mode;
    uint16 groupSilenceMin;
    uint16 groupSilenceMax;
    uint16 playInstancesMin;
    uint16 playInstancesMax;
    uint8  loopCountMin;
    uint8  loopCountMax;
    uint16 interSoundGapMin;
    uint16 interSoundGapMax;
};
// 0x34 bytes
```

### Inline liquid block ("MCLQ" later)
Alpha stores liquid data inline; there is no MCLQ chunk header.

Per liquid instance (one per flag bit set) is 0x324 bytes:
```c
struct MclqInline {
    float minHeight;      // 0x00
    float maxHeight;      // 0x04
    struct {
        float height;
        uint32 data;      // per-vertex flags or packed data
    } verts[81];          // 0x08..0x28F (9x9)
    float tiles[16];      // 0x290..0x2CF (4x4 grid)
    uint32 nFlowvs;       // 0x2D0
    float flowvs[20];     // 0x2D4..0x323 (2 structs x 10 floats)
};
```

Behavior:
- Four possible liquid slots (types 0..3) are allocated based on MCNK flag bits 0x04..0x20.
- Each liquid slot is filled by copying the inline block.

Inline block parsing detail:
- The copy loop advances by 0x324 bytes per liquid instance.
- The per-vertex array is 81 entries with 8 bytes each (height + data).
- The tiles array is 16 floats (4x4 grid).
- Flow vectors are read as 20 floats total, interpreted as two 10-float structures.

## Texture and shader combine path (MCAL + MCSH)
- `CMapChunk::CreateChunkLayerTex` builds per-layer alpha textures by calling `UnpackAlphaBits`.
- `CMapChunk::CreateChunkShaderTex` merges alpha and shadow into a shader texture.
- The combined texture is 32x32 or 64x64 depending on `CWorld::shadowMipLevel`.

Runtime detail:
- For each layer, alpha textures are allocated only if the layer has the alpha flag set.
- Shader textures combine up to four layers of alpha with shadow bits.

## WMO and doodad instantiation (from refs)
- Doodads use Translate + Z-rotate and then `LoadDoodadModel`.
- WMOs use Translate + Z-rotate and cache an inverse matrix.
- WMO names are loaded from MONM and prepared once the WMO has finished loading.

Lighting side-effects:
- When a WMO is prepared, lights are created per group if the light flag is not set.

## Practical parse sequence (single tile)
1. Read MAIN entry for tile offset and size.
2. Parse MHDR, then MCIN, MTEX, MDDF, MODF (token-checked order).
3. For each MCIN entry, read MCNK at offset/size.
4. In MCNK:
   - Validate token
   - Parse header
   - Read MCVT and MCNR fixed blocks
   - Parse MCLY, MCRF, MCAL, MCSH using header offsets and sizes
   - Parse MCSE if `nSndEmitters > 0`
   - Parse inline liquid blocks for each liquid flag bit set

## Reference offsets and sizes (quick facts)
- WDT MPHD size: 0x80 bytes
- WDT MAIN size: 0x10000 bytes
- WDT MAIN entry size: 0x10 bytes
- MCNK header size: 0x88 bytes
- MCVT size: 0x244 bytes
- MCNR size: 0x1C0 bytes
- MCLY header location: MCNK + 0x48C
- Inline liquid block size: 0x324 bytes per instance

## Differences from later versions (high confidence)
- Monolithic WDT with inline MCNK data (no external ADTs).
- MAIN entries contain offsets and sizes (16 bytes each).
- MCNK uses fixed-size MCVT/MCNR with no subchunk headers.
- MCRF directly indexes MDNM/MONM (no MMID/MWID indirection).
- Liquids are inline (no separate MCLQ chunk header).

## Open points (non-blocking)
- MPHD flags beyond bit 0/1 are still unknown.
- `predTex` and `noEffectDoodad` semantics are not fully decoded.

## Quick checklist for implementers
- Treat WDT as monolithic; use MAIN offsets for tiles and MCIN for chunks.
- Validate FourCC tokens in forward order.
- MCNK header is 0x88 bytes including IFF header; offsets are relative to MCNK start.
- MCVT/MCNR have no embedded chunk headers.
- Liquid data is inline and fixed-size (0x324 per liquid instance).
