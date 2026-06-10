# Alpha Format: wowdev.wiki vs Ghidra Ground-Truth Comparison

**Purpose**: Identify discrepancies between community documentation and actual client behavior  
**Sources**:  
- `reference_data/wowdev.wiki/Alpha.md` (Community docs, last edited 2018-12-23)
- `WoWRollback/docs/Alpha-Client-Ghidra-Analysis.md` (Ghidra decompilation, 2025-12-28)
- `WoWRollback/docs/WMO-v14-Ghidra-Verified.md` (Ghidra decompilation, 2025-12-28)

---

## Executive Summary

| Category | Correct | Incorrect/Missing | Unverified |
|----------|---------|-------------------|------------|
| WDT Structure | 6 | 2 | 1 |
| ADT/MCNK Structure | 8 | 3 | 2 |
| WMO v14 Structure | - | 6 | - |

**Overall**: Community docs are ~70% accurate but have critical errors in structure sizes.

---

## 1. WDT Format Comparison

### ✅ CORRECT in wowdev.wiki

| Item | wowdev.wiki | Ghidra Verified |
|------|-------------|-----------------|
| MVER version | 18 (0x12) | ✅ `assert(version == 0x12)` |
| Monolithic format | WDT+ADTs in one file | ✅ Sequential reading confirmed |
| MAIN size | 64×64 = 4096 entries | ✅ `0x10000` bytes read |
| MAIN entry size | 16 bytes | ✅ `sizeof(SMAreaInfo) = 16` |
| MDNM/MONM | Null-terminated strings | ✅ String parsing confirmed |
| MODF optional | Only for WMO-based maps | ✅ Token check before reading |

### ⚠️ INCORRECT/INCOMPLETE in wowdev.wiki

| Item | wowdev.wiki Says | **Ghidra Ground-Truth** | Impact |
|------|------------------|-------------------------|--------|
| **MPHD size** | `uint8_t pad[112]` (128 total implied) | **128 bytes** - but client reads `0x80` explicitly | ✅ Size correct |
| **MPHD offsets usage** | "offsDoodadNames, offsMapObjNames" imply seeking | **NO SEEKING** - client reads sequentially, offsets are metadata only | 🔴 Writers may add unnecessary padding |
| **Chunk reading** | Not specified | **SEQUENTIAL** - no padding between chunks | 🔴 Critical for writers |

### 📝 MISSING from wowdev.wiki

| Item | Ghidra Finding |
|------|----------------|
| **FourCC storage** | Reversed on disk (`MVER` → `REVM`) |
| **Exact chunk order** | MVER → MPHD → MAIN → MDNM → MONM → [MODF] → [Tiles] |
| **Name counting** | `nMapObjNames = actual_count + 1` (for trailing null) |

---

## 2. ADT/MCNK Format Comparison

### ✅ CORRECT in wowdev.wiki

| Item | wowdev.wiki | Ghidra Verified |
|------|-------------|-----------------|
| MCNK header size | 128 bytes | ✅ `0x80` terrain header |
| MCVT layout | 81 outer + 64 inner (not interleaved) | ✅ Different from 3.x |
| MCVT absolute heights | Not relative to header | ✅ Confirmed |
| MCNR layout | Same as MCVT (outer first) | ✅ `145 * 3 bytes + 13 pad` |
| MCRF direct indices | Points to MDNM/MONM, not MMID/MWID | ✅ No MMID/MWID in alpha |
| MCLQ no chunk header | Sub-chunks have no name/size | ✅ Headerless sub-chunks |
| Liquid flags | LQ_RIVER=4, LQ_OCEAN=8, LQ_MAGMA=16 | ✅ `assert(type < 4)` |

### ⚠️ INCORRECT/INCOMPLETE in wowdev.wiki

| Item | wowdev.wiki Says | **Ghidra Ground-Truth** | Impact |
|------|------------------|-------------------------|--------|
| **MCNK.areaid** | Lists as field | **Uses Unknown3 field** - actual field offset may differ | 🟡 Need field mapping |
| **Liquid types** | 3 flags shown | **4 types** (0=Water, 1=Ocean, 2=Magma, 3=Slime) | 🟡 Slime type missing |
| **MDDF entry size** | "See ADT_v18" (implies 36 bytes) | **64 bytes (0x40)** - verified via `CMapArea::Create` | 🔴 Critical size error |
| **MODF entry size** | "See ADT_v18" | **64 bytes (0x40)** - verified via `CMapArea::Create` | 🔴 Size not specified |

### 📝 MISSING from wowdev.wiki

| Item | Ghidra Finding |
|------|----------------|
| **Coordinate system** | Alpha uses (X, Z, Y) - Y-up, unlike LK (X, Y, Z) Z-up |
| **MDDF/MODF position swap** | Y and Z floats at offsets 12/16 must be swapped for LK |
| **Chunk linking** | MHDR offsets are relative to MHDR data start, not file start |
| **World coordinate formula** | `worldPos = -(chunkOffset * 33.333) + 17066.666` with X/Y swap |

---

## 3. WMO v14 Format Comparison

**Note**: wowdev.wiki Alpha.md doesn't detail WMO v14 internal structure, only mentions it exists.

### 🔴 CRITICAL DISCREPANCIES (from general WMO docs)

| Item | Old Community Docs | **Ghidra Ground-Truth** | Impact |
|------|-------------------|-------------------------|--------|
| **MOGI entry size** | 32 bytes | **40 bytes** (`size / 0x28`) | 🔴 Parser corruption |
| **MOPY entry size** | 2 bytes | **4 bytes** (`size >> 2`) | 🔴 Face count wrong |
| **Index chunk name** | MOVI | **MOIN** (0x4D4F494E) | 🔴 Chunk not found |
| **MODD entry size** | Various | **40 bytes** (`size / 0x28`) | 🔴 Doodad parsing fails |
| **MFOG entry size** | Not documented | **48 bytes** (`size / 0x30`) | 🟡 Missing fog data |
| **MOLV chunk** | Not documented | **Present** (8 bytes each, lightmap UVs) | 🟡 Missing lightmaps |

### WMO v14 Verified Structure Sizes

| Chunk | Ghidra Size | Calculation |
|-------|-------------|-------------|
| MOMT (materials) | 44 bytes | `size / 0x2c` |
| MOGI (group info) | 40 bytes | `size / 0x28` |
| MOPT (portals) | 20 bytes | `size / 0x14` |
| MOLT (lights) | 32 bytes | `size >> 5` |
| MODS (doodad sets) | 32 bytes | `size >> 5` |
| MODD (doodad defs) | 40 bytes | `size / 0x28` |
| MFOG (fog) | 48 bytes | `size / 0x30` |
| MOPY (poly flags) | 4 bytes | `size >> 2` |
| MOBA (batches) | 24 bytes | `size / 0x18` |
| MOBN (BSP nodes) | 16 bytes | `size >> 4` |

---

## 4. WDL Format (Not in wowdev.wiki Alpha.md)

### Ghidra Discovery - WDL Structure

| Chunk | Size | Purpose |
|-------|------|---------|
| MVER | 4 bytes | Version = 18 (0x12) |
| MAOF | 16384 bytes | 64×64 offset table |
| MARE | 1090 bytes | 545 int16 heights per tile |

### Height Grid
- **545 heights** = 17×17 outer + 16×16 inner
- Stored as **int16** (signed short)
- Used for distant terrain (beyond FarClip)

---

## 5. Render System (Not in wowdev.wiki)

### Ghidra Discoveries

| Parameter | Value | Source |
|-----------|-------|--------|
| FarClip range | 177-777 yards | `FarClipCallback` |
| ADT width | 533.33 yards | `SetFarClip` |
| Verts per chunk | 145 | Buffer allocation |
| Indices per chunk | 768 | Buffer allocation |

---

## 6. Minimap System (Not in wowdev.wiki)

### Ghidra Discovery

The client uses `Textures/Minimap/md5translate.txt` for minimap lookup!

```
Format: md5hash<TAB>filename
```

This explains the dual minimap locations:
- `Textures/Minimap/` - Actual textures (referenced by md5translate.txt)
- `World/Minimaps/` - Legacy/alternative location

---

## 7. MCNK Header Offsets (Ghidra Verified)

From `CMapChunk::Create @ 00698e99`:

| Offset | Field | Type | Notes |
|--------|-------|------|-------|
| 0x0C | indexX | uint32 | Chunk X (0-15) |
| 0x10 | indexY | uint32 | Chunk Y (0-15) |
| 0x18 | nLayers | uint32 | Max 4 layers |
| 0x1C | nDoodadRefs | uint32 | MCRF count |
| 0x3C | sizeShadow | uint32 | Shadow map size |
| 0x88 | MCVT | 145 floats | Height data |
| 0x2CC | MCNR | 145×3 bytes | Normal data |
| 0x48C | MCLY | IFF header | Layers start |

### World Coordinate Formula

From `CMapChunk::CreateVertices @ 006997e0`:

```c
CHUNK_SIZE = 33.333333f   // ___real_42055555
MAP_CENTER = 17066.6666f  // ___real_46855555 = 32 * 533.333333

// NOTE: X and Y are SWAPPED in the calculation!
worldCorner.x = -(chunkOffset.y * CHUNK_SIZE) + MAP_CENTER
worldCorner.y = -(chunkOffset.x * CHUNK_SIZE) + MAP_CENTER
```

---

## 8. PM4 Format (Server-Side Pathfinding)

> [!IMPORTANT]
> **PM4 files are NOT read by any WoW client**. They are server-side pathfinding data.

### Origin
PM4 files were accidentally shipped to players during a **Cataclysm 4.0.0 PTR build in 2010**.
They originate from the server infrastructure, not the client.

### Key Facts
- **File type**: Server-side navigation/pathfinding mesh
- **Era**: Cataclysm 4.0.0 (2010), NOT Alpha 0.5.3
- **Client support**: None - neither Alpha, Cataclysm, nor any retail client reads PM4
- **Purpose**: Server uses for NPC pathing, line-of-sight, collision
- **Accidental release**: PTR build inadvertently included server files

### Structure (Reverse-Engineered)
Since no client code exists, PM4 structure is reverse-engineered from file analysis:
- MSLK: Surface links
- MSUR: Surface definitions
- MSVT: Vertices
- MSVI: Vertex indices
- MSCN: Collision normals
- MPRL/MPRR: Placement references

---

## 9. Recommended Documentation Updates

### For wowdev.wiki Alpha.md

1. **Add sequential reading note** - Client reads chunks sequentially, no seeking
2. **Add coordinate system note** - Alpha uses Y-up (X,Z,Y), LK uses Z-up (X,Y,Z)
3. **Fix MDDF/MODF sizes** - Both are **64 bytes** in Alpha (not 36)
4. **Add 4th liquid type** - Slime (type 3)
5. **Add WDL section** - Document distant terrain format
6. **Add FourCC reversal note** - On-disk storage is reversed
7. **Add MCNK header offsets** - indexX@0x0C, indexY@0x10, MCVT@0x88, MCNR@0x2CC
8. **Add world coordinate formula** - With X/Y swap documentation

### For WMO v14 Documentation

1. **Fix MOGI size** - 40 bytes, not 32
2. **Fix MOPY size** - 4 bytes, not 2
3. **Fix index chunk** - MOIN, not MOVI
4. **Add MOLV chunk** - Lightmap UVs
5. **Add MFOG chunk** - 48 bytes per fog entry
6. **Fix MODD size** - 40 bytes

---

## 10. Verification Checklist

To verify our parsers are 100% correct:

### WDT Parser
- [x] Reads chunks sequentially (no seeking to MPHD offsets) - Verified via `LoadWdt`
- [x] MAIN entries are 16 bytes - Verified `0x10000 / 4096`
- [x] No padding between chunks - Verified sequential reads
- [ ] FourCCs are reversed on read

### ADT/MCNK Parser
- [x] MCVT has 81+64 non-interleaved floats - Verified via `CreateVertices`
- [x] MCNR has 145×3 + 13 padding bytes - Verified via `CreateNormals`
- [x] MDDF entries are **64 bytes** - Verified via `CMapArea::Create` (0x40)
- [x] MODF entries are **64 bytes** - Verified via `CMapArea::Create` (0x40)
- [x] indexX @ 0x0C, indexY @ 0x10 - Verified via `CMapChunk::Create`
- [ ] Y/Z swap for coordinate conversion

### WMO v14 Parser
- [ ] Version check for 14 (0x0E)
- [x] MOGI entries are 40 bytes - Verified
- [x] MOPY entries are 4 bytes - Verified
- [x] Index chunk is MOIN (not MOVI) - Verified
- [ ] MOLV chunk is parsed (lightmap UVs)
- [x] MODD entries are 40 bytes - Verified
- [x] MFOG entries are 48 bytes - Verified

---

*This document should be used to update community documentation and verify parser implementations against ground-truth from Ghidra decompilation.*
