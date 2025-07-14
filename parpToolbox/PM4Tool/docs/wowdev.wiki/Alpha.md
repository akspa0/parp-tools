# Alpha

From wowdev

Jump to navigation Jump to search

This section only applies to versions < (0.6.0.3592).

This page contains information about the map/terrain files used in the alpha. For information added on 2012-03-24 : files I looked at come from (0.5.3.3368) (Mjollnà).

## Contents

* 1 WDT

  + 1.1 MVER
  + 1.2 MPHD
  + 1.3 MAIN
  + 1.4 MDNM
  + 1.5 MONM
  + 1.6 MODF (optional)
  + 1.7 MHDR
  + 1.8 MCIN
  + 1.9 MTEX
  + 1.10 MDDF
  + 1.11 MODF (in ADT part)
  + 1.12 MCNK

    - 1.12.1 (MCVT) sub-chunk
    - 1.12.2 (MCNR) sub-chunk
    - 1.12.3 MCLY sub-chunk
    - 1.12.4 MCRF sub-chunk
    - 1.12.5 (MCSH) sub-chunk
    - 1.12.6 (MCAL) sub-chunk
    - 1.12.7 (MCLQ) sub-chunk
    - 1.12.8 (MCSE) sub-chunk

## WDT

In the Alpha the ADTs and the WDT were just one big file with this structure :

### MVER

```
struct { uint32_t version; // 18, just as all others } mver;
```

### MPHD

```
struct SMMapHeader { uint32_t nDoodadNames; uint32_t offsDoodadNames; // MDNM uint32_t nMapObjNames; uint32_t offsMapObjNames; // MONM uint8_t pad[112]; };
```

### MAIN

* Map tile table. Needs to contain 64x64 = 4096 entries of sizeof(SMAreaInfo)

```
struct SMAreaInfo { uint32_t offset; // absolute offset MHDR uint32_t size; // offset relative to MHDR start (so value can also be read as : all ADT chunks total size from MHDR to first MCNK) MCNK uint32_t flags; // FLAG_LOADED = 0x1 is the only flag, set at runtime uint8_t pad[4]; };
```

### MDNM

Filenames Doodads. Zero-terminated strings with complete paths to models.

### MONM

Filenames WMOS. Zero-terminated strings with complete paths to models.

### MODF (optional)

Only one instance is possible. It is usually used by WMO based maps which contain no ADT parts with the exception of RazorfenDowns. If this chunk exists, the client marks the map as a dungeon and uses absolute positioning for lights.

See the new files. (ADT_v18.md)

### MHDR

The start of what is now the ADT files.

```
struct SMAreaHeader { uint32_t offsInfo; // MCIN uint32_t offsTex; // MTEX uint32_t sizeTex; uint32_t offsDoo; // MDDF uint32_t sizeDoo; uint32_t offsMob; // MODF uint32_t sizeMob; uint8_t pad[36]; };
```

### MCIN

256 Entries, so a 16\*16 Chunkmap.

See the new files. (ADT_v18.md)

### MTEX

See the new files. (ADT_v18.md)

### MDDF

See the new files. (ADT_v18.md)

### MODF (in ADT part)

See the new files. (ADT_v18.md)

### MCNK

The header is 128 bytes like later versions, but information inside is placed slightly differently. Offsets are relative to the end of MCNK header.

```
struct SMChunk { uint32_t flags; // See SMChunkFlags uint32_t indexX; uint32_t indexY; float radius; uint32_t nLayers; uint32_t nDoodadRefs; uint32_t offsHeight; // MCVT uint32_t offsNormal; // MCNR uint32_t offsLayer; // MCLY uint32_t offsRefs; // MCRF uint32_t offsAlpha; // MCAL uint32_t sizeAlpha; uint32_t offsShadow; // MCSH uint32_t sizeShadow; uint32_t areaid; uint32_t nMapObjRefs; uint16_t holes; uint16_t pad0; uint16_t predTex[8]; uint8_t noEffectDoodad[8]; uint32_t offsSndEmitters; // MCSE uint32_t nSndEmitters; uint32_t offsLiquid; // MCLQ uint8_t pad1[24]; }; enum SMChunkFlags { FLAG_SHADOW = 0x1, FLAG_IMPASS = 0x2, FLAG_LQ_RIVER = 0x4, FLAG_LQ_OCEAN = 0x8, FLAG_LQ_MAGMA = 0x10, };
```

#### (MCVT) sub-chunk

No chunk name and no size.

It's composed of the usual 145 floats, but their order is different : alpha vertices are not interleaved... Which means there are all outer vertices first (all 81), then all inner vertices (all 64) in MCVT (and not 9-8-9-8 etc.). Unlike 3.x format, MCVT have absolute height values (no height relative to MCNK header 0x70).

```
struct { float height[9*9 + 8*8]; } mcvt;
```

#### (MCNR) sub-chunk

No chunk name and no size.

Same as 3.x format, except values order which is like alpha MCVT : all outer values first, then all inner ones.

```
struct SMNormal { uint8_t n[145][3]; uint8_t pad[13]; };
```

#### MCLY sub-chunk

See the new files. (ADT_v18.md)

```
struct SMLayer { uint32_t textureId; uint32_t props; // only use_alpha_map is implemented uint32_t offsAlpha; uint16_t effectId; uint8_t pad[2]; };
```

#### MCRF sub-chunk

Since there are no MMDX/MWMO MMID/MWID in alpha ADT, MCRF entries directly point to index in MDNM and MONM chunks.

#### (MCSH) sub-chunk

No chunk name and no size.

See the new files. (ADT_v18.md)

#### (MCAL) sub-chunk

No chunk name and no size.

See the new files. (ADT_v18.md)

#### (MCLQ) sub-chunk

No chunk name and no size.

See the new files. (ADT_v18.md)

#### (MCSE) sub-chunk

No chunk name and no size.

See the new files. (ADT_v18.md)

Retrieved from "https://wowdev.wiki/index.php?title=Alpha&oldid=26535"

## Navigation menu

* This page was last edited on 23 December 2018, at 22:40.

* Privacy policy
* About wowdev
* Disclaimers