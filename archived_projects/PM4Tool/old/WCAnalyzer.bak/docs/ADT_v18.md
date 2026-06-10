# ADT/v18

From wowdev

(Redirected from ADT)

Jump to navigation Jump to search

## Contents

* 1 Terminology Reference
* 2 An important note about the coordinate system used
* 3 Map size, blocks, chunks

  + 3.1 Introduction
  + 3.2 Map size
  + 3.3 Player's speed
  + 3.4 ADT files and blocks
  + 3.5 Height
* 4 split files (Cata+)
* 5 MVER chunk
* 6 MHDR chunk
* 7 MCIN chunk (<Cata)
* 8 MTEX chunk
* 9 MDID
* 10 MHID
* 11 MMDX chunk
* 12 MMID chunk
* 13 MWMO chunk
* 14 MWID chunk
* 15 MDDF chunk

  + 15.1 Coordinate System Translation
* 16 MODF chunk
* 17 MH2O chunk (WotLK+)

  + 17.1 header
  + 17.2 attributes
  + 17.3 instances

    - 17.3.1 LiquidObject shit

      * 17.3.1.1 Alternate case determination
  + 17.4 instance vertex data

    - 17.4.1 Case 0, Height and Depth data
    - 17.4.2 Case 1, Height and Texture Coordinate data
    - 17.4.3 Case 2, Depth only data
    - 17.4.4 Case 3, Height, Depth and Texture Coordinates
  + 17.5 example, notes
* 18 MCNK chunk

  + 18.1 Terrain Holes
  + 18.2 MCVT sub-chunk
  + 18.3 MCLV sub-chunk (Cata+)
  + 18.4 MCCV sub-chunk (WotLK+)
  + 18.5 MCNR sub-chunk
  + 18.6 MCLY sub-chunk
  + 18.7 MCRF sub-chunk (<Cata)
  + 18.8 MCRD (Cata+)
  + 18.9 MCRW (Cata+)
  + 18.10 MCSH sub-chunk
  + 18.11 MCAL sub-chunk

    - 18.11.1 Uncompressed (4096)
    - 18.11.2 Uncompressed (2048)
    - 18.11.3 Compressed

      * 18.11.3.1 Sample C++ code
    - 18.11.4 Rendering
  + 18.12 MCLQ sub-chunk
  + 18.13 MCSE sub-chunk
  + 18.14 MCBB (MoP+)
  + 18.15 MCMT (Cata+)
  + 18.16 MCDD (Cata?+)
* 19 MFBO chunk (BC+)
* 20 MTXF chunk (WotLK+)
* 21 MTXP chunk (MoP?+)

  + 21.1 legion terrain shader excerpt
* 22 MTCG (Shadowlands+)
* 23 MBMH (MoP+)
* 24 MBBB (MoP+)
* 25 MBNV (MoP+)
* 26 MBMI (MoP+)
* 27 MAMP (Cata+)
* 28 MLHD (Legion+)
* 29 MLVH (Legion+)
* 30 MLVI (Legion+)
* 31 MLLL (Legion+)
* 32 MLND (Legion+)
* 33 MLSI (Legion+)
* 34 MLLD (Legion+)

  + 34.1 compression
  + 34.2 alphaTexture
* 35 MLLN (Legion+)
* 36 MLLV (Legion+)
* 37 MLLI (Legion+)
* 38 MLMD (Legion+)
* 39 MLMX (Legion+)
* 40 MLDD (Legion+)
* 41 MLDX (Legion+)
* 42 MLDL (Legion+)
* 43 MLFD (Legion+)
* 44 MBMB (Legion+)
* 45 MLMB (BfA+)
* 46 MLDB (Shadowlands+)
* 47 MWDR (Shadowlands+)
* 48 MWDS (Shadowlands+)

ADT files contain terrain and object information for map tiles. They have a chunked structure just like the WDT files.

A map tile is split up into 16x16 = 256 map chunks. (not the same as file chunks, although each map chunk will have its own file chunk :) ) So there will be a few initial data chunks to specify textures, objects, models, etc. followed by 256 MCNK (mapchunk) chunks :) Each MCNK chunk has a small header of its own, and additional chunks within its data block, following the same id-size-data format.

## Terminology Reference

| Term | Explanation |
| --- | --- |
| MapChunk (Chunk) | Refers to a chunk (terrain cell) represented by MCNK data chunk. |
| MapTile (Tile) | Referes to one .ADT file (<) or a group of ADT-related files representing one MapTile in WDT. |
| Sub-chunk (used in some software) | Commonly used term to describe an area in MCNK's heightmap. See MCNK holes for understanding. It is an abstraction, and is not represented by any data structure. |

## An important note about the coordinate system used

Wow's main coordinate system is right-handed; understanding it is very important in order to correctly interpret the ADT files.

It's important to remember that:

* The positive X-axis points north, the positive Y-axis points west.
* The Z-axis is vertical height, with 0 being sea level.
* The origin of the coordinate system is in the center of the map.
* The top-left corner of the map has X = 17066, Y = 17066
* The bottom-right corner of the map has X = -17066, Y = -17066
* The bottom-left corner of the map has X = -17006, Y = 17066
* The top-right corner of the map has X = 17006, Y = -17066

Just to be absolutely clear, assuming you playing a character that is not flying or swimming and is facing north:

* Forward = Vector3(1, 0, 0)
* Right = Vector3(0, -1, 0)
* Up = Vector3(0, 0, 1);

This is the coordinate system used internally in all of the network packets and on most chunks in ADT files. Here is an overview of the other used coordinate systems.

## Map size, blocks, chunks

### Introduction

All maps are divided into 64x64 blocks for a total of 4096 (some of which may be unused). Each block are divided into 16x16 chunks (not to be confused with for example the file chunks, such as the "MHDR" chunk.. Completely different thing!). While like I said blocks can be unused, each block will always use all of its 16x16 chunks.

### Map size

Each block is 533.33333 yards (1600 feet) in width and height. The map is divided into 64x64 blocks so the total width and height of the map will be 34133.33312 yards, however the origin of the coordinate system is at the center of the map so the minimum and maximum X and Y coordinates will be ±17066.66656).

Since each block has 16x16 chunks, the size of a chunk will be 33.3333 yards (100 feet).

### Player's speed

Basic running speed of a player (without any speed modifying effects) is 7.1111 yards/s (21.3333 feet/s). Player is able to reach one border of an ADT tile from another in 75 seconds. Thus, the fastest mounts (310%) can get over ADT size in 24.2 seconds.

### ADT files and blocks

There is an .adt file for each existing block. If a block is unused it won't have an .adt file. The file will be: World/Maps/<InternalMapName>/<InternalMapName>\_<BlockX>\_<BlockY>.adt.

* <InternalMapName> - MapRec::m\_Directory
* <BlockX> - Index of the tile on the X axis
* <BlockY> - Index of the tile on the Y axis

Converting ADT co-ords to block X/Y can be done with the following formula (where axis is x or y): floor((32 - (axis / 533.33333)))

### Height

The previous section details on X and Y limits only. The Z (height) limit is only implicit by stuff breaking slowly, like MFBO which is limited by using signed shorts, i.e. 2^15 being their max height. WDL/v18, while not mendatory, is probably the most important, it is also limited to 2^15 (-32k/+32k). There are some database files like DB/ZoneLight or world map related ones that also take height into account and may not be using floats depending on game version. DB/ZoneLight appears to be using -64000 and 64000 as the default if designers didn't put anything. DB/DungeonMapChunk seems to use -10000 for lower default. DB/UIMapAssignment wins with defaulting to -1000000 and 1000000. Generally, stay in the -32k/+32k range.

## split files (Cata+)

This section only applies to versions ≥ .

Beginning with Cataclysm, ADTs are split into multiple files: .adt (root), \_tex%d.adt (tex) and \_obj%d.adt (obj) with %d being the level of detail (0 or 1). Chunks are distributed over the files. To load a map, the client loads a set of three and treats them as one. While the distribution schema appears to be quite fixed, the client does not keep the semantics of which file is which and parses them all the same.

Note that \_tex1.adt files are now longer loaded since the introduction of WDT's MAID. The \_obj1.adt continue to be used.

The main difference content-wise is MCIN being gone, and MCNK in tex and obj files not having the header it has in root files.

added \_lod.adt (lod) files as another type. They are used for increased draw distance, this time including low quality versions of liquids and geometry as well (in the end, root lod bands).

I've written a short guide on how to implement \_lod.adt ADTLodImplementation Zee

## MVER chunk

* split files: all

```
struct MVER { uint32_t version; };
```

## MHDR chunk

* split files: root
* Contains offsets relative to &MHDR.data in the file for specific chunks. WoW only takes this for parsing the ADT file.

```
struct SMMapHeader { enum MHDRFlags { mhdr_MFBO = 1, // contains a MFBO chunk. mhdr_northrend = 2, // is set for some northrend ones. }; uint32_t flags; uint32_t mcin; // MCIN*, Cata+: obviously gone. probably all offsets gone, except mh2o(which remains in root file). uint32_t mtex; // MTEX* uint32_t mmdx; // MMDX* uint32_t mmid; // MMID* uint32_t mwmo; // MWMO* uint32_t mwid; // MWID* uint32_t mddf; // MDDF* uint32_t modf; // MODF* uint32_t mfbo; // MFBO* this is only set if flags & mhdr_MFBO. uint32_t mh2o; // MH2O* uint32_t mtxf; // MTXF* uint8_t mamp_value; // Cata+, explicit MAMP chunk overrides data uint8_t padding[3]; uint32_t unused[3]; } mhdr;
```

## MCIN chunk (<Cata)

This section only applies to versions ≤ . No longer possible due to split files.

* Pointers to MCNK chunks and their sizes.

```
struct SMChunkInfo { uint32_t offset; // absolute offset. uint32_t size; // the size of the MCNK chunk, this is refering to. uint32_t flags; // always 0. only set in the client., FLAG_LOADED = 1 union { char pad[4]; uint32_t asyncId; // not in the adt file. client use only }; } mcin[16*16];
```

## MTEX chunk

This section only applies to versions < (8.1.0.28294). MTEX has been replaced with file data ids in MDID and MHID chunks.

* split files: tex
* List of textures used for texturing the terrain in this map tile.

```
struct MTEX { char filenames[0]; // zero-terminated strings with complete paths to textures. Referenced in MCLY. };
```

## MDID

This section only applies to versions ≥ (8.1.0.27826).

split files: tex0

```
struct { /*0x00*/ uint32_t file_data_id; // _s.blp } diffuse_texture_ids[];
```

## MHID

This section only applies to versions ≥ (8.1.0.27826).

split files: tex0

```
struct { /*0x00*/ uint32_t file_data_id; // _h.blp; 0 if there is none } height_texture_ids[diffuse_texture_ids.size];
```

## MMDX chunk

* split files: obj
* List of filenames for M2 models that appear in this map tile.

```
struct MMDX { char filenames[0]; // zero-terminated strings with complete paths to models. Referenced in MMID. };
```

## MMID chunk

* split files: obj
* List of offsets of model filenames in the MMDX chunk.

```
struct MMID { uint32_t offsets[0]; // filename starting position in MMDX chunk. These entries are getting referenced in the MDDF chunk. };
```

## MWMO chunk

* split files: obj
* List of filenames for WMOs (world map objects) that appear in this map tile.

```
struct MWMO { char filenames[0]; // zero-terminated strings with complete paths to models. Referenced in MWID. };
```

## MWID chunk

* split files: obj
* List of offsets of WMO filenames in the MWMO chunk.

```
struct MWID { uint32_t offsets[0]; // filename starting position in MWMO chunk. These entries are getting referenced in the MODF chunk. };
```

## MDDF chunk

* split files: obj
* Placement information for doodads (M2 models). Additional to this, the models to render are referenced in each MCRF chunk.

```
enum MDDFFlags { mddf_biodome = 1, // this sets internal flags to | 0x800 (WDOODADDEF.var0xC) mddf_shrubbery = 2, // the actual meaning of these is unknown to me. maybe biodome is for really big M2s. 6.0.1.18179 seems // not to check for this flag mddf_unk_4 = 0x4, // Legion+u mddf_unk_8 = 0x8, // Legion+u mddf_unk_10 = 0x10, // Shadowlands+u observed in Shadowlands 9.2.7, sets flag 0x4 on the PVS Doodad. May have been added earlier like the rest. SMDoodadDef::Flag_liquidKnown = 0x20, // Legion+u mddf_entry_is_filedata_id = 0x40, // Legion+u nameId is a file data id to directly load mddf_unk_100 = 0x100, // Legion+u mddf_accept_proj_textures = 0x1000, // Legion+u }; struct SMDoodadDef { /*0x00*/ uint32_t nameId; // references an entry in the MMID chunk, specifying the model to use. // if flag mddf_entry_is_filedata_id is set, a file data id instead, ignoring MMID. /*0x04*/ uint32_t uniqueId; // this ID should be unique for all ADTs currently loaded. Best, they are unique for the whole map. Blizzard has // these unique for the whole game. /*0x08*/ C3Vectori position; // This is relative to a corner of the map. Subtract 17066 from the non vertical values and you should start to see // something that makes sense. You'll then likely have to negate one of the non vertical values in whatever // coordinate system you're using to finally move it into place. /*0x14*/ C3Vectori rotation; // degrees. This is not the same coordinate system orientation like the ADT itself! (see history.) /*0x20*/ uint16_t scale; // 1024 is the default size equaling 1.0f. /*0x22*/ uint16_t flags; // values from enum MDDFFlags. /*0x24*/ } doodadDefs[];
```

* How to compute a matrix to map M2 to world coordinates

Math is the same as for MODF, only with scale being added.

Example in js with gl-matrix:

```
createPlacementMatrix : function(mddf) { var TILESIZE = 533.333333333; var posx = 32 * TILESIZE - mddf.position[0]; var posy = mddf.position[1]; var posz = 32 * TILESIZE - mddf.position[2]; var placementMatrix = mat4.create(); mat4.identity(placementMatrix); mat4.rotateX(placementMatrix, placementMatrix, glMatrix.toRadian(90)); mat4.rotateY(placementMatrix, placementMatrix, glMatrix.toRadian(90)); mat4.translate(placementMatrix, placementMatrix, [posx, posy, posz]); mat4.rotateY(placementMatrix, placementMatrix, glMatrix.toRadian(mddf.rotation[1] - 270)); mat4.rotateZ(placementMatrix, placementMatrix, glMatrix.toRadian(-mddf.rotation[0])); mat4.rotateX(placementMatrix, placementMatrix, glMatrix.toRadian(mddf.rotation[2] - 90)); mat4.scale(placementMatrix, placementMatrix, [mddf.scale / 1024, mddf.scale / 1024, mddf.scale / 1024]); return placementMatrix; }
```

#### Coordinate System Translation

Here is an overview of common coordinate systems. Imagine you are a bird, looking down on the ground, oriented to the north.

| Coordinates | Axis X | Axis Y | Axis Z | Orientation | Vector | Remarks |
| --- | --- | --- | --- | --- | --- | --- |
| WDT/ADT (Terrain) | North ← South | West ← East | Up | RH | Vector3.Forward \* x + Vector3.Left \* y + Vector3.Up \* z |
| M2/WMO (Models) | North → South | West → East | Up | RH | Vector3.Backward \* x + Vector3.Right \* y + Vector3.Up \* z |
| MDDF/MODF (Placement) | West ← East | Up | North ← South | RH | Vector3.Left \* x' + Vector3.Up \* y + Vector3.Forward \* z' Rotation x: around West/East axis Rotation y: around Up axis Rotation z: around North/South axis for LH renderers, all rotations have to be negated (made anti-clockwise) | x' = 32 \* TILESIZE - x ; z' = 32 \* TILESIZE - z |
| Renderer | Axis X | Axis Y | Axis Z | Orientation | Vector Definition | Remarks |
| Blender | West → East | North ← South | Up | RH | Vector3.Right = (1,0,0) ; Vector3.Forward = (0,1,0) ; Vector3.Up = (0,0,1) | Vector.Left = -Vector.Right ; ... |
| Unreal | West → East | North → South | Up | LH | Vector3.Right = (1,0,0) ; Vector3.Backward = (0,1,0) ; Vector3.Up = (0,0,1) |
| Unity | West → East | Up | North ← South | LH | Vector3.Right = (1,0,0) ; Vector3.Up = (0,1,0) ; Vector3.Forward = (0,0,1) |
| Direct3D | West → East | Up | North ← South | LH | Vector3.Right = (1,0,0) ; Vector3.Up = (0,1,0) ; Vector3.Forward = (0,0,1) |
| OpenGL (WebGL) | West → East | Up | North → South | RH | Vector3.Right = (1,0,0) ; Vector3.Up = (0,1,0) ; Vector3.Backward = (0,0,1) |
| Vulkan | West → East | Up | North → South | RH | Vector3.Right = (1,0,0) ; Vector3.Up = (0,1,0) ; Vector3.Backward = (0,0,1) |

How do I read this table?

:   Every 3D renderer, be it a modelling software, a game engine, or a 3D rendering framework, defines its own axes and orientation. This was done back in the days, when companies wanted to be as incompatible to another as possible. Invert an axis, swap two axes and importing/exporting without pre- and post processing will become impossible. Or require expensive conversion tools. History. In general, there are 2 systems with either Y-up, or Z-up and both can be either left-, or right-handed. Blizzard uses a Z-up, left handed coordinate system with swapped x- and y-axis. This coordinate system is also used by the WDT/ADT terrain system. But more importantly, it is the coordinate system between the world servers and the clients to position players, NPCs, all game objects in the world. A 3D vector with the 3 components x, y, and z is positioned inside the game as follows: The x-coordinate lies on the north-south axis with values increasing going to the north. North hereby is the actual north orientation inside the game. Similarly, the y-coordinate lies on the west-east axis with values increasing going to the west. The z-coordinate defines the height above the ground level. Blizzard also uses a yard-based system, 1 unit represents 1 yard.

:   Based on that orientation, the table gives a transformation into other renderers, while keeping directions intact. The north-south axis defines a forward/backward direction. The west-east axis defines a left/right direction. It is defined by the renderer if these axes are labelled x, y, z and if any are negated (going north to south, or going south to north). The table shows, what each renderer defines as its x-, y- and z-axis. Select the correct unit vectors, take the vector expression, and as a result you get the transformed vector that has the correct orientation inside the renderer. That means, for example in Unity or Blender, all models will be correctly positioned to the diverse left-view, right-view, top-view and all the other views.
