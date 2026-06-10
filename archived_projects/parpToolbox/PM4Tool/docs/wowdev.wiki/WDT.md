# WDT

From wowdev

Jump to navigation Jump to search

WDT files specify exactly which map tiles map tiles are present in a world, if any, and can also reference a "global" WMO. They have a chunked file structure.

## Contents

* 1 MPHD chunk
* 2 MAIN chunk
* 3 MAID chunk
* 4 MWMO, MODF chunks

  + 4.1 MWMO chunk
  + 4.2 MODF chunk
* 5 MANM
* 6 _occ, _lgt

  + 6.1 occ

    - 6.1.1 MVER
    - 6.1.2 MAOI
    - 6.1.3 MAOH
  + 6.2 lgt

    - 6.2.1 MVER
    - 6.2.2 MPLT
    - 6.2.3 MPL2 (Legion+)
    - 6.2.4 MPL3 (Shadowlands+)
    - 6.2.5 MSLT (Legion+)
    - 6.2.6 MTEX (Legion+)
    - 6.2.7 MLTA
* 7 _fogs

  + 7.1 MVER
  + 7.2 VFOG
  + 7.3 VFEX
* 8 _mpv

  + 8.1 MVER
  + 8.2 PVPD
  + 8.3 PVMI
  + 8.4 PVBD

## MPHD chunk

Contains 8 32-bit integers.

```
struct SMMapHeader { uint32_t flags; #if ≥ (8.1.0.28294) uint32_t lgtFileDataID; uint32_t occFileDataID; uint32_t fogsFileDataID; uint32_t mpvFileDataID; uint32_t texFileDataID; uint32_t wdlFileDataID; uint32_t pd4FileDataID; // PD4 #else uint32_t something; uint32_t unused[6]; #endif } MPHD;
```

These are the only flags checked:

```
enum mphd_flags { wdt_uses_global_map_obj = 0x0001, // Use global map object definition. adt_has_mccv = 0x0002, // ≥ adds color: ADT.MCNK.MCCV. with this flag every ADT in the map _must_ have MCCV chunk at least with default values, else only base texture layer is rendered on such ADTs. adt_has_big_alpha = 0x0004, // shader = 2. Decides whether to use _env terrain shaders or not: funky and if MCAL has 4096 instead of 2048(?) adt_has_doodadrefs_sorted_by_size_cat = 0x0008, // if enabled, the ADT's MCRF(m2 only)/MCRD chunks need to be sorted by size category FLAG_LIGHTINGVERTICES = 0x0010, // ≥ adds second color: ADT.MCNK.MCLV -- This appears to be deprecated and forbidden in 8.x? adt_has_upside_down_ground = 0x0020, // ≥ Flips the ground display upside down to create a ceiling unk_0x0040 = 0x0040, // ≥ ??? -- Only found on Firelands2.wdt (but only since MoP) before Legion adt_has_height_texturing = 0x0080, // ≥ shader = 6. Decides whether to influence alpha maps by _h+MTXP: (without with) // also changes MCAL size to 4096 for uncompressed entries unk_0x0100 = 0x0100, // ≥ load _lod.adt if set, implicitly sets 0x8000 wdt_has_maid = 0x0200, // ≥ (8.1.0.28294) client will load ADT using FileDataID instead of filename formatted with "%s\\%s_%d_%d.adt" unk_0x0400 = 0x0400, unk_0x0800 = 0x0800, unk_0x1000 = 0x1000, unk_0x2000 = 0x2000, unk_0x4000 = 0x4000, unk_0x8000 = 0x8000, // ≥ implicitly set for map ids 0, 1, 571, 870, 1116 (continents). Affects the rendering of _lod.adt. If set, or if implicitly set by 0x100, maptextures and LOD are required, or else textures further than ~533 yards will not render. mask_vertex_buffer_format = adt_has_mccv | adt_has_mclv, // CMap::LoadWdt mask_render_chunk_something = adt_has_height_texturing | adt_has_big_alpha, // CMapArea::PrepareRenderChunk, CMapChunk::ProcessIffChunks };
```

See a list here.

The second integer as of WotLK is not ignored but stores something too:

```
for( int i = 0; i < WDT_MPHD.something/8; i++ ) { WDT_MAIN[i].flags = 0; WDT_MAIN[i].somedata = 0; }
```

The other bytes seem to be unused from what I can tell. In 6.0.1.18179 I was unable to find any code referencing something other than flags.

```
vertexBufferFormat = !(flags & adt_has_mccv) ? EGxVertexBufferFormat_PN : flags & adt_has_mclv ? EGxVertexBufferFormat_PNC2 : EGxVertexBufferFormat_PNC ; mcal_size = flags & mask_render_chunk_something ? 4096 : 2048;
```

This section only applies to versions ≥ (8.1.0.28294). Could have been changed earlieru.

As preparation for the removal of namehashes, this chunk underwent a change. The 'unused' array is now used for storing FileDataID references to associated WDT files and the 'something' has been removed.

## MAIN chunk

* Map tile table. Needs to contain 64x64 = 4096 entries of sizeof(SMAreaInfo) ( 8 ) bytes each.

```
struct SMAreaInfo // -> CMapAreaTableEntry { #if version < ? // until they maps are split into adts uint32_t offset; uint32_t size; #endif #if version > ? // beginning them being split into adts uint32_t Flag_HasADT : 1; #endif #if version ≥ uint32_t Flag_AllWater : 1; #endif uint32_t Flag_Loaded : 1; uint32_t asyncId; // only set during runtime. } map_area_info[64*64];
```

On Cataclysm, 2 on a tile displays "fake water" ingame. If only one "fake water tile" is activated, "fake water" will appear everywhere you don't have an ADT loaded. (seen on 4.3.4.15595)

## MAID chunk

This section only applies to versions ≥ (8.1.0.28294).

* Map filedataid table. Needs to contain 64x64 = 4096 entries of sizeof(MapFileDataIDs) ( 32 ) bytes each.

```
struct MapFileDataIDs { uint32_t rootADT; // reference to fdid of mapname_xx_yy.adt uint32_t obj0ADT; // reference to fdid of mapname_xx_yy_obj0.adt uint32_t obj1ADT; // reference to fdid of mapname_xx_yy_obj1.adt uint32_t tex0ADT; // reference to fdid of mapname_xx_yy_tex0.adt uint32_t lodADT; // reference to fdid of mapname_xx_yy_lod.adt uint32_t mapTexture; // reference to fdid of mapname_xx_yy.blp uint32_t mapTextureN; // reference to fdid of mapname_xx_yy_n.blp uint32_t minimapTexture; // reference to fdid of mapxx_yy.blp } MapFileDataIDs[64*64];
```

## MWMO, MODF chunks

For worlds with terrain, parsing ends here. If it has none, there is one MWMO and one MODF chunk here. The MODF chunk is limited to one entry. See the ADT format description for details.

### MWMO chunk

* A filename for one WMO (world map object) that appears in this map. A zero-terminated string. 0x100 is the maximum size for this chunk due to being copied into a stack allocated array (at least in MOP)! (including \0).

### MODF chunk

* Placement information for the global WMO. 64 bytes. Only one instance is possible.

```
Offset Type Description 0x00 uint32 ID -- unused, always uses MWMO's content instead 0x04 uint32 unique identifier for this instance -- unused, generates uid dynamically 0x08 3 floats Position (X,Y,Z) 0x14 3 floats Orientation (A,B,C) 0x20 3 floats Upper Extents 0x2C 3 floats Lower Extents 0x38 uint16 Flags 0x3A uint16 Doodad set index 0x3C uint16 Name set? 0x3E uint16 Padding
```

```
struct SMMapObjDef { uint nameId; uint uniqueId; C3Vectori pos; C3Vectori rot; CAaBoxi extents; uint16 flags; uint16 doodadSet; uint16 nameSet; uint16 pad; };
```

* How to compute a matrix to map WMO to world coordinates

Refer to MODF(ADT)

## MANM

This section only applies to version (8.3.0.32272).

Temporarily during 8.3.0 PTR, root WDTs contained an MANM chunk, which appeared to be data used for navigation or scripts, often marking roads or walls. It was not parsed by the client and was removed a patch later again. The shipped files had lots of constants, so little was actually known other than the positions and an ID that nobody was able to map to something useful, which seemed to be globally (across all WDTs) unique though.

```
/*0x00 */ uint32_t version; // 1 /*0x04 */ uint32_t count; struct { /*+0x000*/ uint32_t id; // globally unique /*+0x004*/ uint8_t unk_0x004[0x90]; // 0 vv probably matrices, likely all zeros are also floats /*+0x094*/ float unk_0x094[2]; // 1.0 /*+0x09C*/ uint8_t unk_0x09C[0x98]; // 0 /*+0x134*/ float unk_0x134[2]; // 1.0 /*+0x13C*/ uint8_t unk_0x13C[0x98]; // 0 /*+0x1D4*/ float unk_0x1D4[2]; // 1.0 /*+0x1DC*/ uint8_t unk_0x1DC[0x10]; // 0 ^^ /*+0x1EC*/ float unk_0x1EC; // 1.0 /*+0x1F0*/ uint32_t unk_0x1F0; // 6 /*+0x1F4*/ float unk_0x1F4; // 1.0 /*+0x1F8*/ uint8_t unk_0x1F8[0x8]; // 0 /*+0x200*/ float unk_0x200; // 1.0 /*+0x204*/ uint32_t unk_0x204; // 6 /*+0x208*/ float unk_0x208; // 1.0 /*+0x20C*/ uint8_t unk_0x20C[0x8]; // 0 /*+0x214*/ float unk_0x214; // 1.0 /*+0x218*/ uint32_t unk_0x218; // 6 /*+0x21C*/ uint32_t unk_0x21C; // 1 /*+0x220*/ uint32_t typeish; // 0, 2, 3, 4, 6. 2 or 6: absolute positions (usually paths or walls), 3: relative positions (to what?) /*+0x224*/ int count2; struct { /*+0x00 */ C3Vectori position; /*+0x0C */ C3Vectori normal; // always pointing up /*+0x228*/ } points[count2]; /*0x08 */ } entries[count];
```

This section only applies to versions ≥ .

In Shadowlands the feature was officially added as, revealing the name MapAnima, with the client now parsing it. Four versions have been done during development. As of

* (9.0.1.33978) ... 9.0.1.34199, the version is 6,
* (9.0.1.34278) ... 9.0.1.34821, the version is 7, and starting
* (9.0.1.34972), the version is 8.

```
// offset comments are assuming version 8 struct Anima { struct Data { struct TIA { /*0x000*/ float unk_000; // AnimaMaterial_Field_9_0_1_33978_016 * AnimaMaterial_Field_9_0_1_33978_020 /*0x004*/ float unk_004; // AnimaMaterial_Field_9_0_1_33978_017 * AnimaMaterial_Field_9_0_1_33978_021 /*0x008*/ float unk_008; // AnimaMaterial_Field_9_0_1_33978_018 * AnimaMaterial_Field_9_0_1_33978_022 /*0x00C*/ float AnimaMaterial_Field_9_0_1_33978_023; // For sake of laziness these are fdids at file time and // Texture* at run time, so padded to fit a x64 pointer. struct TextureFdidAndPadding { /*0x000*/ uint32_t fdid; /*0x004*/ char padding[4]; /*0x008*/ }; /*0x010*/ TextureFdidAndPadding AnimaMaterial_EffectTexture[3]; /*0x028*/ TextureFdidAndPadding AnimaMaterial_RibbonTexture; /*0x030*/ float unk_per_texture[4]; // apparently only set at runtime, to 1.0f /*0x040*/ float AnimaMaterial_Field_9_0_1_33978_000; /*0x044*/ float AnimaMaterial_Field_9_0_1_33978_001; /*0x048*/ float AnimaMaterial_Field_9_0_1_33978_002; /*0x04C*/ float AnimaMaterial_Field_9_0_1_33978_003; /*0x050*/ float AnimaMaterial_Field_9_0_1_33978_004; /*0x054*/ float AnimaMaterial_Field_9_0_1_33978_005; /*0x058*/ float AnimaMaterial_Field_9_0_1_33978_006; /*0x05C*/ float AnimaMaterial_Field_9_0_1_33978_007; /*0x060*/ float AnimaMaterial_Field_9_0_1_33978_008; /*0x064*/ float AnimaMaterial_Field_9_0_1_33978_009; /*0x068*/ float AnimaMaterial_Field_9_0_1_33978_010; /*0x06C*/ float AnimaMaterial_Field_9_0_1_33978_011; /*0x070*/ float AnimaMaterial_Field_9_0_1_33978_012; /*0x074*/ float AnimaMaterial_Field_9_0_1_33978_013; /*0x078*/ float AnimaMaterial_Field_9_0_1_33978_014; /*0x07C*/ float AnimaMaterial_Field_9_0_1_33978_015; /*0x080*/ C2Vectori unk_080[4]; // apparently also always 0, and initialized depending on flags below /*0x0A0*/ uint32_t unk_0A0; /*0x0A4*/ float AnimaMaterial_Field_9_0_1_33978_025; /*0x0A8*/ float AnimaMaterial_Field_9_0_1_33978_026; /*0x0AC*/ uint32_t AnimaMaterial_Flags; // &4: randomize float_pairs /*0x0B0*/ uint32_t AnimaMaterial_ID; /*0x0B4*/ uint32_t AnimaMaterial_Field_9_0_1_33978_028; /*0x0B8*/ float AnimaMaterial_Field_9_0_1_33978_029; /*0x0BC*/ uint32_t unk_0BC; /*0x0C0*/ }; struct TIB { /*0x000*/ uint32_t a; /*0x004*/ uint32_t b; /*0x008*/ float c; /*0x00C*/ uint32_t slice_count; /*0x010*/ float d; /*0x014*/ uint32_t e; /*0x018*/ }; /*0x000*/ TIA tube_info_a[6]; /*0x480*/ TIB tube_info_b[6]; /*0x510*/ uint32_t tube_presence_mask; // if tube_info_x[i] has information, set bit 1 << i /*0x514*/ uint32_t AnimaCable_Field_9_0_1_33978_010; // &1: use 6 instead of 1 positions per node /*0x518*/ uint32_t AnimaCable_ParticleModel; // model to use particles from /*0x51C*/ float AnimaCable_Field_9_0_1_33978_001; /*0x520*/ float AnimaCable_Field_9_0_1_33978_002; /*0x524*/ uint32_t AnimaCable_Field_9_0_1_33978_003; /*0x528*/ float AnimaCable_Field_9_0_1_33978_004; /*0x52C*/ float AnimaCable_Field_9_0_1_33978_005; /*0x530*/ uint32_t AnimaCable_Field_9_0_1_33978_007; /*0x534*/ uint32_t AnimaCable_Field_9_0_1_33978_008; /*0x538*/ uint32_t AnimaCable_Field_9_0_1_33978_009; /*0x53C*/ float AnimaCable_Field_9_0_1_33978_027; /*0x540*/ float AnimaCable_Field_9_0_1_33978_028; /*0x544*/ float AnimaCable_Field_9_0_1_33978_029; /*0x548*/ float AnimaCable_Field_9_0_1_33978_030; /*0x54C*/ float AnimaCable_Field_9_0_1_33978_031; /*0x550*/ float AnimaCable_Field_9_0_1_33978_032; // capsules are interleaved in dbc, so mapping is a bit shit. // count is hardcoded to 2, which allows for that horror. union Capsules { struct { struct { /*0x000*/ float unk_000; /*0x004*/ float unk_004; /*0x008*/ float unk_008; /*0x00C*/ float unk_00C; /*0x010*/ float unk_010; /*0x014*/ float unk_014; /*0x018*/ float unk_018; } capsules[2]; } logic_grouping; struct { // capsules[0] float AnimaCable_Field_9_0_1_33978_013; float AnimaCable_Field_9_0_1_33978_011; float AnimaCable_Field_9_0_1_33978_015; float AnimaCable_Field_9_0_1_33978_017; float AnimaCable_Field_9_0_1_33978_019; float AnimaCable_Field_9_0_1_33978_020; float AnimaCable_Field_9_0_1_33978_021; // capsules[1] float AnimaCable_Field_9_0_1_33978_014; float AnimaCable_Field_9_0_1_33978_012; float AnimaCable_Field_9_0_1_33978_016; float AnimaCable_Field_9_0_1_33978_018; float AnimaCable_Field_9_0_1_33978_022; float AnimaCable_Field_9_0_1_33978_023; float AnimaCable_Field_9_0_1_33978_024; } flat_like_in_db; }; /*0x554*/ Capsules capsule_info; /*0x58C*/ uint32_t AnimaCable_Field_9_0_1_33978_006; // sound kit /*0x590*/ uint32_t AnimaCable_ID; /*0x594*/ float AnimaCable_Field_9_0_1_33978_025; /*0x598*/ float AnimaCable_Field_9_0_1_33978_026; #if version >= 8 /*0x59C*/ float AnimaCable_Field_9_0_1_34972_034; #endif #if version >= 7 /*0x5A0*/ uint32_t AnimaCable_Field_9_0_1_34199_033; /*0x5A4*/ uint32_t unk_5A4; /*0x5A8*/ uint32_t unk_5A8; #endif /*0x5AC*/ CAaBoxi bounding_box; #if version >= 8 /*0x5C4*/ uint32_t unk_5C4; #endif /*0x5C8*/ }; struct Node { /*0x000*/ C3Vectori position; /*0x00C*/ C3Vectori normal; /*0x018*/ }; struct Segment { /*0x000*/ C3Vectori unk_000; /*0x00C*/ C3Vectori unk_00C; /*0x018*/ uint32_t unk_018; /*0x01C*/ }; /*0x000*/ uint32_t uid; // globally unique, is a CMapObj /*0x004*/ Data data; /*0x5CC*/ uint32_t numNodes; // >= 6 /*0x5D0*/ Node nodes[numNodes]; #if version >= 7 /* */ Segment segments[numNodes - 3]; #endif /* */ }; /*0x000*/ uint32_t version; // 8 /*0x004*/ uint32_t count; /*0x008*/ Anima animas[count]; /* */
```

# _occ, _lgt

This section only applies to versions ≥ .

WoD added _occ.wdt (occlusion) and _lgt.wdt (lights) for each .wdt. They are only used for adt-maps, not WMO-only ones.

## occ

MAOI and MAOH might be zero size. (WMO-only WDTs)

### MVER

```
struct { uint32_t version; // 18, just as all others } mver;
```

### MAOI

```
struct { uint16_t tile_x; // MAOH entries are per ADT tile uint16_t tile_y; uint32_t offset; // in MAOH uint32_t size; // always (17*17+16*16)*2 } maoi[];
```

### MAOH

Defines a heightmap that occludes everything behind it. Same content as WDL#MARE_chunks.

```
short interleaved_map[17*17+16*16];
```

## lgt

Might only have MVER for WMO-only WDTs. Level designers are able to freely place lights, without placing models containing lights now. This is used below lamp posts and alike. As of Legion, there is support for point and spot lights.

### MVER

```
struct { uint32_t version; // ≤ (7.0.1.20740): 18 just as all others, ≥ (7.0.1.20914): 20 } mver;
```

### MPLT

This section only applies to versions ≤ .

```
struct { uint32 id; uint16 tile_x; uint16 tile_y; CArgbi color; C3Vectori position; float unknown[3]; // intensity, and stuff. flicker? } map_point_lights[];
```

* starting some Legion build, these are no longer read for backwards compatibility but MPL2 is required.

### MPL2 (Legion+)

This section only applies to versions ≥ .

* appears to be either, not both MPLT and MPL2, or they need to have the same size.

struct {

```
/*0x00*/ uint32 id; // Seems to work fine been unique per WDT, increment for each new. /*0x04*/ CImVectori color; // (B, G, R, A) - Note Alpha is 0 for full opaqueness /*0x08*/ C3Vectori position; // X Y Z of the light, this can be obtained using .gps /*0x14*/ float attenuationStart; // Start position of the lights spread, most start at 0 /*0x18*/ float attenuationEnd; // How far the light spreads /*0x1C*/ float intensity; // Lights intensity /*0x20*/ C3Vector rotation // X Y Z - Most values are 0 /*0x2C*/ uint16 tile_x; // ADT Tile Co-ordinate X /*0x2E*/ uint16 tile_y; // ADT Tile Co-ordinate Y /*0x30*/ mlta_index // If you are not using MLTA then this should be '-1' otherwise use MLTA index (zero based) /*0x32*/ textureIndex // Most files have '-1' for this, legion doesn't seem to use fileIDs for everything /*0x34*/
```

} map_point_lights[];

* the only file I know having this (e3148cc88c7f2fcaebe99c53e5e5079e) has a size of 0x40 for MPL2, which does not match to what the client parses (0x34) --Schlumpf (talk) 03:18, 22 November 2015 (UTC)
* unknown_2, as an int16 array, has seen values of [0,-1] and [-1,-1] --Barncastle (OUTDATED - See mlta_index)
* Drikish 24/11/24 - Completed based on research into this chunk finding the unknowns and adding reference into to each part, with thanks to Marlamin for help.

### MPL3 (Shadowlands+)

This section only applies to versions ≥ (9.0.1.34490).

```
struct { /*0x00*/ uint32_t lightIndex; /*0x04*/ CImVectori color; /*0x08*/ C3Vector position; /*0x14*/ float attenuationStart; /*0x18*/ float attenuationEnd; /*0x1C*/ float intensity; /*0x20*/ C3Vector rotation; // Rotation, only used for lightcookie manipulation for point lights /*0x2C*/ uint16_t tileX; /*0x2E*/ uint16_t tileY; /*0x30*/ int16_t mlta_index; // Index into MLTA /*0x32*/ int16_t textureIndex; // Index into MTEX for lightcookie texture /*0x34*/ uint16_t flags; // 1: cast (raytraced?) shadows (on D3D12 + min shadowrt level 2) /*0x36*/ float16 scale; // scale in half-float format, Known values: 0.5 (appears to be default), 10.0 (on 3 lights on map 2222) 10.0 hides the player shadows /*0x38*/ }; map_point_lights_version3[]
```

### MSLT (Legion+)

This section only applies to versions ≥ .

```
struct { /* 0x00 */ uint32_t id; /* 0x00 */ CImVectori color; /* 0x04 */ C3Vector position; /* 0x08 */ float attenuationStart; // When to start the attenuation of the light, must be <= attenuationEnd or glitches /* 0x14 */ float attenuationEnd; /* 0x18 */ float intensity; /* 0x1C */ C3Vector rotation; // radians /* 0x20 */ float spotlightRadius; /* 0x2C */ float innerAngle; /* 0x30 */ float outerAngle; // radians /* 0x34 */ uint16_t tileX; /* 0x38 */ uint16_t tileY; /* 0x3A */ int16_t mlta_index; //Index into MLTA /* 0x3C */ int16_t textureIndex; //Index into MTEX } map_spot_lights[];
```

### MTEX (Legion+)

This section only applies to versions ≥ .

```
uint32_t textureFileDataIds[];
```

### MLTA

This section only applies to versions ≥ .

```
struct { float amplitude; float frequency; int function; // 0 = off, 1 = sine curve, 2 = noise curve, 3 = noise step curve } map_lta[];
```

Map Light Texture(?) Animation

# _fogs

This section only applies to versions ≥ (7.2.5.24076).

Legion added _fogs.wdt for a subset of .wdts. They seem to be only present for terrain maps, not WMO maps. As of (7.2.5.24076) (when they were added) and (7.3.2.25383) (when this paragraph was added), they are all empty and not even read by the client. It is likely that they are merged into the branch by accident and are a feature. The first files with content are zandalar and kultiras with (8.0.1.25902).

## MVER

```
uint32_t version; // ≥ (7.2.5.24076): 1, ≥ (11.0.0.54935): 2
```

## VFOG

This section only applies to versions ≥ (8.0.1.25902).

```
struct { /*0x00*/ C3Vector color; // r g b fog color. 1 equals 255 /*0x0C*/ float intensity[3]; // fog radius related intensity: min: 0, 0.35, 0.4 | max: 1, 6, 3300 /*0x18*/ float _unk18; // min: 0, max: 5000 /*0x1C*/ C3Vector position; // server position /*0x28*/ float _unk28; // set to 1.0 on loading, 0 in files /*0x2C*/ C4Vector rotation; // quat /*0x3C*/ float radius[3]; // fog start radiusu, min: 200, 50, 0.3 | max: 10000, 5000, 22 /*0x48*/ int animationPeriods[4]; // Used to calculate different coefficients /*0x58*/ uint32_t flags; /*0x5C*/ uint32_t modelFileDataId; // the client only supports models with one M2Batch, if 0: 166046 (spells/errorcube.m2) /*0x60*/ uint32_t fogLevel; // min 0, max 2. /*0x64*/ uint32_t id; // globally unique in the files /*0x68*/ } volumetric_fogs[];
```

fogLevel allows client to limit fogs used and it's inclusive. For example, if client-side variable is set to value 2, records with fogLevel set to 2 or lower are processed. Might be related to cvar volumeFogLevel

## VFEX

This section only applies to versions ≥ (11.0.0.54935).

Appears only in version 2 of the format. Optional extra data for VFOG entries (to keep backwards compatibility by not modifying the VFOG structure), thus at most one per VFOG entry. Always 0x60/96 bytes. Mapped to VFOG entry via VFOG_ID.

```
struct { /*0x00*/ uint32_t Unk0; // Default 1 /*0x04*/ float Unk1[16]; // First 3 floats always seem to have proper values. Rest are 1? /*0x44*/ uint32_t VFOG_ID; // Ref to ID in VFOG entry. /*0x48*/ uint32_t Unk3; // Default 0 /*0x4C*/ uint32_t Unk4; // Default 0 /*0x50*/ uint32_t Unk5; // Default 0 /*0x54*/ uint32_t Unk6; // Default 0 /*0x58*/ uint32_t Unk7; // Default 0 /*0x5C*/ uint32_t Unk8; // Default 0 /*0x60*/ } volumetric_fog_ex;
```

# _mpv

This section only applies to versions ≥ (8.0.1.26287).

As of ≥ (8.0.1.26287) references to _mpv.wdt (particulate volume) have been seen in the client. While these files haven't been shipped (26310), the CMap::Load function does attempt to read them when present. These files were first shipped in (8.0.1.26433).

While the file is chunked, it does require the exact order of #PVPD, #PVMI, #PVBD: #PVMI might override #PVPD, and as soon as #PVBD is read, it is finalised.

## MVER

```
enum mpv_version : uint32_t { mpv_version_0, // ignores the rest of the file (actually, all < 1, so probably just ≥1 as requirement)u mpv_version_1, // (8.0.1.26433) mpv_version_2, // (8.0.1.26476) ... (8.0.1.26557) mpv_version_3, // (8.0.1.26567) ... (8.0.1.27404) mpv_version_4, // ≥ (8.0.1.27481) }; mpv_version version;
```

## PVPD

```
struct { C2Vectori _unk00; // [-1.f, 1.f]u float _unk08; // only seen: -0.fu float _unk0c; } particle_volume_pd[];
```

## PVMI

If #PVPD was already read, it is nulled out. Note that the inverse is not true, i.e. if #PVMI comes first, #PVPD may be non-null. This is not a bug but actual files have #PVMI first, followed by #PVPD and #PVBD.u

```
struct { #if version == mpv_version_1 char _unk00[0xF5C]; // appears to be a huge blob, 0xF5C bytes, including five (binary) WWFParticulateGroupsu // this might not actually be pure binary WWFParticulateGroups (or that size changed without MVER change), since the block is 0xF84 in (8.0.1.26433)u #else if version == mpv_version_2 char _unk00[0xFE8]; #else if version >= mpv_version_3 char _unk00[0x10D8]; #endif } particle_volume_mi[];
```

## PVBD

```
struct { uint32_t num_unk1C; CAaBoxi _unk04; // bounds/extentsu uint32_t _unk1C[8]; // indices into #PVPDu uint32_t _unk3C; // boolean: This entry is complete. If false, it is joined with the next entry. It will have the same bounds.u } particle_volume_bd[];