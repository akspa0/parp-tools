# WMO

From wowdev

Jump to navigation Jump to search

## Contents

* 1 MVER
* 2 WMO root file

  + 2.1 MOMO
  + 2.2 MOHD chunk
  + 2.3 MOTX chunk
  + 2.4 MOMT chunk

    - 2.4.1 Texture addressing
    - 2.4.2 Emissive color
    - 2.4.3 Shader types (12340)
    - 2.4.4 Shader types (15464)
    - 2.4.5 Shader types (18179)
    - 2.4.6 Shader types (26522)
    - 2.4.7 CMapObj::CreateMaterial
  + 2.5 MOM3
  + 2.6 MOUV
  + 2.7 MOGN chunk
  + 2.8 MOGI chunk
  + 2.9 MOSB chunk (optional)u
  + 2.10 MOSI (optional)
  + 2.11 MOPV chunk
  + 2.12 MOPT chunk
  + 2.13 MOPR chunk
  + 2.14 MOPE chunk
  + 2.15 MOVV chunk
  + 2.16 MOVB chunk
  + 2.17 MOLT chunk
  + 2.18 MOLV
  + 2.19 MODS chunk
  + 2.20 MODN chunk
  + 2.21 MODI chunk
  + 2.22 MODD chunk
  + 2.23 MFOG chunk
  + 2.24 MCVP chunk (optional)
  + 2.25 GFID
  + 2.26 MDDI
  + 2.27 MPVD
  + 2.28 MAVG
  + 2.29 MAVD
  + 2.30 MBVD
  + 2.31 MFED
  + 2.32 MGI2
  + 2.33 MNLD
  + 2.34 MDDL
* 3 WMO group file

  + 3.1 MOGP chunk

    - 3.1.1 group flags
    - 3.1.2 group flags 2
    - 3.1.3 "antiportal"
    - 3.1.4 Split Groups
  + 3.2 MOGX chunk
  + 3.3 MOPY chunk
  + 3.4 MPY2 chunk
  + 3.5 MOVI chunk
  + 3.6 MOVX chunk
  + 3.7 MOVT chunk
  + 3.8 MONR chunk
  + 3.9 MOTV chunk
  + 3.10 MOLV
  + 3.11 MOIN
  + 3.12 MOBA chunk

    - 3.12.1 unknown\_box
  + 3.13 MOQG chunk
  + 3.14 MOLR chunk
  + 3.15 MODR chunk
  + 3.16 MOBN chunk
  + 3.17 MOBR chunk
  + 3.18 MOCV chunk

    - 3.18.1 CMapObjGroup::FixColorVertexAlpha

      * 3.18.1.1 WMOs with MOHD->flags & 0x08
      * 3.18.1.2 All other WMOs
      * 3.18.1.3 Decompiled code
    - 3.18.2 CMapObj::AttenTransVerts

      * 3.18.2.1 Decompiled code
  + 3.19 MOC2 chunk
  + 3.20 MLIQ chunk

    - 3.20.1 how to determine LiquidTypeRec to use
  + 3.21 MORI
  + 3.22 MORB
  + 3.23 MOTA
  + 3.24 MOBS
  + 3.25 MDAL
  + 3.26 MOPL
  + 3.27 MOPB
  + 3.28 MOLS
  + 3.29 MOLP
  + 3.30 MLSS
  + 3.31 MLSP
  + 3.32 MLSO
  + 3.33 MLSK
  + 3.34 MOS2
  + 3.35 MOP2
  + 3.36 MPVR
  + 3.37 MAVR
  + 3.38 MBVR
  + 3.39 MFVR
  + 3.40 MNLR
  + 3.41 MOLM
  + 3.42 MOLD
  + 3.43 MPB\*

    - 3.43.1 MPBV
    - 3.43.2 MPBP
    - 3.43.3 MPBI
    - 3.43.4 MPBG

WMO files contain world map objects. They, too, have a chunked structure just like the WDT files.

There are two types of WMO files, actually:

* WMO root file - lists textures (BLP Files), doodads (M2 or MDX Files), etc., and orientation for the WMO groups
* WMO group file - 3d model data for one unit in the world map object

The root file and the groups are stored with the following filenames:

* World\wmo\path\WMOName.wmo
* World\wmo\path\WMOName\_NNN.wmo

There is a hardcoded maximum of 512 group files per root object.

This section only applies to versions ≤ (0.5.5.3494).

In the alpha, WMO files were a single file rather than being split into root and group. For that reason the root data has been wrapped in a MOMO chunk followed by the MOGP chunks.

# MVER

```
uint32_t version; // < (0.6.0.3592) 14, (0.6.0.3592) ... < 16, ≥ 17
```

There never have been any additional versions after the alpha, even though the format changed a lot. Classic Blizzard.

# WMO root file

The root file lists the following:

* textures (BLP File references)
* materials
* models (MDX / M2 File references)
* groups
* visibility information
* more data

This section only applies to versions ≤ (0.5.5.3494).

In version 14, the version used in the alpha, the root WMO file has an additional container MOMO chunk, like the MOGP chunk, containing all group data.

## MOMO

This section only applies to versions ≤ (0.5.5.3494). Only used in v14..

Rather than all chunks being top level, they have been wrapped in MOMO. There has been no other additional data, rather than just everything being wrapped.

## MOHD chunk

* Header for the map object. 64 bytes.

```
struct SMOHeader { /*000h*/ uint32_t nTextures; /*004h*/ uint32_t nGroups; /*008h*/ uint32_t nPortals; /*00Ch*/ uint32_t nLights; // Blizzard seems to add one to the MOLT entry count when there are MOLP chunks in the groups (and maybe for MOLS too?)u /*010h*/ uint32_t nDoodadNames; /*014h*/ uint32_t nDoodadDefs; // * /*018h*/ uint32_t nDoodadSets; /*01Ch*/ CArgbi ambColor; // Color settings for base (ambient) color. See the flag at /*03Ch*/. /*020h*/ foreign_keyi<uint32_t, &WMOAreaTableRec::m_WMOID> wmoID; #if ≤ (0.5.5.3494) /*0x24*/ uint8_t padding[0x1c]; #else /*024h*/ CAaBoxi bounding_box; // in the alpha, this bounding box was computed upon loading /*03Ch*/ uint16_t flag_do_not_attenuate_vertices_based_on_distance_to_portal : 1; /*03Ch*/ uint16_t flag_use_unified_render_path : 1; // In 3.3.5a this flag switches between classic render path (MOHD color is baked into MOCV values, all three batch types have their own rendering logic) and unified (MOHD color is added to lighting at runtime, int. and ext. batches share the same rendering logic). See [[1]] for more details. /*03Ch*/ uint16_t flag_use_liquid_type_dbc_id : 1; // use real liquid type ID from DBCs instead of local one. See MLIQ for further reference. /*03Ch*/ uint16_t flag_do_not_fix_vertex_color_alpha: 1; // In 3.3.5.a (and probably before) it prevents CMapObjGroup::FixColorVertexAlpha function to be executed. Alternatively, for the wotlk version of it, the function can be called with MOCV.a being set to 64, whjch will produce the same effect for easier implementation. For wotlk+ rendering, it alters the behavior of the said function instead. See [[2]] for more details. /*03Ch*/ uint16_t flag_lod : 1; // ≥ (20740) /*03Ch*/ uint16_t flag_default_max_lod : 1; // ≥ (21796)u. Usually maxLodLevel = -1 but with this flag, numLod. Entries at this level are defaulted /*03Ch*/ uint16_t : 10; // unused as of (20994) /*03Eh*/ uint16_t numLod; // ≥ (21108) includes base lod (→ numLod = 3 means '.wmo', 'lod0.wmo' and 'lod1.wmo') #endif } header;
```

## MOTX chunk

This section only applies to versions < (8.1.0.28186). MOTX has been replaced with file data ids in MOMT.

* Blob of textures filenames (BLP Files) used in this map object.

A block of zero-terminated strings which are complete filenames with paths. They are referenced by offset in MOMT. Note that there are also empty strings, so there are "random additional zero bytes" which are just another zero-terminated empty string. \_Do not try to parse this chunk out of thin air\_. It makes no sense without the information from MOMT.

```
char textureNameList[];
```

This section only applies to versions ≥ (8.1.0.28186).

Starting with 8.1, MOTX is no longer used. The texture references in MOMT are file data ids directly. As of that version, there is a fallback mode though and some files still use MOTX for sake of avoiding re-export. To check if texture references in MOMT are file data ids, simply check if MOTX exists in the file.

## MOMT chunk

* Materials used in this map object, 64 bytes per texture (BLP file).

```
struct SMOMaterial { #if ≤ (0.5.5.3494) uint32_t version; #endif /*0x00*/ uint32_t F_UNLIT : 1; // disable lighting logic in shader (but can still use vertex colors) /*0x00*/ uint32_t F_UNFOGGED : 1; // disable fog shading (rarely used) /*0x00*/ uint32_t F_UNCULLED : 1; // two-sided /*0x00*/ uint32_t F_EXTLIGHT : 1; // darkened, the intern face of windows are flagged 0x08 /*0x00*/ uint32_t F_SIDN : 1; // (bright at night, unshaded) (used on windows and lamps in Stormwind, for example) (see emissive color) /*0x00*/ uint32_t F_WINDOW : 1; // lighting related (flag checked in CMapObj::UpdateSceneMaterials) /*0x00*/ uint32_t F_CLAMP_S : 1; // tex clamp S (force this material's textures to use clamp s addressing) /*0x00*/ uint32_t F_CLAMP_T : 1; // tex clamp T (force this material's textures to use clamp t addressing) /*0x00*/ uint32_t flag_0x100 : 1; /*0x00*/ uint32_t : 23; // unused as of 7.0.1.20994 #if ≥ (0.6.0.3592) /*0x04*/ uint32_t shader; // Index into CMapObj::s_wmoShaderMetaData. See below (shader types). #endif /*0x08*/ uint32_t blendMode; // Blending: see EGxBlend /*0x0C*/ uint32_t texture_1; // offset into MOTX; ≥ (8.1.0.27826) No longer references MOTX but is a filedata id directly. /*0x10*/ CImVectori sidnColor; // emissive color; see below (emissive color) /*0x14*/ CImVectori frameSidnColor; // sidn emissive color; set at runtime; gets sidn-manipulated emissive color; see below (emissive color) /*0x18*/ uint32_t texture_2; // offset into MOTX or texture file id /*0x1C*/ CImVectori diffColor; /*0x20*/ foreign_keyi<uint32_t, &TerrainTypeRec::m_ID> ground_type; // according to CMapObjDef::GetGroundType #if ≤ (0.6.0.3592) char inMemPad[8]; #else /*0x24*/ uint32_t texture_3; // offset into MOTX or texture file id /*0x28*/ uint32_t color_2; // For shader 23, this can be a texture file id. /*0x2C*/ uint32_t flags_2; // For shader 23, this can be a texture file id. /*0x30*/ uint32_t runTimeData[4]; // This data is explicitly nulled upon loading. For shader 23, this can contain textures file ids. /*0x40*/ #endif } materialList[];
```

texture\_1, 2 and 3 are start positions for texture filenames in the MOTX data block ; texture\_1 for the first texture, texture\_2 for the second (see shaders), etc. texture\_1 defaults to "createcrappygreentexture.blp".

If a texture isn't used the its start position seems to point to a chains of 4 \0u

color\_2 is diffuse color : CWorldView::GatherMapObjDefGroupLiquids(): geomFactory->SetDiffureColor((CImVectori\*)(smo+7));

The flags might used to tweak alpha testing values, I'm not sure about it, but some grates and flags in IF seem to require an alpha testing threshold of 0, at other places this is greater than 0.

### Texture addressing

By default, textures used by WMO materials are assigned an addressing mode of EGxTexWrapMode::GL\_REPEAT (ie wrap mode).

SMOMaterial flags F\_CLAMP\_S and F\_CLAMP\_T can override this default to clamp mode for the S and T dimensions, respectively.

### Emissive color

The sidnColor CImVectori at offset 0x10 is used with the SIDN (self-illuminated day night) scalar from CDayNightObject to light exterior window glows (see flag 0x10 above).

The scalar is interpolated out of a static table in the client, based on the time of day.

The color value eventually is copied into offset 0x14 (frameSidnColor) after being manipulated by the SIDN scalar. This manipulation occurs in CMapObj::UpdateMaterials.

### Shader types (12340)

Wrath of the Lich King only uses shaders 0 to 6. See below for more info on those.

### Shader types (15464)

Depending on the shader, a different amount of textures is required. If there aren't enough filenames given, it defaults to Opaque (with one filename). More filenames than required are just ignored.

Data is from 15464.

| value | name | textures without shader | textures with shader | texcoord count | color count |
| --- | --- | --- | --- | --- | --- |
| 0 | Diffuse | 1 | 1 | 1 | 1 |
| 1 | Specular | 1 | 1 | 1 | 1 |
| 2 | Metal | 1 | 1 | 1 | 1 |
| 3 | Env | 1 | 2 | 1 | 1 |
| 4 | Opaque | 1 | 1 | 1 | 1 |
| 5 | EnvMetal | 1 | 2 | 1 | 1 |
| 6 | TwoLayerDiffuse | 1 | 2 | 2 | 2 |
| 7 | TwoLayerEnvMetal | 1 | 3 | 2 | 2 |
| 8 | TwoLayerTerrain | 1 | 2 | 1 | 2 | automatically adds \_s in the filename of the second texture |
| 9 | DiffuseEmissive | 1 | 2 | 2 | 2 |
| 10 | 1 | 1 | 1 | 1 | SMOMaterial::SH\_WATERWINDOW -- Seems to be invalid. Does something with MOTA (tangents). |
| 11 | MaskedEnvMetal | 1 | 3 | 2 | 2 |
| 12 | EnvMetalEmissive | 1 | 3 | 2 | 2 |
| 13 | TwoLayerDiffuseOpaque | 1 | 2 | 2 | 2 |
| 14 | TwoLayerDiffuseEmissive | 1 | 1 | 1 | 1 | SMOMaterial::SH\_SUBMARINEWINDOW -- Seems to be invalid. Does something with MOTA (tangents). |
| 15 | 1 | 2 | 2 | 2 |
| 16 | Diffuse | 1 | 1 | 1 | 1 | SMOMaterial::SH\_DIFFUSE\_TERRAIN -- "Blend Material": used for blending WMO with terrain (dynamic blend batches) |

tex coord and color count decide vertex buffer format: EGxVertexBufferFormat\_PNC2T2

### Shader types (18179)

| value | #textures without shader | #textures with shader | texcoord count | color count |
| --- | --- | --- | --- | --- |
| 0 - Diffuse | 1 | 1 | 1 | 1 |
| 1 - Specular | 1 | 1 | 1 | 1 |
| 2 - Metal | 1 | 1 | 1 | 1 |
| 3 - Env | 1 | 2 | 1 | 1 |
| 4 - Opaque | 1 | 1 | 1 | 1 |
| 5 - EnvMetal | 1 | 2 | 1 | 1 |
| 6 - TwoLayerDiffuse | 1 | 2 | 2 | 2 |
| 7 - TwoLayerEnvMetal | 1 | 3 | 2 | 2 |
| 8 - TwoLayerTerrain | 1 | 2 | 1 | 2 | automatically adds \_s in the filename of the second texture |
| 9 - DiffuseEmissive | 1 | 2 | 2 | 2 |
| 10 - waterWindow | 1 | 1 | 1 | 1 | SMOMaterial::SH\_WATERWINDOW -- automatically generates MOTA |
| 11 - MaskedEnvMetal | 1 | 3 | 2 | 2 |
| 12 - EnvMetalEmissive | 1 | 3 | 2 | 2 |
| 13 - TwoLayerDiffuseOpaque | 1 | 2 | 2 | 2 |
| 14 - submarineWindow | 1 | 1 | 1 | 1 | SMOMaterial::SH\_SUBMARINEWINDOW -- automatically generates MOTA |
| 15 - TwoLayerDiffuseEmissive | 1 | 2 | 2 | 2 |
| 16 - DiffuseTerrain | 1 | 1 | 1 | 1 | SMOMaterial::SH\_DIFFuse\_Terrain -- "Blend Material": used for blending WMO with terrain (dynamic blend batches) |
| 17 - AdditiveMaskedEnvMetal | 1 | 3 | 2 | 2 |

### Shader types (26522)

| value | vertex shader | pixel shader |
| --- | --- | --- |
| 0 - Diffuse | MapObjDiffuse\_T1 | MapObjDiffuse |
| 1 - Specular | MapObjSpecular\_T1 | MapObjSpecular |
| 2 - Metal | MapObjSpecular\_T1 | MapObjMetal |
| 3 - Env | MapObjDiffuse\_T1\_Refl | MapObjEnv |
| 4 - Opaque | MapObjDiffuse\_T1 | MapObjOpaque |
| 5 - EnvMetal | MapObjDiffuse\_T1\_Refl | MapObjEnvMetal |
| 6 - TwoLayerDiffuse | MapObjDiffuse\_Comp | MapObjTwoLayerDiffuse |
| 7 - TwoLayerEnvMetal | MapObjDiffuse\_T1 | MapObjTwoLayerEnvMetal |
| 8 - TwoLayerTerrain | MapObjDiffuse\_Comp\_Terrain | MapObjTwoLayerTerrain | automatically adds \_s in the filename of the second texture |
| 9 - DiffuseEmissive | MapObjDiffuse\_Comp | MapObjDiffuseEmissive |
| 10 - waterWindow | FFXWaterWindow | FFXWaterWindow | It's FFX instead of normal material. SMOMaterial::SH\_WATERWINDOW -- automatically generates MOTA |
| 11 - MaskedEnvMetal | MapObjDiffuse\_T1\_Env\_T2 | MapObjMaskedEnvMetal |
| 12 - EnvMetalEmissive | MapObjDiffuse\_T1\_Env\_T2 | MapObjEnvMetalEmissive |
| 13 - TwoLayerDiffuseOpaque | MapObjDiffuse\_Comp | MapObjTwoLayerDiffuseOpaque |
| 14 - submarineWindow | FFXSubmarineWindow | FFXSubmarineWindow | It's FFX instead of normal material. SMOMaterial::SH\_SUBMARINEWINDOW -- automatically generates MOTA |
| 15 - TwoLayerDiffuseEmissive | MapObjDiffuse\_Comp | MapObjTwoLayerDiffuseEmissive |
| 16 - DiffuseTerrain | MapObjDiffuse\_T1 | MapObjDiffuse | SMOMaterial::SH\_DIFFuse\_Terrain -- "Blend Material": used for blending WMO with terrain (dynamic blend batches) |
| 17 - AdditiveMaskedEnvMetal | MapObjDiffuse\_T1\_Env\_T2 | MapObjAdditiveMaskedEnvMetal |
| 18 - TwoLayerDiffuseMod2x | MapObjDiffuse\_CompAlpha | MapObjTwoLayerDiffuseMod2x |
| 19 - TwoLayerDiffuseMod2xNA | MapObjDiffuse\_Comp | MapObjTwoLayerDiffuseMod2xNA |
| 20 - TwoLayerDiffuseAlpha | MapObjDiffuse\_CompAlpha | MapObjTwoLayerDiffuseAlpha |
| 21 - Lod | MapObjDiffuse\_T1 | MapObjLod |
| 22 - Parallax | MapObjParallax | MapObjParallax | SMOMaterial::SH\_PARALLAX\_ICE |
| ≥ 23 - UnkDFShader | MapObjDiffuse\_T1 | MapObjUnkDFShader | This shader can use additional texture file IDs from color\_2, flags\_2 and runTimeData. |

### CMapObj::CreateMaterial

```
void CMapObj::CreateMaterial (unsigned int materialId) { assert (m_materialCount); assert (m_materialTexturesList); assert (materialId < m_materialCount); if (++m_materialTexturesList[materialId].refcount <= 1) { SMOMaterial* material = &m_smoMaterials[materialId]; const char* texNames[3]; texNames[0] = &m_textureFilenamesRaw[material->firstTextureOffset]; texNames[1] = &m_textureFilenamesRaw[material->secondTextureOffset]; texNames[2] = &m_textureFilenamesRaw[material->thirdTextureOffset]; if ( *texNames[0] ) texNames[0] = "createcrappygreentexture.blp"; assert (material->shader < SMOMaterial::SH_COUNT); int const textureCount ( CShaderEffect::s_enableShaders ? s_wmoShaderMetaData[material->shader].texturesWithShader : s_wmoShaderMetaData[material->shader].texturesWithoutShader ); int textures_set (0); for (; textures_set < textureCount; ++textures_set) { if (!texNames[textures_set]) { material->shader = MapObjOpaque; textures_set = 1; break; } } for (; textures_set < 3; ++textures_set) { texNames[textures_set] = nullptr; } if (material->shader = MapObjTwoLayerTerrain && texNames[1]) { texNames[1] = insert_specular_suffix (texNames[1]); } int flags (std::max (m_field_2C, 12)); const char* parent_name (m_field_9E8 & 1 ? m_filename : nullptr); m_materialTexturesList[materialId]->textures[0] = texNames[0] ? CMap::CreateTexture (texNames[0], parent_name, flags) : nullptr; m_materialTexturesList[materialId]->textures[1] = texNames[1] ? CMap::CreateTexture (texNames[1], parent_name, flags) : nullptr; m_materialTexturesList[materialId]->textures[2] = texNames[2] ? CMap::CreateTexture (texNames[2], parent_name, flags) : nullptr; } }
```

## MOM3

This section only applies to versions ≥ (11.0.0.54210).

```
struct { m3SI m3SI; } material3
```

Future chunk, defines new materials. If present, materials from MOMT are not used at all and materials from this chunk are used instead. This chunk's data starts m3SI code word and has same structure as in M3 file

## MOUV

This section only applies to versions ≥ (7.3.0.24473).

Optional. If not present, values are {0, 0, 0, 0} for all materials. If present, has same count as materials, so is repeating those zeros for materials not using any transformation. Currently, only a translating animation is possible for two of the texture layers.

```
struct { C2Vector translation_speed[2]; } MapObjectUV[count(materials)];
```

The formula from translation\_speed values to TexMtx translation values is along the lines of

```
a_i = translation_i ? 1000 / translation_i : 0 b_i = a_i ? (a_i < 0 ? (1 - (time? % -a_i) / -a_i) : ((time? % a_i) / a_i)) : 0
```

Note: Until (7.3.0.24920) (i.e. just before release), a missing break; in the engine's loader will overwrite the data for MOGN with that of MOUV if MOUV comes second. Since MOGN comes second in Blizzard-exported files it works for those without issue.

## MOGN chunk

* List of group names for the groups in this map object.

```
char groupNameList[];
```

A contiguous block of zero-terminated strings. The names are purely informational except for "antiportal". The names are referenced from MOGI and MOGP.

There are not always nGroups entries in this chunk as it contains extra empty strings and descriptive names. There are also empty entries. The names are indeed referenced in MOGI, and both the name and a descriptive name are referenced in the group file header (2 firsts uint16 of MOGP).

Looks like ASCII but is not: BWL e.g. has ’, so probably UTF-8. (In fact windows-1252 will work)

## MOGI chunk

* Group information for WMO groups, 32 bytes per group, nGroups entries.

```
struct SMOGroupInfo { #if ≤ (0.5.5.3494) uint32_t offset; // absolute address uint32_t size; // includes IffChunk header #endif /*000h*/ uint32_t flags; // see information in in MOGP, they are equivalent /*004h*/ CAaBoxi bounding_box; /*01Ch*/ int32_t nameoffset; // name in MOGN chunk (-1 for no name) /*020h*/ } groupInfoList[];
```

Groups don't have placement or orientation information, because the coordinates for the vertices in the additional. WMO files are already correctly transformed relative to (0,0,0) which is the entire WMO's base position in model space.

The name offsets point to the position in the file relative to the MOGN header.

## MOSB chunk (optional)u

* Skybox. Contains an zero-terminated filename for a skybox. (padded to 4 byte alignment if "empty"). If the first byte is 0, the skybox flag in all MOGI entries are cleared and there is no skybox.

```
char skyboxName[];
```

## MOSI (optional)

This section only applies to versions ≥ (8.1.0.27826). Could have been added earlieru.

Equivalent to MOSB, but a file data id. Client supports reading both for now.

```
uint32_t skyboxFileId;
```

## MOPV chunk

* Portal vertices, one entry is a float[3], usually 4 \* 3 \* float per portal (actual number of vertices given in portal entry)

```
C3Vectori portalVertexList[];
```

Portals are polygon planes (usually quads, but they can have more complex shapes) that specify where separation points between groups in a WMO are - these are usually doors or entrances, but can be placed elsewhere. Portals are used for occlusion culling, and is a known rendering technique used in many games (among them Unreal Tournament 2004 and Descent. See Portal Rendering on Wikipedia and Antiportal on Wikipedia for more information.

Since when "playing" WoW, you're confined to the ground, checking for passing through these portals would be enough to toggle visibility for indoors or outdoors areas, however, when randomly flying around, this is not necessarily the case.

So.... What happens when you're flying around on a gryphon, and you fly into that arch-shaped portal into Ironforge? How is that portal calculated? It's all cool as long as you're inside "legal" areas, I suppose.

It's fun, you can actually map out the topology of the WMO using this and the MOPR chunk. This could be used to speed up the rendering once/if I figure out how.

This image explains how portal equation in MOPT and relations in MOPR are connected: Portal explanation. Deamon (talk) 17:06, 23 February 2017 (CET)

## MOPT chunk

* Portal information. 20 bytes per portal, nPortals entries. There is a hardcoded maximum of 128 portals in a single WMO.

```
struct SMOPortal { uint16_t startVertex; uint16_t count; C4Planei plane; } portalList[];
```

This structure describes one portal separating two WMO groups. A single portal is usually made up of four vertices in a quad (starting at startVertex and going to startVertex + count). However, portals support more complex shapes, and can fully encompass holes such as the archway leading into Ironforge and parts of the Caverns of Time.

It is likely that portals are drawn as GL\_TRIANGLE\_STRIP in WoW's occlusion pipeline, since some portals have a vertex count that is not evenly divisible by four. One example of this is portal #21 in CavernsOfTime.wmo from Build #5875 (WoW 1.12.1), which has 10 vertices.

## MOPR chunk

* Map Object Portal References from groups. Mostly twice the number of portals. Actual count defined by sum (MOGP.portals\_used).

```
struct SMOPortalRef // 04-29-2005 By ObscuR { uint16_t portalIndex; // into MOPT uint16_t groupIndex; // the other one int16_t side; // positive or negative. uint16_t filler; } portalRefList[];
```

## MOPE chunk

This section only applies to versions ≥ (11.1.0.58221). Could have been added earlieru.

No clue about actual structure outside of the index that seemingly match MOPR values.

```
struct MOPEEntry { uint32_t portalIndex; // into MOPT uint32_t unk1; uint32_t unk1; uint32_t unk3; } MOPE[];
```

## MOVV chunk

Chunk is since ≥ (8.1.0.28294) optional

* Visible block vertices, 0xC byte per entry.

Just a list of vertices that corresponds to the visible block list.

```
C3Vectori visible_block_vertices[];
```

## MOVB chunk

Chunk is since ≥ (8.1.0.28294) optional

* Visible block list

```
struct { uint16_t firstVertex; uint16_t count; ) visible_blocks[];
```

## MOLT chunk

* Lighting information. 48 bytes per light, nLights entries

```
struct SMOLight { enum LightType { OMNI_LGT = 0, SPOT_LGT = 1, DIRECT_LGT = 2, AMBIENT_LGT = 3, }; /*000h*/ uint8_t type; /*001h*/ uint8_t useAtten; /*002h*/ uint8_t pad[2]; // not padding as of v16 /*004h*/ CImVectori color; /*008h*/ C3Vectori position; /*014h*/ float intensity; #if ≥ (0.6.0.3592) /*018h*/ C4Quaternioni rotation; // not needed by omni lights or ambient lights, but is likely used by spot and direct lights. #endif /*028h*/ float attenStart; /*02Ch*/ float attenEnd; } lightList[];
```

First 4 uint8\_t are probably flags, mostly with the values (0,1,1,1).

I haven't quite figured out how WoW actually does lighting, as it seems much smoother than the regular vertex lighting in my screenshots. The light parameters might be range or attenuation information, or something else entirely. Some WMO groups reference a lot of lights at once.

The WoW client (at least on my system) uses only one light, which is always directional. Attenuation is always (0, 0.7, 0.03). So I suppose for...he BSP one and unbatched geometry marked with material ID 0xFF and collision flag in MOPY. In practice, ommiting those collision faces from BSP also yields incorrect results, resulting into the absence of collision for those faces as well as culling issues for indoor groups. --Skarn (talk) 15:56, 17 April 2022 (CEST)

An object could have has 2 collision system. The first one is encoded in a simplified Geometry (when MOPY. MaterialID=0xFF) the second one is encoded in T\_BSP\_NODE. Some object has collision method 1 only, some other uses method 2 only. Some object have both collision systems (some polygons are missing in
