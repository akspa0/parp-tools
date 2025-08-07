<div style="float: right; margin-left: 5px;">__TOC__</div>
[[WMO|WMO]] files contain world map objects. They, too, have a [[Chunk|chunked]] structure just like the [[WDT]] files.

There are two types of [[WMO]] files, actually:

*[[WMO#WMO_root_file|WMO root file]] - lists textures ([[BLP]] Files), doodads ([[M2]] or [[MDX]] Files), etc., and orientation for the [[WMO]] groups
*[[WMO#WMO_group_file|WMO group file]] - 3d model data for one unit in the world map object 

The root file and the groups are stored with the following filenames:

*World\wmo\path\WMOName.wmo
*World\wmo\path\WMOName_NNN.wmo

There is a hardcoded maximum of 512 group files per root object.

{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
In the alpha, [[WMO]] files were a single file rather than being split into root and group. For that reason the root data has been wrapped in a [[#MOMO|MOMO]] chunk followed by the [[#MOGP_chunk|MOGP]] chunks.

=MVER=
 uint32_t version;     // {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.6.0.3592|max_exclusive=1}} 14, {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592|max_expansionlevel=1|max_exclusive=1}} 16,  {{Template:Sandbox/VersionRange|min_expansionlevel=1}} 17

There never have been any additional versions after the alpha, even though the format changed a lot. Classic Blizzard.

= WMO root file =

The root file lists the following:

* textures ([[BLP]] File references)
* materials
* models ([[M2|MDX / M2]] File references)
* groups
* visibility information
* more data

{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}}
In version 14, the version used in the alpha, the root WMO file has an additional container <code>MOMO</code> chunk, like the <code>MOGP</code> chunk, containing all group data.

==MOMO==
{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494|note=Only used in v14.}}

Rather than all chunks being top level, they have been wrapped in <code>MOMO</code>. There has been no other additional data, rather than just everything being wrapped.

==  MOHD chunk ==

*'''Header for the map object. 64 bytes.'''

 struct SMOHeader
 {
 /*000h*/  uint32_t nTextures;    
 /*004h*/  uint32_t nGroups;    
 /*008h*/  uint32_t nPortals;   
 /*00Ch*/  uint32_t nLights;                                        // {{Template:Unverified|Blizzard seems to add one to the MOLT entry count when there are MOLP chunks in the groups (and maybe for MOLS too?)}}
 /*010h*/  uint32_t nDoodadNames; 
 /*014h*/  uint32_t nDoodadDefs;                                    // *
 /*018h*/  uint32_t nDoodadSets;    
 /*01Ch*/  {{Template:Type|CArgb}} ambColor;                                         // Color settings for base (ambient) color. See the flag at /*03Ch*/.   
 /*020h*/  {{Template:Type/foreign_key|table=WMOAreaTable|column=m_WMOID}} wmoID;
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
 /*0x24*/  uint8_t padding[0x1c];
 #else 
 /*024h*/  {{Template:Type|CAaBox}} bounding_box;                                    // in the alpha, this bounding box was computed upon loading
 /*03Ch*/  uint16_t flag_do_not_attenuate_vertices_based_on_distance_to_portal : 1;
 /*03Ch*/  uint16_t flag_use_unified_render_path : 1;               // In 3.3.5a this flag switches between classic render path (MOHD color is baked into MOCV values, all three batch types have their own rendering logic) and unified (MOHD color is added to lighting at runtime, int. and ext. batches share the same rendering logic). See [[https://wowdev.wiki/WMO/Rendering]] for more details.
 /*03Ch*/  uint16_t flag_use_liquid_type_dbc_id : 1;                // use real liquid type ID from DBCs instead of local one. See MLIQ for further reference.
 /*03Ch*/  uint16_t flag_do_not_fix_vertex_color_alpha: 1;          // In 3.3.5.a (and probably before) it prevents CMapObjGroup::FixColorVertexAlpha function to be executed. Alternatively, for the wotlk version of it, the function can be called with MOCV.a being set to 64, whjch will produce the same effect for easier implementation. For wotlk+ rendering, it alters the behavior of the said function instead. See [[https://wowdev.wiki/WMO/Rendering]] for more details.
 /*03Ch*/  uint16_t flag_lod : 1;                                   // {{Template:Sandbox/VersionRange|min_expansionlevel=7|min_build=20740}}
 /*03Ch*/  uint16_t flag_default_max_lod : 1;                       // {{Unverified|{{Template:Sandbox/VersionRange|min_expansionlevel=7|min_build=21796}}}}. Usually maxLodLevel = -1 but with this flag, numLod. Entries at this level are defaulted
 /*03Ch*/  uint16_t : 10;                                           // unused as of {{Template:Sandbox/PrettyVersion|expansionlevel=7|build=20994}}
 /*03Eh*/  uint16_t numLod;                                         // {{Template:Sandbox/VersionRange|min_expansionlevel=7|min_build=21108}} includes base lod (→ numLod = 3 means '.wmo', 'lod0.wmo' and 'lod1.wmo')
 #endif
 } header;

==  MOTX chunk ==
{{SectionBox/VersionRange|max_build=8.1.0.28186|max_expansionlevel=8|max_exclusive=1|note=MOTX has been replaced with file data ids in MOMT}}

*'''Blob of textures filenames ([[BLP]] Files) used in this map object. 

A block of zero-terminated strings which are complete filenames with paths. They are referenced by offset in MOMT. Note that there are also empty strings, so there are "random additional zero bytes" which are just another zero-terminated empty string. _Do not try to parse this chunk out of thin air_. It makes no sense without the information from MOMT.

 char textureNameList[];

{{SectionBox/VersionRange|min_build=8.1.0.28186|min_expansionlevel=8}}

Starting with 8.1, MOTX is no longer used. The texture references in MOMT are file data ids directly. As of that version, there is a fallback mode though and some files still use MOTX for sake of avoiding re-export.
To check if texture references in MOMT are file data ids, simply check if MOTX exists in the file.

==  MOMT chunk ==

*'''Materials used in this map object, 64 bytes per texture ([[BLP]] file).'''

 struct SMOMaterial
 {
 #if {{Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}}  
          uint32_t version;   
 #endif
 
 /*0x00*/ uint32_t F_UNLIT    : 1;                 // disable lighting logic in shader (but can still use vertex colors)
 /*0x00*/ uint32_t F_UNFOGGED : 1;                 // disable fog shading (rarely used)
 /*0x00*/ uint32_t F_UNCULLED : 1;                 // two-sided
 /*0x00*/ uint32_t F_EXTLIGHT : 1;                 // darkened, the intern face of windows are flagged 0x08
 /*0x00*/ uint32_t F_SIDN     : 1;                 // (bright at night, unshaded) (used on windows and lamps in Stormwind, for example) (see emissive color)
 /*0x00*/ uint32_t F_WINDOW   : 1;                 // lighting related (flag checked in CMapObj::UpdateSceneMaterials)
 /*0x00*/ uint32_t F_CLAMP_S  : 1;                 // tex clamp S (force this material's textures to use clamp s addressing)
 /*0x00*/ uint32_t F_CLAMP_T  : 1;                 // tex clamp T (force this material's textures to use clamp t addressing)
 /*0x00*/ uint32_t flag_0x100 : 1;
 /*0x00*/ uint32_t            : 23;                // unused as of 7.0.1.20994
 
 #if {{Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592}}     
 /*0x04*/ uint32_t shader;                         // Index into CMapObj::s_wmoShaderMetaData. See below (shader types).
 #endif
 
 /*0x08*/ uint32_t blendMode;                      // Blending: see [[Rendering#EGxBlend|EGxBlend]]
 /*0x0C*/ uint32_t texture_1;                      // offset into MOTX; {{Sandbox/VersionRange|min_expansionlevel=8|min_build=8.1.0.27826}} No longer references MOTX but is a filedata id directly.
 /*0x10*/ {{Type|CImVector}} sidnColor;                    // emissive color; see below (emissive color)
 /*0x14*/ {{Type|CImVector}} frameSidnColor;               // sidn emissive color; set at runtime; gets sidn-manipulated emissive color; see below (emissive color)
 /*0x18*/ uint32_t texture_2;                      // offset into MOTX or texture file id
 /*0x1C*/ {{Type|CImVector}} diffColor;
 /*0x20*/ {{Type/foreign_key|table=TerrainType}} ground_type;
                                                   // according to CMapObjDef::GetGroundType 
 
 #if {{Sandbox/VersionRange|max_expansionlevel=0|max_build=0.6.0.3592}}
          char inMemPad[8];
 #else  
 
 /*0x24*/ uint32_t texture_3;                      // offset into MOTX or texture file id
 /*0x28*/ uint32_t color_2;                        // For shader 23, this can be a texture file id.
 /*0x2C*/ uint32_t flags_2;                        // For shader 23, this can be a texture file id.
 /*0x30*/ uint32_t runTimeData[4];                 // This data is explicitly nulled upon loading. For shader 23, this can contain textures file ids.
 /*0x40*/
 
 #endif
 } materialList[];

texture_1, 2 and 3 are start positions for texture filenames in the [[WMO#MOTX_chunk|MOTX]] data block ; texture_1 for the first texture, texture_2 for the second (see shaders), etc. texture_1 defaults to "createcrappygreentexture.blp".

{{Unverified|If a texture isn't used the its start position seems to point to a chains of 4 \0}}


color_2 is diffuse color : <tt>CWorldView::GatherMapObjDefGroupLiquids():  geomFactory->SetDiffuseColor(({{Type|CImVector}}*)(smo+7));</tt>

The flags might used to tweak alpha testing values, I'm not sure about it, but some grates and flags in IF seem to require an alpha testing threshold of 0, at other places this is greater than 0.

===Texture addressing===

By default, textures used by WMO materials are assigned an addressing mode of <tt>EGxTexWrapMode::GL_REPEAT</tt> (ie wrap mode).

<tt>SMOMaterial</tt> flags <tt>F_CLAMP_S</tt> and <tt>F_CLAMP_T</tt> can override this default to clamp mode for the <tt>S</tt> and <tt>T</tt> dimensions, respectively.

===Emissive color===

The <tt>sidnColor</tt> {{Type|CImVector}} at offset <tt>0x10</tt> is used with the SIDN (self-illuminated day night) scalar from <tt>CDayNightObject</tt> to light exterior window glows (see flag <tt>0x10</tt> above).

The scalar is interpolated out of a static table in the client, based on the time of day.

The color value eventually is copied into offset <tt>0x14</tt> (<tt>frameSidnColor</tt>) after being manipulated by the SIDN scalar. This manipulation occurs in <tt>CMapObj::UpdateMaterials</tt>.

===Shader types (12340)===

Wrath of the Lich King only uses shaders 0 to 6. See below for more info on those.

===Shader types (15464)===

Depending on the shader, a different amount of textures is required. If there aren't enough filenames given, it defaults to Opaque (with one filename). More filenames than required are just ignored.

Data is from 15464.
{| class="wikitable"
|- 
! value 
! name
! textures without shader
! textures with shader 
! texcoord count
! color count
|-
| 0 || Diffuse || 1 || 1 || 1 || 1
|-
| 1 || Specular || 1 || 1 || 1 || 1
|-
| 2 || Metal || 1 || 1 || 1 || 1
|-
| 3 || Env || 1 || 2 || 1 || 1
|-
| 4 || Opaque || 1 || 1 || 1 || 1
|-
| 5 || EnvMetal || 1 || 2 || 1 || 1
|-
| 6 || TwoLayerDiffuse || 1 || 2 || 2 || 2
|-
| 7 || TwoLayerEnvMetal || 1 || 3 || 2 || 2
|-
| 8 || TwoLayerTerrain || 1 || 2 || 1 || 2 || automatically adds _s in the filename of the second texture
|-
| 9 || DiffuseEmissive || 1 || 2 || 2 || 2
|-
| 10 || || 1 || 1 || 1 || 1 || SMOMaterial::SH_WATERWINDOW -- Seems to be invalid. Does something with MOTA (tangents).
|-
| 11 || MaskedEnvMetal || 1 || 3 || 2 || 2
|-
| 12 || EnvMetalEmissive || 1 || 3 || 2 || 2
|-
| 13 || TwoLayerDiffuseOpaque || 1 || 2 || 2 || 2
|-
| 14 || TwoLayerDiffuseEmissive || 1 || 1 || 1 || 1 || SMOMaterial::SH_SUBMARINEWINDOW -- Seems to be invalid. Does something with MOTA (tangents).
|-
| 15 || || 1 || 2 || 2 || 2
|-
| 16 || Diffuse || 1 || 1 || 1 || 1 || SMOMaterial::SH_DIFFUSE_TERRAIN -- "Blend Material": used for blending WMO with terrain (dynamic blend batches)
|}

tex coord and color count decide vertex buffer format: EGxVertexBufferFormat_PNC''2''T''2''

===Shader types (18179)===
{| class="wikitable"
|-
! value
! #textures without shader
! #textures with shader
! texcoord count
! color count
|-
| 0 - Diffuse || 1 || 1 || 1 || 1
|-
| 1 - Specular || 1 || 1 || 1 || 1
|-
| 2 - Metal || 1 || 1 || 1 || 1
|-
| 3 - Env || 1 || 2 || 1 || 1
|-
| 4 - Opaque || 1 || 1 || 1 || 1
|-
| 5 - EnvMetal || 1 || 2 || 1 || 1
|-
| 6 - TwoLayerDiffuse || 1 || 2 || 2 || 2
|-
| 7 - TwoLayerEnvMetal || 1 || 3 || 2 || 2
|-
| 8 - TwoLayerTerrain || 1 || 2 || 1 || 2 || automatically adds _s in the filename of the second texture
|-
| 9 - DiffuseEmissive || 1 || 2 || 2 || 2
|-
| 10 - waterWindow || 1 || 1 || 1 || 1 || SMOMaterial::SH_WATERWINDOW -- automatically generates MOTA
|-
| 11 - MaskedEnvMetal || 1 || 3 || 2 || 2
|-
| 12 - EnvMetalEmissive || 1 || 3 || 2 || 2
|-
| 13 - TwoLayerDiffuseOpaque || 1 || 2 || 2 || 2
|-
| 14 - submarineWindow || 1 || 1 || 1 || 1 || SMOMaterial::SH_SUBMARINEWINDOW -- automatically generates MOTA
|-
| 15 - TwoLayerDiffuseEmissive || 1 || 2 || 2 || 2
|-
| 16 - DiffuseTerrain || 1 || 1 || 1 || 1 || SMOMaterial::SH_DIFFUSE_TERRAIN -- "Blend Material": used for blending WMO with terrain (dynamic blend batches)
|-
| 17 - AdditiveMaskedEnvMetal || 1 || 3 || 2 || 2
|}


===Shader types (26522)===
{| class="wikitable"
|-
! value
! vertex shader
! pixel shader
|-
| 0 - Diffuse || MapObjDiffuse_T1 || MapObjDiffuse
|-
| 1 - Specular || MapObjSpecular_T1 || MapObjSpecular
|-
| 2 - Metal || MapObjSpecular_T1 || MapObjMetal
|-
| 3 - Env || MapObjDiffuse_T1_Refl || MapObjEnv
|-
| 4 - Opaque || MapObjDiffuse_T1|| MapObjOpaque
|-
| 5 - EnvMetal || MapObjDiffuse_T1_Refl || MapObjEnvMetal
|-
| 6 - TwoLayerDiffuse || MapObjDiffuse_Comp || MapObjTwoLayerDiffuse
|-
| 7 - TwoLayerEnvMetal || MapObjDiffuse_T1 || MapObjTwoLayerEnvMetal
|-
| 8 - TwoLayerTerrain || MapObjDiffuse_Comp_Terrain || MapObjTwoLayerTerrain || automatically adds _s in the filename of the second texture
|-
| 9 - DiffuseEmissive || MapObjDiffuse_Comp || MapObjDiffuseEmissive 
|-
| 10 - waterWindow || FFXWaterWindow || FFXWaterWindow || It's FFX instead of normal material. SMOMaterial::SH_WATERWINDOW -- automatically generates MOTA
|-
| 11 - MaskedEnvMetal || MapObjDiffuse_T1_Env_T2 || MapObjMaskedEnvMetal 
|-
| 12 - EnvMetalEmissive || MapObjDiffuse_T1_Env_T2 || MapObjEnvMetalEmissive 
|-
| 13 - TwoLayerDiffuseOpaque || MapObjDiffuse_Comp || MapObjTwoLayerDiffuseOpaque 
|-
| 14 - submarineWindow || FFXSubmarineWindow|| FFXSubmarineWindow|| It's FFX instead of normal material. SMOMaterial::SH_SUBMARINEWINDOW -- automatically generates MOTA
|-
| 15 - TwoLayerDiffuseEmissive || MapObjDiffuse_Comp || MapObjTwoLayerDiffuseEmissive
|-
| 16 - DiffuseTerrain || MapObjDiffuse_T1 || MapObjDiffuse || SMOMaterial::SH_DIFFUSE_TERRAIN -- "Blend Material": used for blending WMO with terrain (dynamic blend batches)
|-
| 17 - AdditiveMaskedEnvMetal || MapObjDiffuse_T1_Env_T2 || MapObjAdditiveMaskedEnvMetal 
|-
| 18 - TwoLayerDiffuseMod2x|| MapObjDiffuse_CompAlpha || MapObjTwoLayerDiffuseMod2x
|-
| 19 - TwoLayerDiffuseMod2xNA|| MapObjDiffuse_Comp || MapObjTwoLayerDiffuseMod2xNA
|-
| 20 - TwoLayerDiffuseAlpha|| MapObjDiffuse_CompAlpha|| MapObjTwoLayerDiffuseAlpha
|-
| 21 - Lod || MapObjDiffuse_T1 || MapObjLod 
|-
| 22 - Parallax || MapObjParallax || MapObjParallax || SMOMaterial::SH_PARALLAX_ICE
|-
| {{Sandbox/VersionRange|min_expansionlevel=10}}  23 - UnkDFShader || MapObjDiffuse_T1 || MapObjUnkDFShader || This shader can use additional texture file IDs from color_2, flags_2 and runTimeData.
|}

=== CMapObj::CreateMaterial ===

 void CMapObj::CreateMaterial (unsigned int materialId)
 {
   assert (m_materialCount);
   assert (m_materialTexturesList);
   assert (materialId < m_materialCount);
 
   if (++m_materialTexturesList[materialId].refcount <= 1)
   {
     SMOMaterial* material = &m_smoMaterials[materialId];
 
     const char* texNames[3];
     texNames[0] = &m_textureFilenamesRaw[material->firstTextureOffset];
     texNames[1] = &m_textureFilenamesRaw[material->secondTextureOffset];
     texNames[2] = &m_textureFilenamesRaw[material->thirdTextureOffset];
     if ( *texNames[0] )
       texNames[0] = "createcrappygreentexture.blp";
 
     assert (material->shader < SMOMaterial::SH_COUNT);
 
     int const textureCount
       ( CShaderEffect::s_enableShaders
       ? s_wmoShaderMetaData[material->shader].texturesWithShader
       : s_wmoShaderMetaData[material->shader].texturesWithoutShader
       );
 
     int textures_set (0);
 
     for (; textures_set < textureCount; ++textures_set)
     {
       if (!texNames[textures_set])
       {
         material->shader = MapObjOpaque;
         textures_set = 1;
         break;
       }
     }
 
     for (; textures_set < 3; ++textures_set)
     {
       texNames[textures_set] = nullptr;
     }
 
     if (material->shader == MapObjTwoLayerTerrain && texNames[1])
     {
       texNames[1] = insert_specular_suffix (texNames[1]);
     }
 
     int flags (std::max (m_field_2C, 12));
 
     const char* parent_name (m_field_9E8 & 1 ? m_filename : nullptr);
 
     m_materialTexturesList[materialId]->textures[0] = texNames[0] ? CMap::CreateTexture (texNames[0], parent_name, flags) : nullptr;
     m_materialTexturesList[materialId]->textures[1] = texNames[1] ? CMap::CreateTexture (texNames[1], parent_name, flags) : nullptr;
     m_materialTexturesList[materialId]->textures[2] = texNames[2] ? CMap::CreateTexture (texNames[2], parent_name, flags) : nullptr;
   }
 }

==MOM3==
{{Template:SectionBox/VersionRange|min_expansionlevel=11|min_build=11.0.0.54210}}
  struct {
    [[M3#M3SI|m3SI]] m3SI; 
  } material3

Future chunk, defines new materials. If present, materials from MOMT are not used at all and materials from this chunk are used instead. This chunk's data starts <code>m3SI</code> code word and has same structure as in [[M3#M3SI|M3 file]] 

==MOUV==
{{Template:SectionBox/VersionRange|min_expansionlevel=7|min_build=7.3.0.24473}}

Optional. If not present, values are <tt>{0, 0, 0, 0}</tt> for all materials. If present, has same count as materials, so is repeating those zeros for materials not using any transformation. Currently, only a translating animation is possible for two of the texture layers.

 struct 
 {
   C2Vector translation_speed[2];
 } MapObjectUV[count(materials)];

The formula from <tt>translation_speed</tt> values to <tt>TexMtx</tt> translation values is along the lines of 

 a_i = translation_i ? 1000 / translation_i : 0
 b_i = a_i ? (a_i < 0 ? (1 - (time? % -a_i) / -a_i) : ((time? % a_i) / a_i)) : 0

Note: Until {{Template:Sandbox/PrettyVersion|expansionlevel=7|build=7.3.0.24920}} (i.e. just before release), a missing <tt>break;</tt> in the engine's loader will overwrite the data for MOGN with that of MOUV if MOUV comes second. Since MOGN comes second in Blizzard-exported files it works for those without issue.

== MOGN chunk ==

*'''List of group names for the groups in this map object.'''

 char groupNameList[];

A contiguous block of zero-terminated strings. The names are purely informational except for "antiportal". The names are referenced from MOGI and MOGP.

There are '''not''' always nGroups entries in this chunk as it contains extra empty strings and descriptive names. There are also empty entries. The names are indeed referenced in MOGI, and both the name and a descriptive name are referenced in the group file header (2 firsts uint16 of MOGP).

Looks like ASCII but is not: BWL e.g. has <tt>’</tt>, so probably UTF-8. (In fact windows-1252 will work)

==  MOGI chunk ==

*'''Group information for WMO groups, 32 bytes per group, nGroups entries.'''

 struct SMOGroupInfo
 {
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
   uint32_t offset;             // absolute address
   uint32_t size;               // includes IffChunk header
 #endif
 /*000h*/  uint32_t flags;      //  see [[WMO#group_flags|information in in MOGP]], they are equivalent
 /*004h*/  {{Template:Type|CAaBox}} bounding_box;
 /*01Ch*/  int32_t nameoffset;  // name in [[WMO#MOGN_chunk|MOGN]] chunk (-1 for no name)
 /*020h*/
 } groupInfoList[];

Groups don't have placement or orientation information, because the coordinates for the vertices in the additional. [[WMO]] files are already correctly transformed relative to (0,0,0) which is the entire [[WMO]]'s base position in model space.

The name offsets point to the position in the file relative to the MOGN header.

==  MOSB chunk  {{Unverified|(optional)}} ==

*'''Skybox.''' Contains an zero-terminated filename for a skybox. (padded to 4 byte alignment if "empty"). If the first byte is 0, the skybox flag in all MOGI entries are cleared and there is no skybox.

 char skyboxName[];

== MOSI (optional) ==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}

Equivalent to MOSB, but a file data id. Client supports reading both for now.

 uint32_t skyboxFileId;

==  MOPV chunk ==

*'''Portal vertices, one entry is a float[3], '''usually''' 4 * 3 * float per portal''' (actual number of vertices given in portal entry)

 {{Template:Type|C3Vector}} portalVertexList[];

Portals are polygon planes (usually quads, but they can have more complex shapes) that specify where separation points between groups in a [[WMO]] are - these are usually doors or entrances, but can be placed elsewhere. Portals are used for occlusion culling, and is a known rendering technique used in many games (among them ''Unreal Tournament 2004'' and ''Descent''. See [https://en.wikipedia.org/wiki/Portal_rendering Portal Rendering on Wikipedia] and [https://en.wikipedia.org/wiki/Antiportal Antiportal on Wikipedia] for more information.

Since when "playing" WoW, you're confined to the ground, checking for passing through these portals would be enough to toggle visibility for indoors or outdoors areas, however, when randomly flying around, this is not necessarily the case.

So.... What happens when you're flying around on a gryphon, and you fly into that arch-shaped portal into Ironforge? How is that portal calculated? It's all cool as long as you're inside "legal" areas, I suppose. 

It's fun, you can actually map out the topology of the [[WMO]] using this and the [[WMO#MOPR_chunk|MOPR]] chunk. This could be used to speed up the rendering once/if I figure out how.


This image explains how portal equation in MOPT and relations in MOPR are connected: [[:File:Portal Explanation.png|Portal explanation]]. [[User:Deamon|Deamon]] ([[User talk:Deamon|talk]]) 17:06, 23 February 2017 (CET)

==  MOPT chunk ==

*'''Portal information. 20 bytes per portal, nPortals entries.''' There is a hardcoded maximum of 128 portals in a single WMO.

 struct SMOPortal
 {
   uint16_t startVertex;
   uint16_t count;
   {{Template:Type|C4Plane}} plane;
 } portalList[];

This structure describes one portal separating two WMO groups. A single portal is usually made up of four vertices in a quad (starting at startVertex and going to startVertex + count). However, portals support more complex shapes, and can fully encompass holes such as the archway leading into Ironforge and parts of the Caverns of Time.

It is likely that portals are drawn as GL_TRIANGLE_STRIP in WoW's occlusion pipeline, since some portals have a vertex count that is not evenly divisible by four. One example of this is portal #21 in CavernsOfTime.wmo from Build #5875 (WoW 1.12.1), which has 10 vertices.

==  MOPR chunk ==

* Map Object Portal References from groups. Mostly twice the number of portals. Actual count defined by sum (MOGP.portals_used).

 struct SMOPortalRef'' // 04-29-2005 By ObscuR''
 {
   uint16_t portalIndex;  // into MOPT
   uint16_t groupIndex;   // the other one
   int16_t side;          // positive or negative.
   uint16_t filler;
 } portalRefList[];

==  MOPE chunk ==
{{SectionBox/VersionRange|min_build=11.1.0.58221|min_expansionlevel=11|note={{Unverified|Could have been added earlier}}}}

No clue about actual structure outside of the index that seemingly match MOPR values.
 struct MOPEEntry
 {
   uint32_t portalIndex;  // into MOPT
   uint32_t unk1;
   uint32_t unk1;
   uint32_t unk3;
 } MOPE[];

==  MOVV chunk ==
Chunk is since {{Template:Sandbox/VersionRange|min_expansionlevel=8|min_build=8.1.0.28294}} optional

*'''Visible block vertices''', 0xC byte per entry.

Just a list of vertices that corresponds to the visible block list.

 {{Template:Type|C3Vector}} visible_block_vertices[];

==  MOVB chunk ==
Chunk is since {{Template:Sandbox/VersionRange|min_expansionlevel=8|min_build=8.1.0.28294}} optional
*'''Visible block list'''

 struct
 {
   uint16_t firstVertex;
   uint16_t count;
 ) visible_blocks[];

==  MOLT chunk ==

*'''Lighting information. 48 bytes per light, nLights entries'''

 struct SMOLight
 {
   enum LightType
   {
     OMNI_LGT = 0,
     SPOT_LGT = 1,
     DIRECT_LGT = 2,
     AMBIENT_LGT = 3,
   };
   /*000h*/  uint8_t type;
   /*001h*/  uint8_t useAtten;
   /*002h*/  uint8_t pad[2];      // not padding as of v16
   /*004h*/  {{Template:Type|CImVector}} color;
   /*008h*/  {{Template:Type|C3Vector}} position;
   /*014h*/  float intensity;
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592}}
   /*018h*/  {{Template:Type|C4Quaternion}} rotation;     // not needed by omni lights or ambient lights, but is likely used by spot and direct lights.
 #endif
   /*028h*/  float attenStart;
   /*02Ch*/  float attenEnd;
 } lightList[];

First 4 uint8_t are probably flags, mostly with the values (0,1,1,1).

I haven't quite figured out how WoW actually does lighting, as it seems much smoother than the regular vertex lighting in my screenshots. The light parameters might be range or attenuation information, or something else entirely. Some [[WMO]] groups reference a lot of lights at once.

The WoW client (at least on my system) uses only one light, which is always directional. Attenuation is always (0, 0.7, 0.03). So I suppose for models/doodads (both are [[M2]] files anyway) it selects an appropriate light to turn on. Global light is handled similarly. Some [[WMO]] textures ([[BLP]] files) have specular maps in the alpha channel, the pixel shader renderpath uses these. Still don't know how to determine direction/color for either the outdoor light or [[WMO]] local lights... :)

The entire MOLT and related chunks seem to be unused at least in 3.3.5a. Changing light colors and other settings on original WMOs leads to no effect. Removing the light leads to no effect either. I assume that MOLT rendering is disabled somewhere in the WoW.exe, as it might use the same principle as the M2 light emitters which are not properly supported up to WoD. However, when you explore the WMOs in 3D editors you can clearly see that MOCV layer is different under those lamps. So, I assume they are used for baking MOCV colors and also written to the actual file in case the renderer will ever get updated, or just because you can easily import the WMO back and rebake the colors. --- [[User:Skarn|Skarn]] ([[User talk:Skarn|talk]])

==MOLV==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.1.0.39015}}

Extension to MOLT. Present in file 3623016 (9.1 Broker Dungeon). Not to be confused with the old 0.5.3 MOLV chunk.

 struct {
 /*0x00*/  struct {
   /*0x00*/  {{Type|C3Vector}} direction;      // usually either xy or z and the remainder 0.
   /*0x0C*/  float value;
   /*0x10*/ } _0x00[6];
 /*0x60*/  byte _0x60[3];
 /*0x63*/  uint8_t molt_index;         // multiple MOLV may reference/extend the same MOLT.
 /*0x64*/
 } mapObjectLightV[];

==  MODS chunk ==

*'''This chunk defines doodad sets.''' 

Doodads in WoW are [[M2]] model files. There are 32 bytes per doodad set, and nSets entries. Doodad sets specify several versions of "interior decoration" for a [[WMO]]. Like, a small house might have tables and a bed laid out neatly in one set, and have a horrible mess of abandoned broken things in another set called "Set_Abandoned01".

Sets are exclusive except for the very first one, "Set_$DefaultGlobal" which is additive and is always displayed. The client determines that set by index, not name though. Up to 8 doodad sets can be enabled at the same time, e.g. via destructible buildings or garrisons.

The doodad set number for every WMO instance is specified in the [[ADT]] files, or via DBC or via game object fields, depending on how it is spawned.

 struct SMODoodadSet
 {
 /*0x00*/  char     name[0x14];     // set name, informational
 /*0x14*/  uint32_t startIndex;     // index of first doodad instance in this set, into [[#MODD_chunk|MODD]] directly.
 /*0x18*/  uint32_t count;          // number of doodad instances in this set
 /*0x1C*/  char     pad[4];
 /*0x20*/
 } doodadSetList[];

==  MODN chunk ==

*'''List of filenames for [[M2]] ([[MDX|mdx]]) models that appear in this [[WMO]].''' 
A block of zero-padded, zero-terminated strings. There are nModels file names in this list. They have to be .[[MDX]]!

 char doodadNameList[];

==  MODI chunk ==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note=Replaces filenames in [[#MODN chunk|MODN]]}}

 uint32_t doodad_ID[];    // should be same count as SMOHeader.nDoodadNames in [[WMO#MOHD_chunk|MOHD]] chunk

==  MODD chunk ==

*'''Information for doodad instances. 40 bytes per doodad instance, nDoodads entries.''' 

-- There are not nDoodads entries here! Divide the chunk length by 40 to get the correct amount.

While [[WMO]]s and models ([[M2]]s) in a map tile are rotated along the axes, doodads within a [[WMO]] are oriented using quaternions! Hooray for consistency!

I had to do some tinkering and mirroring to orient the doodads correctly using the quaternion, see model.cpp in the WoWmapview source code for the exact transform matrix. It's probably because I'm using another coordinate system, as a lot of other coordinates in [[WMO]]s and models also have to be read as (X,Z,-Y) to work in my system. But then again, the [[ADT]] files have the "correct" order of coordinates. Weird.

 struct SMODoodadDef
 {
   /*000h*/  uint32_t nameIndex : 24;          // reference offset into [[WMO#MODN_chunk|MODN]], or [[WMO#MODI_chunk|MODI]], depending on version and presence. Note that if MODN is used it is a byte offset to the beginning of filepath 0-terminated string, for MODI it is an index.
   /*003h*/  uint32_t flag_AcceptProjTex : 1;  // If set, the doodad can have textures projected onto it, e.g. the npc selection circle
   /*003h*/  uint32_t flag_0x2 : 1;            // MapStaticEntity::field_34 |= 1 (if set, MapStaticEntity::AdjustLighting is _not_ called). When enabled uses wmo interior lighting function, when disabled uses exterior lighting
   /*003h*/  uint32_t flag_0x4 : 1;
   /*003h*/  uint32_t flag_0x8 : 1;
   /*003h*/  uint32_t : 4;                     // unused as of 7.0.1.20994
   /*004h*/  {{Template:Type|C3Vector}} position;               // (X,Z,-Y)
   /*010h*/  {{Template:Type|C4Quaternion}} orientation;        // (X, Y, Z, W)
   /*020h*/  float scale;                      // scale factor
   /*024h*/  {{Template:Type|CImVector}} color;                 // (B,G,R,A) overrides pc_sunColor
                                                                // when A is != 0xff && A < 255, A is a MOLT index and that's used instead the RGB given here, taking distance and intensity into account
                                                                // If A > MOLT count, then MOLT[0] is used
                                                                // If A == 255, the shading direction vector is based on the center of the group and not the sun direction vector, the look-at vector from group bounds center to doodad position
 } doodadDefList[];

It looks like in order to get correct picture the color from SMODoodadDef should be applied only to opaque submeshes of M2. [[User:Deamon|Deamon]] ([[User talk:Deamon|talk]]) 

   
* How to compute a matrix to map WMO's M2 to world coordinates

The coordinate system here is WMO's local coordinate system. It's Z-up already, that differs it from Y-up in '''[[ADT#MODF_chunk|MODF(ADT)]]''', '''[[WDT#MODF_chunk|MODF(WDT)]]'''  and '''[[ADT#MDDF_chunk|MDDF]]''' chunks.
To compute the whole placement matrix for doodad you would need take positionMatrix of WMO from '''[[ADT#MODF_chunk|MODF(ADT)]]''' or '''[[WDT#MODF_chunk|MODF(WDT)]]''' and multiply it by positionMatrix calculated here.

Example implementation in js with gl-matrix library:
<syntaxhighlight lang="javascript">
 function createPlacementMatrix(modd, wmoPlacementMatrix){
     var placementMatrix = mat4.create();
     mat4.identity(placementMatrix);
     mat4.multiply(placementMatrix, placementMatrix, wmoPlacementMatrix);
 
     mat4.translate(placementMatrix, placementMatrix, [modd.pos[0],modd.pos[1], modd.pos[2]]);
 
     var orientMatrix = mat4.create();
     mat4.fromQuat(orientMatrix,
         [modd.rotation[0], //imag.x
         modd.rotation[1],  //imag.y,
         modd.rotation[2],  //imag.z,
         modd.rotation[3]   //real
         ]
     );
     mat4.multiply(placementMatrix, placementMatrix, orientMatrix);
 
     mat4.scale(placementMatrix, placementMatrix, [modd.scale, modd.scale, modd.scale]);
     return placementMatrix;
 }
</syntaxhighlight>

== MFOG chunk ==

*'''Fog information. Made up of blocks of 48 bytes.'''

 struct SMOFog
 {
   /*000h*/  uint32_t flag_infinite_radius : 1; // F_IEBLEND: Ignore radius in CWorldView::QueryCameraFog
   /*000h*/  uint32_t : 3;                      // unused as of 7.0.1.20994
   /*000h*/  uint32_t flag_0x10 : 1;
   /*000h*/  uint32_t : 27;                     // unused as of 7.0.1.20994
   /*004h*/  {{Template:Type|C3Vector}} pos;
   /*010h*/  float smaller_radius;              // start
   /*014h*/  float larger_radius;               // end
             enum EFogs 
             {
               FOG,
               UWFOG,                           // uw = under water
               NUM_FOGS,
             };
             struct Fog
             {
               float end;
               float start_scalar;              // (0..1) -- minimum distance is end * start_scalar
               {{Template:Type|CImVector}} color;                // The back buffer is also cleared to this colour
   /*018h*/  } fogs[NUM_FOGS];
 } fogList[];

*Fog end: This is the distance at which all visibility ceases, and you see no objects or terrain except for the fog color.
*Fog start: This is where the fog starts. Obtained by multiplying the fog end value by the fog start multiplier.

*There should always be at least one fog entry in MFOG. The empty fog entry has both radiuses set to zero, 444.4445 for end, 0.25 for start_scalar, 222.2222 for underwater end, -0.5 for underwater start_scalar.

*F_IEBLEND - InteriorExteriorBlend
:These fog entries are used to reduce fog visibility based on the player's proximity i.e. the closer you are, the less on-screen fog. They are usually placed near exits to prevent fog showing in unintended places such as behind instance portals (e.g. Stockades fog showing on the Stormwind side of the portal). Whilst not being rendered they are still computed; the resulting blend percentage is applied as a multiplier (<code>1.0 - ComputedBlendPercentage</code>) to the scalar and colour calculations of the area fog.
:This fog ignores all visibility checks (so that the multiplier is always applied) and is excluded from fog queries. Only one is used per <tt>mapObjGroup->fogList</tt> with the last taking precedence. (verified {{Template:Sandbox/VersionRange|max_expansionlevel=3}})

== MCVP chunk (optional) ==

*'''Convex Volume Planes. Contains blocks of floating-point numbers.''' 0x10 bytes (4 floats) per entry.

 {{Template:Type|C4Plane}} convexVolumePlanes[];   // normal points out

These are used to define the volume of when you are inside this WMO. Important for transports. If a point is behind all planes (i.e. point-plane distance is negative for all planes), it is inside.

==GFID==
{{SectionBox/VersionRange|min_expansionlevel=7}}

* required when WMO is load from fileID (e.g. game objects)
 struct {
     uint32 id[MOHD.nGroups];
 } groupFileDataIDs[ !MOHD.Flag_Lod ? 1
                   : MOHD.numLod ? MOHD.numLod : 3   // fallback for missing numLod: assume numLod=2+1base
                   ];

==MDDI==
{{SectionBox/VersionRange|min_build=8.3.0.32044|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
  struct MDDI {
      float colorMult;
  } doodadAdditionalInfo[nDoodads];

The <code>colorMult</code> in the chunk serve as multiplier for <code>color</code> from [[#MODD_chunk|MODD]] chunk

==MPVD==
{{SectionBox/VersionRange|min_build=8.3.0.32044|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
  struct MPVD {
      // Unknown
  } particulateVolumes[];

==MAVG==
{{SectionBox/VersionRange|min_build=8.3.0.32044|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
Same structure as MAVD, except pos/start/end are 0 values because this is a global ambient, but they are still written in the file

 struct MAVG {
  /*000h*/ {{Type|C3Vector}} pos;
  /*00Ch*/ float start;
  /*010h*/ float end;
  /*014h*/ {{Type|CImVector}} color1;
  /*018h*/ {{Type|CImVector}} color2;
  /*01Ch*/ {{Type|CImVector}} color3;
  /*020h*/ uint32_t flags;    // &1: use color1 and color3
  /*024h*/ uint16_t doodadSetID;
  /*026h*/ char _0x26[10];
 } globalAmbientVolumes[];

{{Template:Unverified|WMO base ambient color is now determined from the following:}}
* if MAVG exists use the entry with matching doodadSetID else MAVG[0]
** if (entry.flags & 1) then use entry.color3 and entry.color1 {{Template:Unverified|(secondary ambient color?)}}
** else entry.color1
* else if MAVD exists use MAVD[0]
** if (entry.flags & 1) then use entry.color3 and entry.color1
** else entry.color1
* else use MOHD.ambColor

==MAVD==
{{SectionBox/VersionRange|min_build=8.3.0.32044|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
 struct MAVD {
 /*000h*/ {{Type|C3Vector}} pos;
 /*00Ch*/ float start;
 /*010h*/ float end;
 /*014h*/ {{Type|CImVector}} color1; // overrides MOHD.ambColor
 /*018h*/ {{Type|CImVector}} color2;
 /*01Ch*/ {{Type|CImVector}} color3;
 /*020h*/ uint32_t flags;    // &1: use color2 and color3
 /*024h*/ uint16_t doodadSetId;
 /*026h*/ char _0x26[10];
 } ambientVolumes[];

==MBVD==
{{SectionBox/VersionRange|min_build=8.3.0.32044|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
  struct MBVD {
 /*000h*/ {{Type|C4Plane}} _0x00[6]; // {{Template:Unverified|position + start}}
 /*060h*/ float end;
 /*064h*/ {{Type|CImVector}} color1;
 /*068h*/ {{Type|CImVector}} color2;
 /*06Ch*/ {{Type|CImVector}} color3;
 /*070h*/ uint32_t flags;    // &1: use color2 + color3
 /*074h*/ uint16_t doodadSetId;
 /*076h*/ char _0x76[10];
  } ambientBoxVolumes[];

Only read if a MAVG or MAVD chunk exists.

==MFED==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
 struct MFED {
 /*0x00*/ uint16_t doodadSetId;
 /*0x02*/ char unk1[0xE];
 /*0x10*/
 } m_fog_extra_data[shall be same as MFOG count];

==MGI2==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
  struct MGI2 {
  /*0x00*/  SMOGroupFlags2 flags2;  // a copy of the flags2 that is present in the group file as well
  /*0x04*/  uint32_t lodIndex; // groupInfoList[i].flags & SMOGroup::LOD
  /*0x08*/
  } mapobject_group_info_v2[];
{{Template:Unverified|Used to explicitly control what groups use what level lod}}. If present, overrides the previous lod loading implementation.

Requires exact same count as MOGI. Or replacement of it? Probably replacmeent.

==MNLD==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
*'''These are a new type of dynamic lights added in Shadowlands. E.g. Castle Nathria raid has 833 of them.'''
*'''They're used for everything from torch fires to projecting light/shadow on the ground to make it look like light is coming through a window.'''

  struct MNLD {
    int type;                       // 0 = Point light (sphere), 1 = Spot light (cone) 
    int lightIndex;                 // Appears to be same as index in mapobject_new_light_defs[]
    int enableColorGradient;        // 0 = false (use only startColor), 1 = true (use start and end color gradient)
    int doodadSet;              // Doodad Set this light belongs to
    {{Template:Type|CImVector}} startColor;          // Start Color
    {{Template:Type|C3Vector}} position;             // Light position in WMO
    {{Template:Type|C3Vector}} rotation;             // Euler rotation in radians, for spot light rotates the light, for point light rotates the light cookie
    float attenStart;               // Start attenuation
    float attenEnd;                 // End attenuation
    float intensity;                // Light intensity
    {{Template:Type|CImVector}} endColor;            // End Color
    float colorBlendStart;          // Gradient start distance from emitter position, for mixing start and end color
    float colorBlendEnd;            // Gradient end distance from emitter position,  for mixing start and end color
    char gap0[4];                   // empty
    float flickerIntensity;         // Flickering light intensity
    float flickerSpeed;             // Flickering light speed
    int flickerMode;                // 0 = off, 1 = sine curve, 2 = noise curve, 3 = noise step curve
    {{Template:Type|C3Vector}} field_54;             // Only found 0's so far
    char gap1[4];                   // empty
    uint lightCookieFileID;         // file ID for light cookie texture. For point light it's a cube map
    char gap2[20];                  // empty
    float spotlightRadius;          // The overall radius of the spot light, in radians
    float spotlightDropoffStart;    // Start of drop-off gradient, in radians. Starts at center, ends at edge. Controls the rate at which light intensity decreases from the center to the edge of the spot light beam
    float spotlightDropoffEnd;      // End of drop-off gradient, in radians. Both start and end drop-off angles have to be smaller than radius else sharp edge
    uint unk0;                      // 14336 (power of 2)
    char gap4[41];                  // empty
    char field_50;                  // Only found 0's so far
    char unk1[2];                   // Only found 0's so far
  } mapobject_new_light_defs[];

==MDDL==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
 struct {
 /*0x00*/ float _0x0;                              // can apparently be overwritten by groups,
                                                   // potentially inside MOBA's unknown reuse blob.
                                                   // used in determining locs based on moba vertices.
                                                   // minimum triangle area?
 /*0x04*/ uint16_t layerCount;
 /*0x06*/ Layer detailDoodadLayers[layerCount];    // global
 /*0x??*/ GroupData groupData[until-end-of-chunk]; // per WMO group
 /*0x??*/
 };

 struct Layer {
 /*0x00*/ uint8_t density?;       // if density? < (v14 >> 20) - 24 * (v14 / 25165824), have no
                                  // doodad on location, where v14 is a random number based on 
                                  // a seed based on the vertex index
 
 /*0x01*/ uint8_t detailDoodadsCount;
 /*0x02*/ DetailDoodad detailDoodads[detailDoodadsCount];
 /*0x??*/
 };
 struct DetailDoodad {
 /*0x00*/ {{Type/foreign_key|table=GroundEffectDoodad}} doodad;
 /*0x04*/ uint8_t weight;                          // not required to accumulate to something specific.
 /*0x05*/
 };

 struct GroupData {
 /*0x00*/ uint16_t groupIndex;
 /*0x02*/ uint32_t dataSize;
 /*0x06*/ char data[dataSize];     // interpreted when parsing groups as per parse_group_data() below
 /*0x??*/
 };
 
 void parse_group_data() {
 restart_layer:
   uint16_t layer_index = read_uint16_t();  // index into detailDoodadLayers
   if (layer_index == 0xFFFF) {
     return;
   }
 
 restart_batch:
   uint16_t batch_index = read_uint16_t();    // batch as in MOBA
   if (batch_index == 0xFFFF) {
     goto restart_layer;
   }
   else if (batch_index & 0x8000) {
     // roll for locs of all loc_ranges of this batch
     goto restart_batch;
   }
 
   int locrange_index = 0;                    // loc as in triangles that satisfy some criteria
                                              // for all batches in batch order
 restart_locrange_index_part:
   uint8_t locrange_index_part = read_uint8_t();
   locrange_index += locrange_index_part & 0x7F;
   if (locrange_index_part == 0xFF) {
     goto restart_batch;
   }
   else if (locrange_index_part == 0x7F) {   // RLE for an integer?!
     goto restart_locrange_index_part;
   }
   bool single_loc = locrange_index_part & 0x80;
 
   int loc = 0;
 restart_loc_part:
   uint8_t loc_part = read_uint8_t();
   loc += loc_part;
   if (loc_part == 0xFF) {
     goto restart_locrange_index_part; /// not in front of that, i.e. resetting to 0?!
   }
   else if (loc_part == 0xFE) {   // RLE for an integer?!, yes different sentinel!
     goto restart_loc_part;
   }
 
   // take loc range for loc_range_index
   // if single_loc, roll once for locs[loc_range.begin + loc]
   // else, roll for locs[loc_range.begin + 0...loc]
 
   goto restart_batch;
 }

= WMO group file =

WMO group files contain the actual polygon soup for a particular section of the entire [[WMO]].

Every group file has one top-level [[WMO#MOGP_chunk|MOGP]] chunk, that has a 68-byte header followed by more subchunks. So it can be effectively treated as a file with a header at 0x14 and chunks starting at 0x58. 

The subchunks are not always present. Some are fixed and needed while others are only checked for if some flags in the header are set. The chunks '''need''' to be in the right order if you want WoW to read it.

The following chunks are always present in the following order:
*[[WMO#MOGP_chunk|MOGP]]
*[[WMO#MOGX_chunk|MOGX]]
*[[WMO#MOPY_chunk|MOPY]]
*[[WMO#MPY2_chunk|MPY2]]
*[[WMO#MOVI_chunk|MOVI]]
*[[WMO#MOVT_chunk|MOVT]]
*[[WMO#MONR_chunk|MONR]]
*[[WMO#MOTV_chunk|MOTV]]
*[[WMO#MOBA_chunk|MOBA]]
*[[WMO#MOQG_chunk|MOQG]]

These chunks are only present if a flag in the header is set. See the list below for the flags.
*Cataclysm introduced a new optional MOBS chunk, I guess it's related to [[WMO#MOBA_chunk|MOBA]]. ---[[User:Bananenbrot|Bananenbrot]], 12-18-2010
*[[WMO#MOLR_chunk|MOLR]]
*[[WMO#MODR_chunk|MODR]]
*[[WMO#MOBN_chunk|MOBN]]
*[[WMO#MOBR_chunk|MOBR]]
*MPBV
*MPBP
*MPBI
*MPBG
*[[WMO#MOCV_chunk|MOCV]]
*[[WMO#MLIQ_chunk|MLIQ]]
*[[WMO#MORI|MORI]]
*[[WMO#MORB|MORB]]
* [[WMO#MOTV_chunk|MOTV]] 2
* [[WMO#MOCV_chunk|MOCV]] 2

== MOGP chunk ==

'''IMPORTANT''': This chunk contains all other chunks! The following variables are a header only. The MOGP chunk size will be way more than the header variables!

 struct {
 /*0x00*/  uint32_t groupName;               // offset into [[#MOGN_chunk|MOGN]]
 /*0x04*/  uint32_t descriptiveGroupName;    // offset into [[#MOGN_chunk|MOGN]]
 /*0x08*/  uint32_t flags;                   // see below
 /*0x0C*/  {{Template:Type|CAaBox}} boundingBox;              // as with flags, same as in corresponding [[#MOGI_chunk|MOGI]] entry
 
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
           uint32_t portalStart;             // index into [[#MOPR_chunk|MOPR]]
           uint32_t portalCount;             // number of [[#MOPR_chunk|MOPR]] items used after portalStart
 #else
 /*0x24*/  uint16_t portalStart;             // index into [[#MOPR_chunk|MOPR]]
 /*0x26*/  uint16_t portalCount;             // number of [[#MOPR_chunk|MOPR]] items used after portalStart
 #endif
 
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592}} 
 /*0x28*/  uint16_t transBatchCount;
 /*0x2A*/  uint16_t intBatchCount;
 /*0x2C*/  uint16_t extBatchCount;
 /*0x2E*/  uint16_t padding_or_batch_type_d; // probably padding, but might be data?
 #endif 
 
 /*0x30*/  uint8_t fogIds[4];                // ids in [[#MFOG_chunk|MFOG]]
 /*0x34*/  uint32_t groupLiquid;             // see below in the [[#MLIQ_chunk|MLIQ]] chunk
 
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
           SMOGxBatch intBatch[4];
           SMOGxBatch extBatch[4];
 #endif
 
 /*0x38*/  {{Template:Type/foreign_key|table=WMOAreaTable|column=m_WMOGroupID}} uniqueID;
 
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
           uint8_t padding[8];
 #else
           /*0x3C*/  SMOGroupFlags2 flags2;
           #if {{Template:Sandbox/VersionRange|max_expansionlevel=9|max_build=9.1.5}} 
                    /*0x40*/  uint32_t unk;             // UNUSED: 20740
           #else
                    /*0x40*/  int16_t parentOrFirstChildSplitGroupIndex; //See [[WMO#Split_Groups|Split Groups]]
                    /*0x42*/  int16_t nextSplitChildGroupIndex;                    
           #endif
 #endif
 } map_object_group_header;
 // remaining chunks follow
 
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
 struct SMOGxBatch
 {
   uint16_t vertStart;
   uint16_t vertCount;
   uint16_t batchStart;
   uint16_t batchCount;
 };
 #endif

The fields referenced from the [[WMO#MOPR_chunk|MOPR]] chunk indicate portals leading out of the [[WMO]] group in question.

For the "Number of batches" fields, <code>transBatchCount</code> + <code>intBatchCount</code> + <code>extBatchCount</code> == the total number of batches in the [[WMO]] group (in the [[#MOBA_chunk|MOBA]] chunk). This might be some kind of LOD thing, or just separating the batches into different types/groups…?

Flags: always contain more information than flags in [[#MOGI_chunk|MOGI]]. I suppose [[#MOGI_chunk|MOGI]] only deals with topology/culling, while flags here also include rendering info.

===group flags===
 '''Flag		Meaning'''
 0x1		Has BSP tree ([[#MOBN_chunk|MOBN]] and [[#MOBR_chunk|MOBR]] chunk).
 0x2		Has light map ([[#MOLM|MOLM]], [[#MOLD|MOLD]]). (UNUSED: 20740) possibly: subtract mohd.color in mocv fixing 
 0x4 		Has vertex colors ([[#MOCV_chunk|MOCV]] chunk).
 0x8 		SMOGroup::EXTERIOR -- Outdoor - also influences how doodads are culled. If camera is AABB present in a group with this flag, and not present in any group with SMOGroup::INTERIOR, render all exteriors
 0x10		(UNUSED: 20740)
 0x20		(UNUSED: 20740)
 0x40		SMOGroup::EXTERIOR_LIT -- "Do not use local diffuse lightning". Applicable for both doodads from this wmo group(color from MODD) and water(CWorldView::GatherMapObjDefGroupLiquids). If group has SMOGroup::INTERIOR flag and this flag then exterior lighting is used for the group)
 0x80 		SMOGroup::UNREACHABLE
 0x100          Show exterior sky in interior WMO group (Used for interiors of city in stratholme_past.wmo)
 0x200 		Has lights ([[#MOLR_chunk|MOLR]] chunk)
 0x400		<= Cataclysm: Has [[#MPBV|MPBV]], [[#MPBP|MPBP]], [[#MPBI|MPBI]], [[#MPBG|MPBG]] chunks, neither 0.5.5, 3.3.5a nor Cataclysm alpha actually use them though, but just skips them. Legion+(?): SMOGroup::LOD: Also load for LoD != 0 (_lod* groups). Seems to disable shadow casting when on
 0x800 		Has doodads ([[#MODR_chunk|MODR]] chunk)
 0x1000		SMOGroup::LIQUIDSURFACE -- Has water ([[#MLIQ_chunk|MLIQ]] chunk)
 0x2000		SMOGroup::INTERIOR -- Indoor
 0x4000		(UNUSED: 20740)
 0x8000          QueryMountAllowed in pre-WotLK
 0x10000         SMOGroup::ALWAYSDRAW -- clear 0x8 after CMapObjGroup::Create() in MOGP and MOGI
 0x20000		(UNUSED: 20740) Has [[WMO#MORI|MORI]] and [[WMO#MORB|MORB]] chunks.
 0x40000		Show skybox -- automatically unset if MOSB not present.
 0x80000		is_not_water_but_ocean, LiquidType related, see below in the MLIQ chunk.
 0x100000
 0x200000	IsMountAllowed
 0x400000	(UNUSED: 20740)
 0x800000
 0x1000000	SMOGroup::CVERTS2: Has the second [[#MOCV_chunk|MOCV]] chunks: If the flag 0x4 isn't set this is the only MOCV chunk in the group. Whether the flag 0x4 is set or not: only the alpha values from this chunk are used (to blend the textures). '''[[#CMapObjGroup::FixColorVertexAlpha|FixColorVertexAlpha]] must not be used on this chunk !'''
 0x2000000	SMOGroup::TVERTS2: Has two [[#MOTV_chunk|MOTV]] chunks: Just add two.
 0x4000000	SMOGroup::ANTIPORTAL: Just call CMapObjGroup::CreateOccluders() independent of groupname being "antiportal". requires intBatchCount == 0, extBatchCount == 0, UNREACHABLE.
 0x8000000	unk. requires intBatchCount == 0, extBatchCount == 0, UNREACHABLE. When set seems to disable rendering of batches, but still renders doodads
 0x10000000	(UNUSED: 20740)
 0x20000000	{{Template:Unverified|>> 20740}} SMOGroup::EXTERIOR_CULL
 0x40000000	SMOGroup::TVERTS3: Has three [[#MOTV_chunk|MOTV]] chunks, eg. for [[#MOMT_chunk|MOMT]] with shader 18.
 0x80000000     Seen in world/wmo/kultiras/human/8hu_warfronts_armory_v2_000.wmo
 vv flags2
 0x01????????   canCutTerrain
 0x30000000	SMOGroup::depSHADOWMAPGEN | SMOGroup::depSHADOWMAPGEN_DEPTH as per "(m_groupFlags & (SMOGroup::depSHADOWMAPGEN | SMOGroup::depSHADOWMAPGEN_DEPTH)) == 0" and *(_DWORD *)(a1 + 36) & 0x30000000. yes, this clashes with EXTERIOR_CULL, but that's in the same version. weird.

===group flags 2===

 struct SMOGroupFlags2 {
     //0x1
     uint32_t canCutTerrain : 1;   = 1,        // {{Template:Sandbox/VersionRange|min_expansionlevel=5}} has [[#MOPL_.28WoD.28.3F.29.2B.29|portal planes]] to cut
     //0x2
     uint32_t unk2 : 1;
     //0x4
     uint32_t unk4 : 1;
     //0x8
     uint32_t unk8 : 1;
     //0x10
     uint32_t unk0x10 : 1;
     //0x20
     uint32_t unk0x20 : 1;
     //0x40
     uint32_t isSplitGroupParent : 1;          // {{Template:Sandbox/VersionRange|min_expansionlevel=9}} since around 9.2.0
     //0x80
     uint32_t isSplitGroupChild : 1;           // {{Template:Sandbox/VersionRange|min_expansionlevel=9}} since around 9.2.0
     //0x100
     uint32_t FLAGS2_ATTACHMENT_MESH : 1;      // ≥ 9.2, ≤ 11.0
 };

=== "antiportal" ===

If a group wmo is named "antiportal", CMapObjGroup::CreateOccluders() is called and group flags 0x4000000 and 0x80 are set automatically in both, MOGP and MOGI. Also, the BSP tree is cleared and batch_count[interior] and [exterior] is set to 0. If flags & 0x4000000 is set, just CMapObjGroup::CreateOccluders() is called, without setting flags or clearing bsp.

m_vertices is content of MOVT

 void CMapObjGroup::CreateOccluders()
 {
   for ( unsigned int mopy_index (0), movi_index (0)
       ; mopy_index < this->mopy_count
       ; ++mopy_index, ++movi_index
       ) 
   {
     {{Template:Type|C3Vector}}* points[3] = 
       { &this->m_vertices[this->movi[3*mopy_index + 0]]
       , &this->m_vertices[this->movi[3*mopy_index + 1]]
       , &this->m_vertices[this->movi[3*mopy_index + 2]]
       };
 
     float avg ((points[0]->z + points[1]->z + points[2]->z) / 3.0); 
 
     unsigned int two_points[2];
     unsigned int two_points_index (0);
 
     for (unsigned int i (0); i < 3; ++i)
     {
       if (points[i]->z > avg)
       {
         two_points[two_points_index++] = i;
       }
     }
 
     if (two_points_index > 1)
     {
       CMapObjOccluder* occluder (CMapObj::AllocOccluder());
       occluder->p1 = points[two_points[0]];
       occluder->p2 = points[two_points[1]];
 
       append (this->occluders, occluder);
     }
   }
 }

===Split Groups===

First spotted in 9.2.0 Split groups is a new mechanic, which makes parent-child relation between Group WMOs. 
There is "Parent Split" Group and "Child Split" Groups, that be belong to Parent. From the way the data is organized right now, it's not possible for a "Child Split" Group to be a "Parent Split" for other Group WMOs. So this graph can have only one level of depth.


Essentially "Child Split" Group do not have Portals leading to them. Instead all portals leading in and out are connected to "Parent Split". So portal culling algorithm needs to render "Child Split" Groups, when "Parent Split" passes Portal culling test.


How it works:


if <code>SMOGroupFlags2.isSplitGroupParent</code> flag is set, the group is marked as "Parent Split" Group. In this case <code>MOGP.parentOrFirstChildSplitGroupIndex</code> is a group index of first "Split Child" Group and <code>MOGP.nextSplitChildGroupIndex</code> is set to -1 (at least as of current observations).


For "Split Child" <code>SMOGroupFlags2.isSplitGroupParent</code> is not set and <code>SMOGroupFlags2.isSplitGroupChild</code> is set instead. In this case <code>MOGP.parentOrFirstChildSplitGroupIndex</code> is index of "Parent Split" Group to which this "Split Child" belongs to, and <code>MOGP.nextSplitChildGroupIndex</code> is group index of next "Split Child" in the list.


So to get all "Split Child" for particular "Parent Split", one would need to take <code>MOGP.parentOrFirstChildSplitGroupIndex</code> of "Parent Split" and go through chain of referencing <code>MOGP.nextSplitChildGroupIndex</code>, while <code>MOGP.nextSplitChildGroupIndex</code> is not -1


For earliest known example of WMO with such mechanic, see FDID 4217818

== MOGX chunk ==
{{Template:SectionBox/VersionRange|min_expansionlevel=10|min_build=10.0.0.46181}}

  uint32_t queryFaceStart;

Contains one single value. It used in combination with [[WMO#MOQG_chunk|MOQG]] chunk to determine per polygon groundType

== MOPY chunk ==

*'''Material info for triangles, two bytes per triangle. So size of this chunk in bytes is twice the number of triangles in the WMO group.'''

 struct SMOPoly
 {
   struct
   {
     /*0x01*/ uint8_t F_UNK_0x01: 1;
     /*0x02*/ uint8_t F_NOCAMCOLLIDE : 1;
     /*0x04*/ uint8_t F_DETAIL : 1;
     /*0x08*/ uint8_t F_COLLISION : 1; // Turns off rendering of water ripple effects. May also do more. Should be used for ghost material triangles.
     /*0x10*/ uint8_t F_HINT : 1;
     /*0x20*/ uint8_t F_RENDER : 1;
     /*0x40*/ uint8_t F_CULL_OBJECTS : 1; // tested on 1.12, flag enables/disables game object culling. might be wrong, previously UNK_0x40
     /*0x80*/ uint8_t F_COLLIDE_HIT : 1;
 
     bool isTransFace() { return F_UNK_0x01 && (F_DETAIL || F_RENDER); } // triangles flagged as TRANSITION.  These triangles blend lighting from exterior to interior
     bool isColor() { return !F_COLLISION; }
     bool isRenderFace() { return F_RENDER && !F_DETAIL; }
     bool isCollidable() { return F_COLLISION || isRenderFace(); }
   } flags;
 
 #if version {{Template:Sandbox/VersionRange|max_expansionlevel=1|max_exclusive=1}} 
   uint8_t lightmapTex;           // index into [[#MOLD|MOLD]]
 #endif
   uint8_t material_id;           // index into [[#MOMT_chunk|MOMT]], 0xff for collision faces
 #if version {{Template:Sandbox/VersionRange|max_expansionlevel=1|max_exclusive=1}} 
   uint8_t padding;
 #endif
 } polyList[];

0xFF is used for collision-only triangles. They aren't rendered but have collision. Problem with it: WoW seems to cast and reflect light on them. Its a bug in the engine. --[[User:Schlumpf|schlumpf_]] 20:40, 7 June 2009 (CEST)

Triangles stored here are more-or-less pre-sorted by texture, so it's ok to draw them sequentially.

== MPY2 chunk ==
{{Template:SectionBox/VersionRange|min_expansionlevel=10|min_build=10.0.0.46181}}
*'''Material info for triangles, 4 bytes per triangle. So size of this chunk in bytes is four times the number of triangles in the WMO group.'''

Replacement for MOPY chunk, purpose - holding multiple materials information.

 struct MPY2Poly
 {
    uint16_t flags;
    uint16_t materialId;
 };

== MOVI chunk ==

''('''M'''ap'''O'''bject '''V'''ertex '''I'''ndices)''

The group's vertex indices from the group's vertex list (MOVT, MONR, MOTV) to form triangles. 

 uint16[] Indices;

Three indices form a single triangle. Therefore, the number of indices should be divisible by 3. 

All triangles are set in a right-handed coordinate system, which means the order of vertices is anti-clockwise to make a front-face triangle (positive area). When used in a left-handed coordinate system, the 2nd and 3rd vertex indices of each triangle have to be swapped, otherwise these triangles form a negative area, and with back-side culling enabled, get culled. When incorrectly set, a 3D renderer will make textured meshes look "inside-out".

== MOVX chunk ==
Possible replacement for MOVI chunk but allowing for larger indices (uint vs MOVI's ushort)? Spotted in 9.0, but might have existed for a while.

== MOVT chunk ==

*'''Vertices chunk.''', count = size / (sizeof(float) * 3). 3 floats per vertex, the coordinates are in (X,Z,-Y) order. It's likely that [[WMO]]s and models ([[M2]]s) were created in a coordinate system with the Z axis pointing up and the Y axis into the screen, whereas in OpenGL, the coordinate system used in WoWmapview the Z axis points toward the viewer and the Y axis points up. Hence the juggling around with coordinates.

 C3Vector vertexList[];

== MONR chunk ==

*'''Normals.''' count = size / (sizeof(float) * 3). 3 floats per vertex normal, in (X,Z,-Y) order.

 C3Vector normalList[];

== MOTV chunk ==

*'''Texture coordinates, 2 floats per vertex in (X,Y) order.''' The values usually range from 0.0 to 1.0, but it's ok to have coordinates out of that range. Vertices, normals and texture coordinates are in corresponding order, of course. Not present in [[WMO#.22antiportal.22|antiportal]] WMO groups.

 C2Vector textureVertexList[];    // ranging [0, 1], can be outside that range though and will be normalised.

''Client loads multiple MOTV chunks into an array but only keeps the count of the last one. This behavior is different to all other chunk types read. The array has 3 entries, after that the client will overwrite its data structures, starting with the MOTV_Counter field itself. (checked with client 29297, client 30918 still has this severe bug)''

 else                                                      // MOTV
 {
   this->MOTV[this->MOTV_Counter++] = Chuck->Payload;      // careful, unchecked array access
   this->MOTV_Count = Chunk->Length >> 3;
 }

==MOLV==
{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494|note=Only used in v14}}
This chunk is referenced by [[#MOPY_chunk|MOPY]] index with 3 entries per SMOPoly.
 C2Vector lightmapVertexList[];

==MOIN==
{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494|note=Only used in v14}}
 uint16_t indexList[];

It's most of the time only a list incrementing from <code>0</code> to <code>nFaces * 3</code> or less, not always up to <code>nPolygons</code> (calculated with [[#MOPY_chunk|MOPY]]).

Unlike in {{Template:Sandbox/VersionRange|min_expansionlevel=1}} where the faces indices ([[#MOVI_chunk|MOVI]]) point to a vertex in [[#MOVT_chunk|MOVT]], here there are exactly <code>nFaces * 3</code> vertices in [[#MOVT_chunk|MOVT]], and the client just read them straightforward. If you want to read them, just make <code>nPolygons</code> faces going incrementing, like <code>(0, 1, 2), (3, 4, 5), …</code> --Gamhea 15:44, 10 March 2013 (UTC)

== MOBA chunk ==
 
*'''Render batches. Records of 24 bytes.'''
 
 struct SMOBatch
 {
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
   uint8_t lightMap;                                 // index into [[#MOLM|MOLM]]
   uint8_t texture;                                  // index into [[#MOMT_chunk|MOMT]]
 #endif
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=7|max_exclusive=1}}
   /*0x00*/ int16_t bx, by, bz;                      // a bounding box for culling, see "unknown_box" below
   /*0x06*/ int16_t tx, ty, tz;
 #else
   /*0x00*/ uint8_t unknown[0xA];
   /*0x0A*/ uint16_t material_id_large;              // used if flag_use_uint16_t_material is set.
 #endif
 #if {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494}} 
   uint16_t startIndex;                              // index of the first face index used in [[#MOVI_chunk|MOVI]]
 #else
   /*0x0C*/ uint32_t startIndex;                     // index of the first face index used in [[#MOVI_chunk|MOVI]]
 #endif
   /*0x10*/ uint16_t count;                          // number of [[#MOVI_chunk|MOVI]] indices used
   /*0x12*/ uint16_t minIndex;                       // index of the first vertex used in [[#MOVT_chunk|MOVT]]
   /*0x14*/ uint16_t maxIndex;                       // index of the last vertex used (batch includes this one)
   /*0x16*/ uint8_t flag_unknown_1 : 1;
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=7}}
   /*0x16*/ uint8_t flag_use_material_id_large : 1;  // instead of material_id use material_id_large
 #endif
                                                     // F_RENDERED = 0xf0, so probably upper nibble isn't unused
 
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592}} 
   /*0x17*/ uint8_t material_id;                     // index in [[#MOMT_chunk|MOMT]]
 #else
   uint8_t padding;
 #endif
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.6.0.3592|max_expansionlevel=1|max_exclusive=1}}  
   uint8_t unknown[8];                               // always 0 filled
 #endif
 } batchList[];
 
Batches are groups of faces with the same material ID in root's MOMT, and they're used to accelerate rendering. Note that the client doesn't use them in the same way while rendering in D3D or OpenGL (only D3D uses all batches information). The vertex buffer containing vertices from <code>minIndex</code> to <code>maxIndex</code> can contain vertices that aren't used by the batch. On the other hand, if one of the faces used need a vertex, it has to be in the buffer. <del>Concerning the byte at 0x16, as a material ID is coded on a uint8, I guess it is completely unused.</del>
--[[User:Gamhea|Gamhea]] 12:23, 29 July 2013 (UTC)

===unknown_box===
This is a very low resolution bounding box of the contained vertices. The client appears to be using them to do batch-level culling, so if they are set incorrectly, the batch may be randomly disappearing. According to [[User:Adspartan|Adspartan]] ([[User talk:Adspartan|talk]]), the box can be calculated by just iterating over all vertices contained (by following <code>minIndex</code> and <code>maxIndex</code> to [[#MOVT|MOVT]] and taking the minimum/maximum of those. They should probably be rounded away from zero instead of being truncated on conversion to <code>int16_t</code>. 

{{Template:SectionBox|This section only applies to version {{Template:Sandbox/PrettyVersion|expansionlevel=0|build=0.5.3.3368}}}}
In the 0.5.3 Alpha this box is used for batch-level culling. The values are converted to a {{Template:Type|CAaBox}} inside <code>CMapObj::CullBatch</code>, by being directly cast to floats, this box is then passed to <code>CWorldScene::FrustumCull</code> for rendering.

{{Template:SectionBox/VersionRange|min_expansionlevel=7}}

<code>unknown_box</code> seems no longer used (and nulled). Instead, <code>flag_use_material_id_large</code> can be set to use <code>material_id_large</code> which was the last of <code>unknown_box</code>'s fields. This means that when "retroporting" files, <code>unknown_box</code>'s values need to be calculated (by building minimum and maximum from the corresponding vertices) and <code>material_id</code> should be set, if it can fit a <code>uint8_t</code>. --based on [[User:Rangorn|Rangorn]] ([[User talk:Rangorn|talk]])

{{Template:SectionBox/VersionRange|min_expansionlevel=9}}
Previous versions of the game are forgiving to cases when there are OxFF collision only invisible triangles between minIndex and maxIndex. In Shadowlands having these inside the batches will result into various rendering glitches on some machines that support bindless texturing.

== MOQG chunk ==
{{Template:SectionBox/VersionRange|min_expansionlevel=10|min_build=10.0.0.46181}}

  uint32_t queryFace[];

Each value of <code>queryFace</code> is <code>groundType</code>. 

It's used only if for polygon represented by <code>polygonIndex</code> the <code>0x100</code> flag is set in [[WMO#MPY2_chunk|MPY2]] chunk.

The proper index into this array is calculated as <code>polygonIndex - queryFaceStart</code>, where <code>queryFaceStart</code> is from [[WMO#MOGX_chunk|MOGX]] chunk

==  MOLR chunk ==

*'''Light references, one 16-bit integer per light reference.'''

 uint16_t lightRefList[];

This is basically a list of lights used in this [[WMO]] group, the numbers are indices into the [[WMO]] root file's [[WMO#MOLT_chunk|MOLT]] table.

For some [[WMO]] groups there is a large number of lights specified here, more than what a typical video card will handle at once. I wonder how they do lighting properly. Currently, I just turn on the first GL_MAX_LIGHTS and hope for the best. :(

10.1.5: MOLR goes by the name <code>m_doodadOverrideLightRefList</code>. Thus MOLR = <code>MapOverrideLightRef</code>.

== MODR chunk ==

*'''Doodad references, one 16-bit integer per doodad.'''

 uint16_t doodadRefList[];

The numbers are indices into the doodad instance table ([[#MODD_chunk|MODD]] chunk) of the [[WMO]] root file. These have to be filtered to the doodad set being used in any given [[WMO]] instance.

== MOBN chunk ==

*'''Nodes of the BSP tree, used for collision (along with bounding boxes ?). Array of t_BSP_NODE. / CAaBspNode.''' 0x10 bytes.

 enum Flags
 {
   Flag_XAxis = 0x0,
   Flag_YAxis = 0x1,
   Flag_ZAxis = 0x2,
   Flag_AxisMask = 0x3,
   Flag_Leaf = 0x4,
   Flag_NoChild = 0xFFFF,
 };
 
 struct CAaBspNode
 {	
   uint16_t flags;        // See above enum. 4: leaf, 0 for YZ-plane, 1 for XZ-plane, 2 for XY-plane
   int16_t negChild;      // index of bsp child node (right in this array)
   int16_t posChild;
   uint16_t nFaces;       // num of triangle faces in [[WMO#MOBR_chunk|MOBR]]
   uint32_t faceStart;    // index of the first triangle index(in [[WMO#MOBR_chunk|MOBR]])
   float planeDist;
 };

planetype might be 0 for YZ-plane, 1 for XZ-plane, 2 for XY-plane, 4 for BSP leaf. fDist is where split plane locates based on planetype, ex, you have a planetype 0 and fDist 15, so the split plane is located at offset ( 15, 0, 0 ) with Normal as ( 1, 0, 0 ), I think the offset is relative to current node's bounding box center. The BSP root ( ie. node 0 )'s bounding box is the WMO's boundingbox, then you subdivide it with plane and fdist, then you got two children with two bounding box, and so on. you got the whole BSP tree. As the bsp leaf might overlapping the dividing plane, i think you might have two same face exist on two different bsp leaf. I'll make further tests to prove this. --[[User talk:mobius|mobius]].

The biggest leaf in terms of number of faces in 3.3.5 contains more than 2100 faces (some ice giant in the Storm Peaks), so it's not advised to use more. (While I haven't investigated properly, there might be a limit at 8192 in 6.0.1.18179 --[[User:Schlumpf|Schlumpf]] ([[User talk:Schlumpf|talk]]) 11:18, 3 January 2016 (UTC))

fDist is relative to point (0,0,0) of whole WMO. children[0] is child on negative side of dividing plane, children[1] is on positive side. --[[User:Deamon|Deamon]] ([[User talk:Deamon|talk]]) 10:01, 15 January 2016 (UTC)


 #define epsilon 0.01F
 void MergeBox(CVect3 (&result)[2], float  *box1, float  *box2)'''
 {
  result[0][0] = box1[0];
  result[0][1] = box1[1];
  result[0][2] = box1[2];
  result[1][0] = box2[0];
  result[1][1] = box2[1];
  result[1][2] = box2[2];
 }
 void AjustDelta(CVect3 (&src)[2], float *dst, float coef)'''
 {
  float d1 = (src[1][0]- src[0][0]) * coef;// delta x
  float d2 = (src[1][1]- src[0][1]) * coef;// delta y
  float d3 = (src[1][2]- src[0][2]) * coef;// delta z
  dst[1] = d1 + src[0][1];
  dst[0] = d2 + src[0][0];
  dst[2] = d3 + src[0][2];
 }
 void TraverseBsp(int iNode, CVect3 (&pEyes)[2] , CVect3 (&pBox)[2],void *(pAction)(T_BSP_NODE *,void *param),void *param)'''
  {
  int plane;
  float eyesmin_boxmin;
  float boxmax_eyesmax;
  float eyesmin_fdist;
  float eyes_max_fdist;
  float eyesmin_div_deltadist;
  CVect3 tBox1[2];
  CVect3 tBox2[2];
  CVect3 newEyes[2];
  CVect3 ajusted;
  T_BSP_NODE *pNode = &m_tNode[iNode];
  if ( pNode)
  {
   if (pNode->planetype & 4 )
   {
    if(pAction == 0)
    {
     RenderGeometry(GetEngine3DInstance(),pNode);
     return;
    }
    else
    {
     pAction(pNode,param);
    }
   }
   plane =pNode->planetype  & 3;
   eyesmin_boxmin = pEyes[0][plane] - pBox[0][plane];
   if ( ( -epsilon < eyesmin_boxmin) | (-epsilon == eyesmin_boxmin) || (pEyes[1][plane]- pBox[0][plane])  >= -epsilon )
   {
    boxmax_eyesmax = pBox[1][plane] - pEyes[1][plane];
    if ( (epsilon < boxmax_eyesmax) | (epsilon == boxmax_eyesmax) || (pBox[1][plane] -  pEyes[0][plane]) >= epsilon )
    {
     memmove(tBox1,pBox,sizeof(pBox));
     tBox1[0][plane] = pNode->fDist;
     memmove(tBox2,pBox,sizeof(pBox));
     tBox2[1][plane] = pNode->fDist;
     eyesmin_fdist = pEyes[0][plane] - pNode->fDist;
     eyes_max_fdist = (pEyes[1][plane]) - pNode->fDist;
     if ( eyesmin_fdist >= -epsilon && eyesmin_fdist <= epsilon|| (eyes_max_fdist >= -epsilon) && eyes_max_fdist <= epsilon )
     {
      if ( pNode->children[1] != (short)-1 ) TraverseBsp(pNode->children[1],  pEyes,  tBox1,pAction,param);
      if ( pNode->children[0] != (short)-1 ) TraverseBsp(pNode->children[0] , pEyes, tBox2,pAction,param);
      return;
     }
     if ( eyesmin_fdist > epsilon && eyes_max_fdist < epsilon)
     {
       if ( pNode->children[1] != (short)-1 ) TraverseBsp(pNode->children[1], pEyes, tBox1,pAction,param);
       return;
     }
     if ( eyesmin_fdist < -epsilon && eyes_max_fdist < -epsilon)
     {
       if ( pNode->children[0] != (short)-1 ) TraverseBsp(pNode->children[0] , pEyes, tBox2,pAction,param);
       return;
     }
     eyesmin_div_deltadist = (float)(eyesmin_fdist / (eyesmin_fdist - eyes_max_fdist));
     AjustDelta(pEyes, ajusted, eyesmin_div_deltadist);
     if ( eyesmin_fdist <= 0.0 )
     {
      if ( pNode->children[0]  != (short)-1 )
      {
       MergeBox(newEyes, &pEyes[0][0], ajusted);
       TraverseBsp(pNode->children[0] , newEyes, tBox2,pAction,param);
      }
      if (pNode->children[1]  != (short)-1 )
      {
       MergeBox(newEyes, ajusted, &pEyes[1][0]);
       TraverseBsp(pNode->children[1] , newEyes, tBox1,pAction,param);
      }
     }
     else
     {
      if ( pNode->children[1]  != (short)-1 )
      {
       MergeBox(newEyes, &pEyes[0][0], ajusted);
       TraverseBsp(pNode->children[1] , newEyes, tBox1,pAction,param);
      }
      if (pNode->children[0]  != (short)-1 )
      {
       MergeBox(newEyes, ajusted, &pEyes[1][0]);
       TraverseBsp(pNode->children[0] , newEyes, tBox2,pAction,param);
      }
     }
    }
   }
  }
 }

 CheckFromEyes(CVect3 (&pEyes)[2],void *(pAction)(T_BSP_NODE *,void *param),void *param )
 {
 /*CVect3 eyes[2];
 instance_mat.invert();
 eyes[0] = _fixCoordSystemInv((instance_mat*p->m_pCameraViewport->GetCameraTarget())+CVect3(0,-10,0) );
 eyes[1] = _fixCoordSystemInv((instance_mat*p->m_pCameraViewport->GetCameraTarget())+CVect3(0,60,0) ); 
  // make vector down
 */
 /* eyes[0] = CVect3(-1.474797e+001F, -1.195053e+001F,  5.416779e+000F); // Debug absolute position from WP  Azaroth 1164,58,-10645.83
 eyes[1] = CVect3(-1.474797e+001F, -1.195053e+001F, -1.754583e+003F);
 */
 TraverseBsp(0,pEyes,m_bbox,pAction);
 }

There was a common misconception that this BSP is only used for collision purposes. In fact, it is also used by the client to determine if you are currently inside the WMO group (empirically observed). 
So, if any faces are missing from the BSP, indoor groups will be culled on approaching them. For outdoor groups they seem to just not have collision.
This, in turn, debunks another common misconception that WMOs use two separate collision systems - the BSP one and unbatched geometry marked with material ID 0xFF and collision flag in MOPY.
In practice, ommiting those collision faces from BSP also yields incorrect results, resulting into the absence of collision for those faces as well as culling issues for indoor groups. --[[User:Skarn|Skarn]] ([[User talk:Skarn|talk]]) 15:56, 17 April 2022 (CEST)

An object could have has 2 collision system. The first one is encoded in a simplified Geometry (when MOPY. MaterialID=0xFF) the second one is encoded in T_BSP_NODE.
Some object has collision method 1 only, some other uses method 2 only. Some object have both collision systems (some polygons are missing in the BSP but are present in the simplified geometry). how  to use these 2 system remains unclear. 

For the time being, I check first the simplified geometry, and then if there is no collision, I apply a second pass using the BSP. It is sub-optimum, but it seems to work.
Probably there is somewhere a flag telling us with which method we should use for the object.

The code attached seems to work fine for BSP method--[[peter-pan|peter-pan]].

== MOBR chunk ==

*'''Face indices''' for CAaBsp ([[#MOBN_chunk|MOBN]]). Unsigned shorts.
*'''Triangle indices (in [[WMO#MOVI_chunk|MOVI]] which define triangles) to describe polygon planes defined by [[WMO#MOBN_chunk|MOBN]] BSP nodes.'''

 uint16_t nodeFaceIndices[];

Example code required to get an actual indices array from MOBR array:
 var bpsIndicies = new Array(mobr.length*3);
 for (var i = 0; i < mobr.length; i++) {
     bpsIndices[i*3 + 0] = movi[3*mobr[i]+0];
     bpsIndices[i*3 + 1] = movi[3*mobr[i]+1];
     bpsIndices[i*3 + 2] = movi[3*mobr[i]+2];
 }

Example code to get indices into MOVT for triangles, referenced from BSP node definition:
 for (var triangleInd = node.firstFace; triangleInd<node.firstFace+node.numFaces; triangleInd++) {
     //3 vertices per triangle
     movt[bpsIndices[3*triangleInd + 0]]
     movt[bpsIndices[3*triangleInd + 1]]
     movt[bpsIndices[3*triangleInd + 2]]
 }

== MOCV chunk ==

*'''Vertex colors, 4 bytes per vertex (BGRA), for [[WMO]] groups using indoor lighting.''' 

 CImVector colorVertexList[];

I don't know if this is supposed to work together with, or replace, the lights referenced in [[WMO#MOLR_chunk|MOLR]]. But it sure is the only way for the ground around the goblin smelting pot to turn red in the Deadmines. (but some corridors are, in turn, too dark - how the hell does lighting work anyway, are there lightmaps hidden somewhere?)

- I'm pretty sure WoW does not use lightmaps in it's [[WMO]]s...

After further inspection, this is it, actual pre-lit vertex colors for [[WMO]]s - vertex lighting is turned off. This is used if flag 0x2000 in the [[WMO#MOGI_chunk|MOGI]] chunk is on for this group. This pretty much fixes indoor lighting in Ironforge and Undercity. The "light" lights are used only for [[M2]] models (doodads and characters). (The "too dark" corridors seemed like that because I was looking at it in a window - in full screen it looks pretty much the same as in the game) Now THAT's progress!!!

''Yes, 0x2000 (INDOOR) flagged WMO groups use _only_ MOCV for lighting, however this chunk is also used to light outdoor groups as well like lantern glow on buildings, etc.  If 0x8 (OUTDOOR) flag is set, you start out with normal world lighting (like with light db params) and then you multiply these vertex colors by the texture color and add it to the world lighting.  This makes many models look much better.  See the Forsaken buildings in Howling Fjord for an example of some that make use of this a lot for glowing windows and lamps. [[User:Relaxok|Relaxok]] 18:29, 20 March 2013 (UTC)''

=== CMapObjGroup::FixColorVertexAlpha ===

Prior to being passed to the shaders, MOCV values are manipulated by the CMapObj::FixColorVertexAlpha function in the client. This function performs different manipulations depending on the relationship between the vertex and the MOBA it appears in. It's possible that FixColorVertexAlpha did not always exist, or does not exist in later versions of WoW. It appears to have existed in WotLK, Cata, MoP, and WoD.

In client versions that use FixColorVertexAlpha, without applying the function, certain parts of WMOs are noticeably wrong: fireplaces lack a glowing effect; the red light cast from bellows in blacksmith WMOs is undersaturated; etc.

'''Warning:''' this should only be used for the "first" MOCV chunk which is referenced by the [[#group_flags|group flag 0x4]]. 
Regardless of whether this flag is set or not, this process should never be applied to the MOCV chunk referenced by the flag SMOGroup::CVERTS2 (0x1000000) as its purpose is different. Only the alpha values from that "second" chunk (which can be the only MOCV chunk present in the group file) are used for the purpose of blending the textures together.


==== WMOs with MOHD->flags & 0x08 ====

Only one manipulation takes place:

MOCVs matching vertices in MOGP->batchCounts[1] and MOGP->batchCounts[2] are modified like so:

 1. If MOGP.flags & 0x08, replace MOCV->color[a] with 255; else replace MOCV->color[a] with 0

==== All other WMOs ====

The following manipulations take place:

MOCVs matching vertices in MOGP->batchCounts[0] (aka unkBatchCount) are modified like so:
 1. Subtract MOHD->color[r|g|b]
 2. Subtract MOCV->color[r|g|b] * MOCV->color[a]
 3. Divide new MOCV->color[r|g|b] values by 2.0

MOCVs matching vertices in MOGP->batchCounts[1] and MOGP->batchCounts[2] are modified like so:
 1. Subtract MOHD->color
 2. Add (MOCV->color[r|g|b] * MOCV->color[a]) >> 6
 3. Divide MOCV->color[r|g|b] values by 2.0
 4. If values are >= 0 and  <= 255, keep value as is; else clamp new value to 0, 255.
 5. If MOGP.flags & 0x08, replace MOCV->color[a] with 255; else replace MOCV->color[a] with 0

==== Decompiled code ====

From build 18179, courtesy of schlumpf

<pre>
void CMapObjGroup::FixColorVertexAlpha(CMapObjGroup *mapObjGroup)
{
  int begin_second_fixup = 0;
  if ( mapObjGroup->unkBatchCount )
  {
    begin_second_fixup = mapObjGroup->moba[mapObjGroup->transBatchCount-1].maxIndex+ 1;
  }

  if ( mapObjGroup->m_mapObj->mohd->flags & flag_has_some_outdoor_group )
  {
    for (int i (begin_second_fixup); i < mapObjGroup->mocv_count; ++i)
    {
      mapObjGroup->mocv[i].w = mapObjGroup->m_groupFlags & SMOGroup::EXTERIOR ? 0xFF : 0x00;
    }
  }
  else
  {
    if ( mapObjGroup->m_mapObj->mohd->flags & flag_skip_base_color )
    {
      v35 = 0;
      v36 = 0;
      v37 = 0;
    }
    else
    {
      v35 = (mapObjGroup->m_mapObj->mohd.color >> 0) & 0xff;
      v37 = (mapObjGroup->m_mapObj->mohd.color >> 8) & 0xff;
      v36 = (mapObjGroup->m_mapObj->mohd.color >> 16) & 0xff;
    }

    for (int mocv_index (0); mocv_index < begin_second_fixup; ++mocv_index)
    {
      mapObjGroup->mocv[mocv_index].x -= v36;
      mapObjGroup->mocv[mocv_index].y -= v37;
      mapObjGroup->mocv[mocv_index].z -= v35;

      v38 = mapObjGroup->mocv[mocv_index].w / 255.0f;

      v11 = mapObjGroup->mocv[mocv_index].x - v38 * mapObjGroup->mocv[mocv_index].x;
      assert (v11 > -0.5f);
      assert (v11 < 255.5f);
      mapObjGroup->mocv[mocv_index].x = v11 / 2;
      v13 = mapObjGroup->mocv[mocv_index].y - v38 * mapObjGroup->mocv[mocv_index].y;
      assert (v13 > -0.5f);
      assert (v13 < 255.5f);
      mapObjGroup->mocv[mocv_index].y = v13 / 2;
      v14 = mapObjGroup->mocv[mocv_index].z - v38 * mapObjGroup->mocv[mocv_index].z;
      assert (v14 > -0.5f);
      assert (v14 < 255.5f);
      mapObjGroup->mocv[mocv_index++].z = v14 / 2;
    }

    for (int i (begin_second_fixup); i < mapObjGroup->mocv_count; ++i)
    {
      v19 = (mapObjGroup->mocv[i].x * mapObjGroup->mocv[i].w) / 64 + mapObjGroup->mocv[i].x - v36;
      mapObjGroup->mocv[i].x = std::min (255, std::max (v19 / 2, 0));

      v30 = (mapObjGroup->mocv[i].y * mapObjGroup->mocv[i].w) / 64 + mapObjGroup->mocv[i].y - v37;
      mapObjGroup->mocv[i].y = std::min (255, std::max (v30 / 2, 0));

      v33 = (mapObjGroup->mocv[i].w * mapObjGroup->mocv[i].z) / 64 + mapObjGroup->mocv[i].z - v35;
      mapObjGroup->mocv[i].z = std::min (255, std::max (v33 / 2, 0));

      mapObjGroup->mocv[i].w = mapObjGroup->m_groupFlags & SMOGroup::EXTERIOR ? 0xFF : 0x00;
    }
  }
}
</pre>

=== CMapObj::AttenTransVerts ===

Similar to FixColorVertexAlpha above, the client will also run MOCV values through the CMapObj::AttenTransVerts function prior to rendering.

In MoP and WoD, it appears that the client only runs AttenTransVerts in cases where flag 0x01 is NOT set on MOHD.flags.

AttenTransVerts only modifies MOCV values for vertices in MOGP.batchCounts[0] (aka unkBatchCount) batches.

The function iterates over all vertices in MOGP.batchCounts[0], and checks all portals for the group:
* If no portals are found that lead to a group with MOGI.flags & (0x08 | 0x40), all MOCV alpha values are set to 0.0.
* If a portal is found leading to a group with MOGI.flags & (0x08 | 0x40), each MOCV alpha is manipulated to be a range of 0.0 to 1.0 based on the distance of the corresponding vertex to the portal. Additionally, the RGB values for each MOCV are bumped by: (0.0 to 1.0) * (127 - existingRGB)

==== Decompiled code ====

 void CMapObj::AttenTransVerts (CMapObj *mapObj, CMapObjGroup *mapObjGroup)
 {
   mapObjGroup->field_98 |= 1u;
   if (!mapObjGroup->unkBatchCount)
   {
     return;
   }
 
   for ( std::size_t vertex_index (0)
       ; vertex_index < (*((unsigned __int16 *)&mapObjGroup->moba[(unsigned __int16)mapObjGroup->unkBatchCount] - 2) + 1)
       ; ++vertex_index
       )
   {
     float opacity_accum (0.0);
 
     for ( std::size_t portal_ref_index (mapObjGroup->mogp->mopr_index)
         ; portal_ref_index < (mapObjGroup->mogp->mopr_index + mapObjGroup->mogp->mopr_count)
         ; ++portal_ref_index
         )
     {
       SMOPortalRef const& portalRef (mapObj->mopr[portal_ref_index]);
       SMOPortal const& portal (mapObj->mopt[portalRef.portalIndex]);
       C3Vector const& vertex (&mapObjGroup->movt[vertex_index]);
 
       float const portal_to_vertex (distance (portal.plane, vertex));
 
       C3Vector vertex_to_use (vertex);
 
       if (portal_to_vertex > 0.001 || portal_to_vertex < -0.001)
       {
         C3Ray ray ( C3Ray::FromStartEnd
                       ( vertex
                       , vertex
                       + (portal_to_vertex > 0 ? -1 : 1) * portal.plane.normal
                       , 0
                       )
                   );
         NTempest::Intersect
           (ray, &portal.plane, 0LL, &vertex_to_use, 0.0099999998);
       }
 
       float distance_to_use;
 
       if ( NTempest::Intersect ( vertex_to_use
                                , &mapObj->mopv[portal.base_index]
                                , portal.index_count
                                , C3Vector::MajorAxis (portal.plane.normal)
                                )
          )
       {
         distance_to_use = portalRef.side * distance (portal.plane, vertex);
       }
       else
       {
         distance_to_use = NTempest::DistanceFromPolygonEdge
           (vertex, &mapObj->mopv[portal.base_index], portal.index_count);
       }
 
       if (mapObj->mogi[portalRef.group_index].flags & 0x48)
       {
         float v25 (distance_to_use >= 0.0 ? distance_to_use / 6.0f : 0.0f);
         if ((1.0 - v25) > 0.001)
         {
           opacity_accum += 1.0 - v25;
         }
       }
       else if (distance_to_use > -1.0)
       {
         opacity_accum = 0.0;
         if (distance_to_use < 1.0)
         {
           break;
         }
       }
     }
 
     float const opacity ( opacity_accum > 0.001
                         ? std::min (1.0f, opacity_accum)
                         : 0.0f
                         );
 
     //! \note all assignments asserted to be > -0.5 && < 255.5f
     CArgb& color (mapObjGroup->mocv[vertex_index]);
     color.r = ((127.0f - color.r) * opacity) + color.r;
     color.g = ((127.0f - color.g) * opacity) + color.g;
     color.b = ((127.0f - color.b) * opacity) + color.b;
     color.a = opacity * 255.0;
   }
 }
== MOC2 chunk ==

*'''Vertex colors, 4 bytes per vertex (BGRA).''' 

 CImVector colorVertex2[];

These are not colors in a strict sense, but more like weights, that that function like colors. These are used in math for PARALLAX and UNK_DF_SHADER_23 shaders

== MLIQ chunk ==

*'''Specifies liquids inside WMOs.''' 
This is where the water from Stormwind and BFD etc. is hidden. (slime in Undercity, pool water in the Darnassus temple, some lava in IF)

Chunk header:
 struct header
 {
 /*0x00*/  {{Template:Type|C2iVector}} liquidVerts; // number of vertices (x, y)
 /*0x08*/  {{Template:Type|C2iVector}} liquidTiles; // number of tiles (ntiles = nverts-1)
 /*0x10*/  {{Template:Type|C3Vector}} liquidCorner; // base coordinates for X and Y
 /*0x1C*/  uint16_t liquidMtlId;   // material ID (index into [[#MOMT_chunk|MOMT]])
 }

After the header, verts and tiles follow:

 struct SMOLVert
 {
   union
   {
     struct SMOWVert
     {
       uint8_t flow1;
       uint8_t flow2;
       uint8_t flow1Pct;
       uint8_t filler;
       float height;
     }  waterVert;
     struct SMOMVert
     {
       int16_t s;
       int16_t t;
       float height;
     } magmaVert;
   };
 } liquidVertexList[xverts*yverts];
 
 struct SMOLTile
 {
   uint8_t legacyLiquidType : 4; // For older WMOs, used to set liquid type. 
   uint8_t unknown1 : 1;
   uint8_t unknown2 : 1;
   uint8_t fishable : 1;
   uint8_t shared : 1;
 } liquidTileList[xtiles*ytiles];

The liquid data contains the vertex height map (xverts * yverts * 8 bytes) and the tile flags (xtiles * ytiles bytes) as described in [[ADT]] files ([[ADT#MCLQ_sub-chunk|MCLQ]] chunk). The length and width of a liquid tile is the same as on the map, that is, 1/8th of the length of a map chunk. (which is in turn 1/16th the length of a map tile).

Note that although I could read Mh2o's heightmap and existstable in row major order (like reading a book), I had to read this one in column major order to compensate for a 90° misrotation. --[[User:Bananenbrot|Bananenbrot]] 22:02, 1 August 2012 (UTC)

Either the unknown data or the "types" must somehow control how the points at the edges work. In looking at 3D mesh screen captures, something is changed to create a flat edge where it meets other MLIQ chunks. The first Unknown data is always 0 when a point isn't used. Other seen values: 1, 4, 12, 22, 27, 31, 105, & 124. Not yet sure what they mean/how to use them, I suspect they become the modifier for the edge placement points. --[[User:Kjasi|Kjasi]] 14 February 2016

WMOs can have liquid in them even if MLIQ is not present! If MOGP.groupLiquid is set but no MLIQ is present or xtiles = 0 or ytiles = 0 then entire group is filled with liquid. In this case liquid height is equal to MOGP.boundingBox.max.z.
This seems to only happen if MOHD.flags.use_liquid_type_dbc_id is set

In older WMOs without the MOHD root flag flag_use_liquid_type_dbc_id set :  if MOGP.groupLiquid == 15 (green lava), the tile flags legacyLiquidType are used to set the liquid type.
First it checks if legacyLiquidType <= 20. If so : 
  - if legacyLiquidType == 1 : Liquidtype = 14 (Ocean)
  - if legacyLiquidType == 2 : Liquidtype = 19 (WMO Magma)
  - if legacyLiquidType == 3 : Liquidtype = 20 (WMO Slime)
  - if  legacyLiquidType >= 4 : Liquidtype = 13 (WMO Water)
  
Else : Liquidtype = legacyLiquidType + 1
  


=== how to determine {{DBRef|table=LiquidType}} to use ===

 enum liquid_basic_types
 {
   liquid_basic_types_water = 0,
   liquid_basic_types_ocean = 1,
   liquid_basic_types_magma = 2,
   liquid_basic_types_slime = 3,
 
   liquid_basic_types_MASK = 3,
 };
 enum liquid_types
 {
   // ...
   LIQUID_Slow_Water = 5,
   LIQUID_Slow_Ocean = 6,
   LIQUID_Slow_Magma = 7,
   LIQUID_Slow_Slime = 8,
   LIQUID_Fast_Water = 9,
   LIQUID_Fast_Ocean = 10,
   LIQUID_Fast_Magma = 11,
   LIQUID_Fast_Slime = 12,
   LIQUID_WMO_Water = 13,
   LIQUID_WMO_Ocean = 14,
   LIQUID_Green_Lava = 15,
   LIQUID_WMO_Water_Interior = 17,
   LIQUID_WMO_Magma = 19,
   LIQUID_WMO_Slime = 20,
 
   LIQUID_END_BASIC_LIQUIDS = 20,
   LIQUID_FIRST_NONBASIC_LIQUID_TYPE = 21,
 
   LIQUID_NAXX_SLIME = 21,
   LIQUID_Coilfang_Raid_Water = 41,
   LIQUID_Hyjal_Past_Water = 61,
   LIQUID_Lake_Wintergrasp_Water = 81,
   LIQUID_Basic Procedural Water = 100,
   LIQUID_CoA_Black_Magma = 121,
   LIQUID_Chamber_Magma = 141,
   LIQUID_Orange_Slime = 181,
   // ...
 };
 
 enum SMOGroup::flags
 {
   LIQUIDSURFACE = 0x1000,
   is_not_water_but_ocean = 0x80000,
 };
 
 liquid_types to_wmo_liquid (int x)
 {
   liquid_basic_types const basic (x & liquid_basic_types_MASK);
   switch (basic)
   {
   case liquid_basic_types_water:
     return (smoGroup->flags & is_not_water_but_ocean) ? LIQUID_WMO_Ocean : LIQUID_WMO_Water;
   case liquid_basic_types_ocean:
     return LIQUID_WMO_Ocean;
   case liquid_basic_types_magma:
     return LIQUID_WMO_Magma;
   case liquid_basic_types_slime:
     return LIQUID_WMO_Slime;
   }
 }
 
 
 if ( mapObj->mohd_data->flag_use_liquid_type_dbc_id )
 {
   if ( smoGroup->groupLiquid < LIQUID_FIRST_NONBASIC_LIQUID_TYPE )
   {
     this->liquid_type = to_wmo_liquid (smoGroup->groupLiquid - 1);
   }
   else
   {
     this->liquid_type = smoGroup->groupLiquid;
   }
 }
 else
 {
   if ( smoGroup->groupLiquid == LIQUID_Green_Lava )
   {
     this->liquid_type = 0; {{Template:Unverified| // use to_wmo_liquid(SMOLTile->liquid) ? It seems to work alright. }}
     // edit : the code above will turn any "green alva" to water, but green lava is used for both water and lava liquids in vanilla/bc models, so this is not right. Need to figure out how to determine if it is lava or water.
   }
   else
   {
     int const liquidType (smoGroup->groupLiquid + 1);
     if ( smoGroup->groupLiquid < LIQUID_END_BASIC_LIQUIDS )
     {
       this->liquid_type = to_wmo_liquid (smoGroup->groupLiquid);
     }
     else
     {
       this->liquid_type = smoGroup->groupLiquid + 1;
     }
     assert (!liquidType || !(smoGroup->flags & SMOGroup::LIQUIDSURFACE));
   }
 }

== MORI ==
 uint16_t triangle_strip_indices[];

== MORB ==
{{SectionBox/VersionRange|min_expansionlevel=4|note={{Unverified|Could have been added earlier}}}}

* ignored if !CMap::enableTriangleStrips
* modifies MOBA, therefore has same count.
* size is not checked, but 2 * sizeof(int), even though it is only (int, short).
 struct MORB_entry
 {
   uint32_t start_index;
   uint16_t index_count;
   uint16_t padding;
 }
* overwrites 0xC and 0x10 of MOBA (start, count).

== MOTA ==
{{SectionBox/VersionRange|min_expansionlevel=4|note={{Unverified|Could have been added earlier}}}}

* Map Object Tangent Array

 struct MOTA
 {
   unsigned short first_index[moba_count]; // either -1 or first index of batch.count indices into tangents[]. 
                                           // if auto-generated, only has entries for batches with 
                                           // material[batch.material].shader == 10 or 14.
   {{Template:Type|C4Vector}} tangents[accumulated_num_indices]; // sum (batches[i].count | material[batches[i].material].shader == 10 or 14)
 };

Is auto generated, if there are batches with shaders 10 or 14, but no tangents. (And maybe some additional condition.) See CMapObjGroup::Create().

== MOBS ==
{{SectionBox/VersionRange|min_expansionlevel=4|note={{Unverified|Could have been added earlier}}}}

 struct {
   char unk0[10];
   short materialIDBig; // Index into MOMT
   int field_2; // Divided by 3 upon usage
   short field_6; // Divided by 3 upon usage
   char unk1[4];
   char flagThing; // If & 2 use materialIDBig otherwise use materialIDSmall
   char materialIDSmall; // Index into MOMT
 } map_object_shadow_batches[];

== MDAL ==
{{SectionBox/VersionRange|min_expansionlevel=6|note={{Unverified|Could have been added earlier}}}}

 struct
 {
   {{Template:Type|CArgb}} replacement_for_header_color; // if not present, take color from header
 } mdal;

==MOPL==
{{SectionBox/VersionRange|min_expansionlevel=6|note={{Unverified|Could have been added earlier}}}}

* requires MOGP.canCutTerrain
 {{Template:Type|C4Plane}} terrain_cutting_planes[<=32];

==MOPB==
{{SectionBox/VersionRange|min_expansionlevel=7|note={{Unverified|Could have been added earlier}}}}

 struct {
   char _1[0x18];
 } map_object_prepass_batches[];

==MOLS==
{{SectionBox/VersionRange|min_expansionlevel=7|note={{Unverified|Could have been added earlier}}}}

 struct {
   char _1[0x38];
 } map_object_spot_lights[];
==MOLP==
{{SectionBox/VersionRange|min_expansionlevel=7|note={{Unverified|Could have been added earlier}}}}

 struct {
    uint32_t unk;
    CImVector color; 
    C3Vector pos; //position of light
    float intensity; 
    float attenStart;
    float attenEnd;
    float unk4;   //Only seen zeros here 
    uint32_t unk5;
    uint32_t unk6; //CArgb?
 } map_object_point_lights[];

==MLSS==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
 struct {
   uint32_t offset;
   uint32_t mols_count; // spotlights per set
 } map_object_lightset_spotlights[];

''note: client uses a record size of 8 (checked with client 29297)''

==MLSP==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
 struct {
   uint32_t offset;
   uint32_t molp_count; // pointlights per set
 } map_object_lightset_pointlights[];

''note: client again uses a record size of 8 (checked with client 29297)''

==MLSO==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}

In binary, not in files

 struct {
   uint32_t offset;
   uint32_t MOS2_count;
 } mapobject_spotlight_animsets [];

''note: client uses a record size of 8 (checked with client 29297)''

==MLSK==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
 struct {
   uint32_t offset;
   uint32_t MOP2_count;
 } mapobject_pointlight_animsets[];

==MOS2==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}

In binary, not in files

*'''Unknown struct layout, 108 bytes per struct.'''

 struct {
   byte data[108]; // unknown
 } map_object_spotlight_anims[];

==MOP2==
{{SectionBox/VersionRange|min_build=8.1.0.27826|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}

Currently only in file 2143042 as of 8.1.5.28938: world/wmo/zuldazar/orc/8or_pvp_warsongbg_main01.wmo.

  struct {
  /*0x00*/  uint32_t lightIndex;
  /*0x04*/  CImVector color;
  /*0x08*/  C3Vector pos;
  /*0x14*/  float attenuationStart;
  /*0x18*/  float attenuationEnd;
  /*0x1C*/  float intensity;
  /*0x20*/  C3Vector rotation;
            struct {
  /*0x2C*/    float flickerIntensity;
  /*0x30*/    float flickerSpeed;
  /*0x34*/    int flickerMode;
            } lightTextureAnimation;
            struct {
  /*0x38*/    int unk0;
  /*0x3C*/    int unk1;
  /*0x40*/    int unk2;
  /*0x44*/    int unk3;
  /*0x48*/    int lightTextureFileDataId;
  /*0x4C*/    int unk5;
  /*0x50*/    int unk6;
  /*0x54*/    int unk7;
  /*0x58*/    int unk8;
  /*0x5C*/    int unk9;
            } lightUnkRecord;
  } map_object_pointlight_anims[];


<code>flickerIntensity</code>, <code>flickerSpeed</code> and <code>flickerMode</code> function the same as in [[WMO#MNLD|MNLD]] chunk.

<code>lightUnkRecord</code> is structure, but only <code>textureFileDataId</code> out of it is used by client

==MPVR==
{{SectionBox/VersionRange|min_build=8.3.0.33775|min_expansionlevel=8|note={{Unverified|Could have been added earlier}}}}
  uint16_t mapobject_particulate_volume_refs[];

==MAVR==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
  uint16_t mapobject_ambient_volume_refs[];

==MBVR==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
  uint16_t mapobject_box_volume_refs[];
==MFVR==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.???}}
  uint16_t mapobject_fog_volume_refs[]; // into MFOG (and MFED)

==MNLR==
{{Template:SectionBox/VersionRange|min_expansionlevel=9|min_build=9.0.1.33978}}
  uint16_t mapobject_new_light_refs[];

==MOLM==
{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494|note=Only used in v14}}

Lightmaps were the original lighting implementation for WMOs and the default light mode used in the alpha clients. They were replaced by "vertex lighting" in {{Template:Sandbox/PrettyVersion|expansionlevel=0|build=0.6.0.3592}}.
The alpha clients can switch between light modes using the <tt>mapObjLightMode</tt> console command (CWorld:enables & 0x400).

This chunk contains information for blitting the [[#MOLD|MOLD]] colour palette. There is one entry for each [[#MOPY_chunk|MOPY]] and is referenced by matching index.

Exterior lit groups (SMOGroup::EXTERIOR | SMOGroup::EXTERIOR_LIT) are excluded and default to (0,0,0). All other groups have their light colour calculated from the visible SMOPolys using their associated [[#MOLV|MOLV]], [[#MOLM|MOLM]] and [[#MOLD|MOLD]] entries. This colour is then blended with the texture. The client enforces a minimum of 24 for each colour component {{Template:Unverified|and skews the colour based on the dominant RGB component.}}

 struct SMOLightmap
 {
   char x;
   char y;
   char width;
   char height;
 } lightmapList[];

==MOLD==
{{Template:SectionBox/VersionRange|max_expansionlevel=0|max_build=0.5.5.3494|note=Only used in v14}}
This chunk stores a {{Template:Unverified|255x255}} DXT1 compressed colour palette.
 struct SMOLightmapTex
 {
   char texels[32768];
   union
   {
     char inMemPad[4];
     CGxTex *gxTexture;
     HTEXTURE__ *hTexture;
   };                      // always inMemPad == 0 in file
 } lightmapTexList[];

==MPB*==
These chunks are barely ever present (the one file known is <tt>StonetalonWheelPlatform.wmo</tt> from alpha). No version of the client ever read them though. They might be an early form of [[PD4]] files, inlined into the WMO and not per root but per group.

{{Template:Unverified|MPBV and MPBP appear to be <tt>(uint16_t start, uint16_t count)s</tt>. This is reasoned by the values being sequential and totalling the entry count of the next chunk. If this is the case, the structure may actually produce groups of groups of vertices e.g. [https://gist.github.com/barncastle/13f24fbdea8d41980e29734c34063f13 StonetalonWheelPlatform].}}

===MPBV===
 uint16_t mpbv[];
===MPBP===
 uint16_t mpbp[];
===MPBI===
 uint16_t mpb_indices[];     // {{Template:Unverified|triangle}} vertex indices into into [[#MPBG]]
===MPBG===
 {{Template:Type|C3Vector}} mpb_vertices[];

[[Category:Format]]
