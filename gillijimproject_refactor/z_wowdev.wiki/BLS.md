BLS is the container format that stores the GPU shaders used to render the world. In WoD, there are now four different shader types under the Shaders\* directory.

*Vertex
**(versions: arbvp1, vp40, glvs_150, ps_2_0, ps_3_0, ps_4_0, ps_5_0)
*Fragment
**(versions: arbfp1, fp40, glfs_150, ps_2_0, ps_3_0, ps_4_0, ps_5_0)
*Geometry
**(versions: glgs_150, gs_4_0, gs_5_0)
*Hull/Domain (equivalent to Tessellation in OpenGL, except WoW currently only has these shaders for DX)
**(versions: ds_5_0/hs_50)

In MoP+, the BLS format changed significantly (the version increased from 1.3 to 1.4), a new header has been introduced and all of the shader text inside the BLS files is now contained inside compressed chunks. Very little appears to be known about the structure of the format itself, and the meaning of the various fields. Below is enough information to be able to extract the compressed chunks and read out shader text from the various shader components. Additionally, the Shaders\Effects folder contains [[WFX]] files that allow WoW to map shader types from models onto the various shader components (although, there are occasions in the M2 source code where shaders are picked in code).

==BLS v1.4+ (Cata+)==
{{Template:SectionBox/VersionRange|min_expansionlevel=4}}
The BLS format changed with the release of Cata, the version number of the file was bumped and all BLS files have the same "GXSH" magic signature. These new v1.4 files now contain compressed shader text but extra steps are required to be able to decode the text.

===Header===
 struct BLSHeader {
 ''/*0x00*/''	char magic[4];			// FourCC-style magic, "GXVS", "GXPS", etc. Character order reversed in-file. "GXSH" in all new WoD shaders.
 ''/*0x04*/''	uint32_t version;		// version, 0x10003 in WotLK, 0x10004 in Cata/MoP/WoD, 0x10005 in BfA (25902), 0x10006 in BfA (25976), see below table for more
 ''/*0x08*/''	uint32_t permutationCount;	// (from old definition)
 ''/*0x0c*/''	uint32_t nShaders;		// the expected number of shaders in this file (this is the number
 						// of distinct shaders that will be found if you inspect the file after decompression)
 ''/*0x10*/''	uint32_t ofsCompressedChunks;	// offset to array of offsets to compressed chunks
 ''/*0x14*/''	uint32_t nCompressedChunks;	// number of compressed chunks
 ''/*0x18*/''	uint32_t ofsCompressedData;	// offset to the start of the compressed (zlib) data 
 						// bytes (all offsets from the array above are offset by this much)

This header starts at 0x00 in the BLS file. It is followed by an BLSHeader.nShaders amount of uint32 offsets pointing to the start of each shader block in the decompressed data. Said data is at ofsCompressedData and must be decompressed with zlib inflate. There are multiple chunks of separately compressed zlib data (all sections start with ''78 9C'', the zlib magic header for default compression). To decompress the file, take the ofsCompressedData offset, then for each chunk read the offset from ofsCompressedChunks:

 (unsigned char *)ptr + header->ofsCompressedData + ofsCompressedChunks[index]

Each decompressed chunk should be appended to the last. Some of the shader text spans multiple chunks. The offsets just help with finding the start of the zlib header for each chunk. Inside the decompressed data, the old WotLK block header is present, along with a shader preamble (WoD+).

'''Versions'''
{| class="wikitable sortable"
|-
! Version
! Patch
|-
| <tt>0x10003</tt>
| <tt>{{Template:Sandbox/PrettyVersion|expansionlevel=3}}</tt>
|-
| <tt>0x10004</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=4|max_expansionlevel=6}}</tt>
|-
| <tt>0x10005</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=8|max_expansionlevel=8}}</tt>
|-
| <tt>0x10006</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=8|max_expansionlevel=8|min_build=8.0.1.25976}}</tt>
|-
| <tt>0x10007</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=8|max_expansionlevel=8}}</tt>
|-
| <tt>0x10008</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=8|max_expansionlevel=8}}</tt>
|-
| <tt>0x10009</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=9|max_expansionlevel=9}}</tt>
|-
| <tt>0x1000A</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=9|max_expansionlevel=9|max_build=9.0.1.34278}}</tt>
|-
| <tt>0x1000B</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=9|max_expansionlevel=9|min_build=9.0.1.34365}}</tt>
|-
| <tt>0x1000C</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=10|max_expansionlevel=10}}</tt>
|-
| <tt>0x1000D</tt>
| <tt>{{Template:Sandbox/VersionRange|min_expansionlevel=11|min_build=11.0.0.54935|max_expansionlevel=11}}</tt>
|}
===Shader Block===
The decompressed BLS data is formatted in the same format that old WotLK shaders used, with the exception of WoD shaders that have a new (important) structure that is needed to find the start of the shader text. It no longer appears to be true that the top-level ''permutationCount'' field represents the number of shaders present in the inner BLS data.

'''Note:''' all the instances of ''BLSBlock'' are padded to the next nearest 4-byte alignment.

'''Note:''' In BFA? 3 fields of this struct were removed.

'''Note:''' [this refers to DirectX shaders only] As of 8.1.5, the BLSBlock header has 1 less field than the below struct, len is at 0x18.  Also, the DXBC blob starts immediately after the block header.  DXBC blob size is at 0x18 in the blob.

 struct BLSBlock {
 ''/*0x00*/''	uint32_t flags;
 ''/*0x04*/''	uint32_t flags2;
 ''/*0x08*/''	uint32_t unknown;
 ''/*0x0c*/''	uint32_t unknown2;
 ''/*0x10*/''	uint32_t unknown3;
 ''/*0x14*/''	uint32_t unknown4;
 ''/*0x18*/''	uint32_t unknown5;
 ''/*0x1c*/''	uint32_t len;
 }

The length field dictates where this block ends, in order to iterate over all blocks in BLS file you would start at the first block, read ''len'' and then skip to the next block using that. In MoP, the shader text was present immediately after the BLSBlock structure. However, in WoD there is a significant amount of new data between the end of this structure and the beginning of the shader text. Experimentally, there appears to be a header in front of this data that holds the offset to the text itself.

The content of the decompressed data depends on the shader type. The ShaderDataHeader below is for 'GLS2', but not verified for any other type.
 struct ShaderDataHeader_GLS2 {
   uint32_t magic; // 'GLS2'
   uint32_t size;
   uint32_t shaderType; // GLShader::ShaderType
   uint32_t type2; // ?, 3 = GLSL
   uint32_t target; // as in GL_FRAGMENT_SHADER
   uint32_t codeOffset;
   uint32_t codeSize;
   uint8_t _unk1C[0x8];
   uint32_t some_length;
   uint32_t some_offset;
   uint8_t _unk2C[0x14]
   uint32_t some_other_count;
   uint32_t ofsUniformMapTable; // points to uint32_t[] entries; which points to shader_uniform_info_ts
   uint32_t nUniformMapTable;
 };
 struct shader_uniform_info_t
 {
   uint32_t _unk0;
   uint8_t _unk4;
   uint8_t uniformClass; // 4 = struct
   uint8_t _unk6;
   uint8_t enabled;
   uint16_t shader_constant_index;
   uint16_t mapIndex;
   uint16_t shader_constant_count;
 };


Rough structure of GLS3 header:
  struct ShaderDataHeader_GLS3 {
   uint32_t magic; // 'GLS3'
   uint32_t size;
   uint32_t type2; // ?, 3 = GLSL
   uint32_t unk1;
   uint32_t target;
   uint32_t codeOffset;
   uint32_t codeSize;
   uint32_t unk2;
   uint32_t unk3;//-1
   uint32_t inputParamsOffset; //offset to array of input_shader_uniform_info_t
   uint32_t inputParamCount;
   uint32_t outputOffset; // offset to array of output_shader_uniform_info_t
   uint32_t outputCount;
   uint32_t uniformBufferOffset; // offset to array of uniformBuffer_shader_uniform_info_t
   uint32_t uniformBufferCount;
   uint32_t samplerUniformsOffset; //offset to sampler_shader_uniform_info_t
   uint32_t samplerUniformsCount;
   uint32_t unk5Offset;
   uint32_t unk5Count;
   uint32_t unk6Offset;
   uint32_t unk6Count;
   uint32_t variableStringsOffset;
   uint32_t variableStringsSize;
 };
 struct input_shader_uniform_info_t
 {
   uint32_t glslParamNameOffset; //Offset to zero terminated string
   uint32_t unk0;
   uint32_t internalParamNameOffset;
   uint32_t unk1;
 };
 struct output_shader_uniform_info_t
 {
   uint32_t glslParamNameOffset; //Offset to zero terminated string
   uint32_t unk0;
   uint32_t internalParamNameOffset;
   uint32_t unk1;
 };
 struct uniformBuffer_shader_uniform_info_t
 {
   uint32_t glslParamNameOffset; //Offset to zero terminated string
   uint32_t unk0;
   uint32_t unk1;
 }
 struct sampler_shader_uniform_info_t
 {
   uint32_t glslParamNameOffset; //Offset to zero terminated string
   uint32_t unk0;
   uint32_t unk1;
   uint32_t unk2;
 };

The shader text starts at offset of (BLSBlock + BLSBlock2.offset). The shader text is then present in the format used by the particular graphics API used in that subdirectory. (ie. glvs/glfs == OpenGL GLSL, arbvp1/arbfp1 == OpenGL ARB assembly, vs_*/ps_* == DirectX, etc).

For GLS3 it's not guaranteed that start of next BLSBlock will be at (BLSBlock + BLSBlock.len).


'''Note:''' in 6.1.2 the OpenGL GLSL (glvs/glfs/etc) shaders appear to have been recompiled with a newer version of the HLSL cross-compiler and they are ''exceptionally'' informative. In particular, there are named uniform (constant) blocks that indicate the meaning of the variables.

==BLS v1.3 (WotLK)==
{{Template:SectionBox/VersionRange|min_expansionlevel=0|min_build=0.7.0.3694|max_expansionlevel=3}}
Part of the early WotLK development included an overhaul to the [[BLS]] format introducing the first iteration of the v1.4 structure. However; this did not incur a version change. Luckily, this format was already in place by the first publicly available WotLK build so v1.3 can be simplified to WotLk and pre-WotLK.

===Header===
{{SectionBox/Version|expansionlevel=3}}
*Main header (0xC bytes)
This header is in all files - pixel and vertex shaders in all profiles.
 struct BLSHeader {
 ''/*0x00*/''	char[4] magix;		// in reverse character order: "SVXG" in case of a vertex shader, "SPXG" in case of a fragment shader
 ''/*0x04*/''	uint32 version;		// Always 0x10003 - version 1.3 of format
 ''/*0x08*/''	uint32 permutationCount;
 ''/*0x0C*/''
 };
{{Template:SectionBox/VersionRange|max_expansionlevel=3|max_exclusive=1}}
The next iteration of the old format. A new field has been added after <tt>version</tt> and the <tt>DirEntry</tt> structure has been replaced by a flat offset array.
 struct BLSHeader {
   char magic[4];                // "SVXG" for vertex and "SPXG" for pixel
   uint32_t version;             // always 0x10003 - version 1.3 of format
   uint32_t _unk08;              // always 1
   
   #if pixel_shader
    uint32_t offsets[12];
   #else
    uint32_t offsets[6];
   #endif
 }; 

===Blocks===
{{SectionBox/Version|expansionlevel=3}}
There are permutationCount blocks of the following structure. They are padded to 0x*0, 0x*4, 0x*8 and 0x*C.
 struct BLSBlock {
 ''/*0x00*/''	DWORD flags0;		// seen: 0x3FE80 in pixel shaders; 0x1A0F in vertex shaders. there may be more ..
 ''/*0x04*/''	DWORD flags4;		// seen: 0x200 in pixel shaders; 0x3FEC1 in vertex shaders (there may be more ..)
 ''/*0x08*/''	DWORD unk8;		// Never seen anything in here.
 ''/*0x0C*/''	uint32 size;		// Tells you how large the block actually is.
 ''/*0x10*/''	char data[size];	// In whatever format defined.
 ''/*----*/''
 };
{{Template:SectionBox/VersionRange|max_expansionlevel=3|max_exclusive=1}}
The pre-WotLK block structure is the same as v1.2 however the code is prefixed with an index and conditional sub-index. The sub-index is only used if the index is not 0.

==BLS v1.2 (Beta)==
{{Template:SectionBox/VersionRange|min_expansionlevel=0|max_expansionlevel=0|min_build=0.7.0.3694|max_build=0.9.1.3810}}

===Pixel blocks===
{| class="wikitable"
! Block !! Shader code
|-
| 0 || ps_1_1
|-
| 1 || ps_1_2
|-
| 2 || ps_1_3
|-
| 3 || ps_1_4
|-
| 4 || ps_2_0
|-
| 5 || NV_register_shader
|-
| 6 || NV_texture_shader
|-
| 7 || NV_texture_shader
|-
| 8 || NV_texture_shader
|-
| 9 || ATIfs
|-
| 10 || ARBfp1.0
|-
| 11 || ???
|-
|}

===Vertex blocks===
{| class="wikitable"
! Block !! Shader code
|-
| 0 || vs_1_1
|-
| 1 || vs_2_0
|-
| 2 || ARBvp1.0
|-
| 3 || OpenGL-related program (almost unimplemented in 8606)
|-
| 4 || OpenGL-related program (almost unimplemented in 8606)
|-
| 5 || ???
|-
|}

===Header===
This header is used for both pixel and vertex shaders. It should be noted that, while the code existed there are no vertex shaders in the files of this version.
 struct BLSHeader
 {
   char magic[4];                // "SVXG" for vertex and "SPXG" for pixel
   uint32_t version;             // always 0x10002 - version 1.2 of format
  
   #if pixel_shader
    DirEntry dir[11];            // pixel shaders have 11 blocks
   #else
    DirEntry dir[3];             // vertex shaders have 3 blocks
   #endif
  
   DirEntry endoffile;           // start = file length, count = 0
 }; 
 
 struct DirEntry
 {
   uint32_t start;               // offset to block
   uint32_t count;               // total size of block
 };

===Blocks===
Enumerating the <code>BLSHeader.dir</code> provides the offsets for each block. Entries with a count of 0 should be ignored.

 struct CGxShader {
   uint32_t ccount;
   CGxShaderParam consts[ccount];  
   uint32_t pcount;
   CGxShaderParam params[pcount];   
   uint32_t unknown;                // always 1  
   uint32_t bytes;
   char code[bytes];
 }
 
 struct CGxShaderParam {
   char name[64];
   uint32_t binding;
   float f[16];
   Type type;
   uint32_t unknown; // looks like an offset, almost always 0
   uint32_t unknown; // omnilight index
   
   enum Type
   {
     Type_Vector4 = 0x0,            // [[Common_Types#C4Vector|C4Vector]]
     Type_Matrix34 = 0x1,           // [[Common_Types#C34Matrix|C34Matrix]]
     Type_Matrix44 = 0x2,           // [[Common_Types#C44Matrix|C44Matrix]]
     Type_Texture = 0x3,            // used in terrain3*.bls, terrain4*.bls
     Type_BumpMatrix = 0x4,         // used in terrain3*.bls, terrain4*.bls, Matrix 2x2 for bump
     Type_Vec3 = 0x5,
     Type_Vec2 = 0x6,
     Type_Vec1 = 0x7,
     Type_Matrix33 = 0x8,
     Type_Struct = 0x9, /* no data */
     Type_Array = 0xA, /* no data */
     Type_Force32Bit = 0xFFFFFFFF,
   };
 };

==BLS v1.1 (Alpha)==
{{Template:SectionBox/VersionRange|min_expansionlevel=0|max_expansionlevel=0|min_build=0.5.3.3368|max_build=0.6.0.3592}}

===Header===
This header is used for both pixel and vertex shaders. It should be noted that, while the code existed there are no vertex shaders in the files of this version.
 struct BLSHeader
 {
   char magic[4];                // "SVXG" for vertex and "SPXG" for pixel
   uint32_t version;             // always 0x10001 - version 1.1 of format
   
   #if pixel_shader
    DirEntry dir[11];            // pixel shaders have 11 blocks
   #else
    DirEntry dir[3];             // vertex shaders have 3 blocks
   #endif
 }; 
 
 struct DirEntry
 {
   uint32_t start;               // offset to block
   uint32_t count;               // total size of block
 };

===Blocks===
Enumerating the <code>BLSHeader.dir</code> provides the offsets for each block. Entries with a count of 0 should be ignored.

 struct CGxShader {
   uint32_t ccount;
   CGxShaderParam consts[ccount];
   uint32_t pcount;
   CGxShaderParam params[pcount];
   uint32_t bytes;
   char code[bytes];
 }
 
 struct CGxShaderParam {
   char name[32];
   Type type;
   uint32_t index;
   float f[16];                     // if the type doesn't use all 16 floats the remainder is 0 padded
   
   enum Type
   {
     Type_Vector4 = 0x0,            // [[Common_Types#C4Vector|C4Vector]]
     Type_Matrix34 = 0x1,           // [[Common_Types#C34Matrix|C34Matrix]]
     Type_Matrix44 = 0x2,           // [[Common_Types#C44Matrix|C44Matrix]]
     Type_Force32Bit = 0xFFFFFFFF,
   };
 };

[[Category:Format]]

==nvrs data format==
See NV_register_combiners & NV_register_combiners2 OpenGL extensions to known how those data are used

 struct nvrs_register_input_s
 {
   uint32_t source;
   uint32_t mapping;
   uint32_t usage;
 };
 
 struct nvrs_register_output_s
 {
   uint32_t ab;
   uint32_t cd;
   uint32_t sum;
   uint32_t scale;
   uint32_t bias;
   uint32_t ab_dot;
   uint32_t cd_dot;
   uint32_t mux_sum;
 };
 
 struct nvrs_combiner_data_s
 {
   struct nvrs_register_input_s input_rgb[4];
   struct nvrs_register_output_s output_rgb;
   struct nvrs_register_input_s input_alpha[4];
   struct nvrs_register_output_s output_alpha;
   float constant_color0[4];
   float constant_color1[4];
 };
 
 struct nvrs_data_s
 {
   uint32_t magic;
   uint32_t combiners_count;
   uint8_t clamp_color;
   uint8_t per_stage_constants;
   struct nvrs_combiner_data_s combiners[8];
   struct nvrs_input_data_s inputs[7];
   float constant_color0[4];
   float constant_color1[4];
   uint8_t unknown[0x14];
 };

==nvts data format==
See NV_texture_shader, NV_texture_shader2 and NV_texture_shader3 OpenGL extensions to know how those data are used

 struct nvts_unit_s
 {
   uint32_t operation;
   uint8_t cull_modes[4];
   uint32_t mapping;
   uint32_t previous_texture;
 };
 
 struct nvts_data_s
 {
   uint32_t magic;
   struct nvts_unit_s units[4];
 };
