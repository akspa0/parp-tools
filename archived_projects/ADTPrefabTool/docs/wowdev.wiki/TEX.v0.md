{{Template:SectionBox/VersionRange|min_expansionlevel=4}}
This is storing really low resolution textures for use if the real file is not yet loaded or streamed. One per continent (e.g. Azeroth.tex) and some special ones (spell effects by class). [[Chunk|Chunked]] structure. Internally called 'TextureBlob'.

The main purpose of this file is to have a fast access to thousands of BLP textures in low resolution to render continent data from a far distance without the need to load each BLP texture with a high resolution. This will speed up level loading times drastically as well as reduce memory footprint. Most likely texture data is simply extracted from the referenced BLP textures starting at the around 7 lowest mipmap levels.

==structure==
While this is properly chunked, order is required.
===TXVR===
 uint32_t version; // As of 4.0.12479 and 6.0.1.18179: 0. Not checked by client.
                   // As of 8.1.5 (28938): 1 -- TXFN chunks are gone, in TXMB filenameOffset changed to fileId

===TXBT===
 struct SBlobTexture {
 #if {{Template:Sandbox/VersionRange|min_expansionlevel=8}} 8.1.5 (28939) [TXVR:Version >= 1]
   uint32_t fileId;
 #else
   uint32_t filenameOffset;  // offset within TXFN chunk
 #endif
   uint32_t txmdOffset;      // TXMD chunk offset relative to end of TXFN chunk (or TXBT chunk for TXVR:Version=1 >8.1.5 as the TXFN chunk was removed)
   uint8_t sizex;            // 64,32,16,8
   uint8_t sizey;            // 64,32,16,8 (sizex, sizey can have different sizes) [special case: sizex = sizey * 6]
   uint7_t numLevels;        // mipmaps (<= 7 as it limits to 64x64)
   uint1_t loaded;
   uint4_t dxt_type;         // 0  = DXT1, 1 = DXT3, 2 = DXT5
   uint4_t alpha_dxt1   : 1; // DXT1 with alpha (for dxt_type = DXT1, prefer argb1555 over rgb565, when DXT1 unavailable)
   uint4_t _            : 1;
   uint4_t unknown_dxt3 : 1; // seems to be only set for DXT3 
   uint4_t unknown      : 1; // seems to be only set for DXT3, DXT5
 } entries[];

Starting at around 28939 (8.1.5 PTR) the TXFN chunk was removed with all filenames and the filenameOffset field changed to fileId.

Flags should be correlated in some way to [[BLP]] flags and formats.

===TXFN===
{{SectionBox/VersionRange|max_build=8.1.5.28938|max_expansionlevel=8|max_exclusive=1|note=TXFN has been removed}}

 char filenames[]; // zero-terminated, no file ending

===TXMD[]===
 char textureData[]; // length depends on size, type and flags.

Can contain multiple mipmap levels for one texture. See UpdateBlobTexture() for unpacking.

----

Data is an array of compressed texture data in format DXT1, DXT3 or DXT5. The size of the texture is determined by the mipmap level which for every next level halves both width/height dimensions. As the size can be calculated, there is no need for an index.

DXT1, DXT3, DXT5 all use the same color encoding. The only difference is the depth of the alpha channel. For DXT1 there is no alpha channel, only in pixel data there is a bit to determine transparent or opaque. For DXT3 the alpha channel has a depth of 4, for DXT5 the alpha channel has a depth of 8. All 3 texture formats have a tile layout of 4x4 pixels. For DXT1 the tile size is always 8 bytes per 4x4 pixels, for DXT3/DXT5 the tile size is always 16 bytes per 4x4 pixels.

For a 32x16 DXT1 sample texture the data layout is as follows and should have 6 mipmap levels:

 Mipmap Level 0 =>   32x16 (8 x 4 tiles each 8 bytes = 256 bytes)
 Mipmap Level 1 =>   16x8  (4 x 2 tiles each 8 bytes =  64 bytes)
 Mipmap Level 2 =>    8x4  (2 x 1 tiles each 8 bytes =  16 bytes)
 Mipmap Level 3 =>    4x2  (1 x 1 tiles each 8 bytes =   8 bytes)
 Mipmap Level 4 =>    2x1  (1 x 1 tiles each 8 bytes =   8 bytes)
 Mipmap Level 5 =>    1x1  (1 x 1 tiles each 8 bytes =   8 bytes)

This adds up to a total size of 360 bytes. For DXT3/DXT5 double all mipmap level sizes as each tile as 16 bytes instead of 8.

Because of an old broken Microsoft DDS writer (which in old times exported all textures to the .dds format and most game engines just imported data back into a proprietary format like this one), the low level mipmaps with a size below 4 were wrongly aligned and had dimensions below 4x4 which by nature of the DXTx compression format is not possible. These mipmap levels are broken beyond repair and need to be reconstructed. The [[BLP]] format suffers the same issue with broken mipmaps.

This makes extracting mipmaps for all levels problematic. Maybe there are additional bugs in the Blizzard data writer, but for some textures even with a size correction for broken mipmaps the combined size does not add up. For about 60,000 textures it fails for around 50. Best option is to either let the GPU re-generate all mipmap levels from level 0, or re-calculate all mipmaps when any dimension falls below 4. Same applies to BLP textures as well.


[[Category:Format]]
