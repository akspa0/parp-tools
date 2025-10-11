BLP files store textures with precalculated mipmaps.
BLP files are capable of storing data with a few different formats.  The primary variables are palettized/RGB and alpha bit depth.  The RGB formats are actually stored using DXT compression (DXT1 for 0-bit alpha and DXT3 for the others), and thus conversion to these formats is somewhat lossy.  Here is a list (with my shorthand for the format in parenthesis):

Palettized, 0-bit alpha (P0) - Example: character skins, clothing.

Palettized, 1-bit alpha (P1) - Example:. clothing (relatively rare).

Palettized, 8-bit alpha (P8) - Example: clothing.


RGB, 0-bit alpha (RGB0) - Example: Sansamroot.blp.

RGB, 1-bit alpha (RGB1) - Example: Peaceflower.blp.

RGB, 8-bit alpha (RGB8) - Example: Sungrass.blp.


''From BLPConverter.rtf''

==  Header ==

 '''Offset 	Type 		Description'''
 0x00 	char[4]		always 'BLP2'
 0x04 	uint32 		Version, always 1
 0x08 	uint8 		Compression: 1 for uncompressed, 2 for DXTC, 3 (cataclysm) for plain A8R8G8B8 textures (see remarks)
 0x09 	uint8 		Alpha channel bit depth: 0, 1 or 8
 0x0A 	uint8 		Something about alpha? (compressed, alpha:0 or 1): 0, (compressed, alpha:8): 1, uncompressed: 2,4 or 8 (mostly 8)
 0x0B 	uint8 		when 0 there is only 1 mipmaplevel. compressed: 0, 1 or 2, uncompressed: 1 or 2 (mostly 1)
 0x0C 	uint32 		X resolution (power of 2)
 0x10 	uint32 		Y resolution (power of 2)
 0x14 	uint32[16] 	offsets for every mipmap level (or 0 when there is no more mipmap level)
 0x54 	uint32[16] 	sizes for every mipmap level (or 0 when there is no more mipmap level)

 struct BLPHeader {
   uint32_t magic;
   uint32_t formatVersion;
   uint8_t colorEncoding;
   uint8_t alphaSize;
   uint8_t preferredFormat;
   uint8_t hasMips;
   uint32_t width;
   uint32_t height;
   uint32_t mipOffsets[16];
   uint32_t mipSizes[16];
   union {
     struct BlpPalPixel {
       uint8_t b; uint8_t g; uint8_t r; uint8_t pad;
     } palette[256];
     struct {
       uint32_t headerSize;
       char headerData[1020];
     } jpeg;
   } extended;
 };

Alpha channel bit depth = 4 are found in Interface\Glues\Credits\*.blp. Most BLP converters are not compatible with this.

== Compressed textures ==

For compressed textures, DXT1 or DXT3 (possibly DXT5 depending on the header values) data follows, for every mipmap level. Unused mipmap levels have an offset and size of 0.

Detailed description of the DXTC formats can be found on MSDN or on Wikipedia (https://en.wikipedia.org/wiki/S3_Texture_Compression)

Some lower mipmap levels in compressed textures have incorrect size. The correct size can be calculated using following formulas:
  if ((textureFormat == "DXT5") || (textureFormat == "DXT3")) {
      validSize = Math.floor((width + 3) / 4) * Math.floor((height + 3) / 4) * 16;
  }
  if ((textureFormat == "RGB_DXT1") || (textureFormat == "RGBA_DXT1")) {
      validSize = Math.floor((width + 3) / 4) * Math.floor((height + 3) / 4) * 8;
  }

To get it right, you need to create a separate buffer of validSize length, read current mipmap level into it, and the rest bytes in buffer can be filled with zeros or left intact.

== Uncompressed textures ==

After the header, 256 32-bit BGRA values make up the palette. The image data (for each mipmap level) consists of 8-bit indices into this palette.

If there is an alpha channel, then transparency data follows the image data in each mipmap level. If the Alpha bit depth is 0, there is no alpha channel. If the Alpha bit depth is 1, the alpha data is 8 alpha bits backed into each byte. If the Alpha big depth is 8, there is one byte per alpha value.

If compression is set to 3 we also have uncompressed textures but they do not use a color palette. They just store the pixels in A8R8G8B8 format. You can read all the pixels and fill them directly into a texture. Examples for such textures are found in:
tileset\Terrain Cube Maps\

--[[User:Cromon|Cromon]] 09:56, 28 February 2012 (UTC)

== Struct definition ==

 enum BLPColorEncoding : uint8_t {
     COLOR_JPEG = 0, // not supported
     COLOR_PALETTE = 1,
     COLOR_DXT = 2,
     COLOR_ARGB8888 = 3,
     COLOR_ARGB8888_dup = 4,    // same decompression, likely other PIXEL_FORMAT
 };
 
 enum BLPPixelFormat : uint8_t {
     PIXEL_DXT1 = 0,
     PIXEL_DXT3 = 1,
     PIXEL_ARGB8888 = 2,
     PIXEL_ARGB1555 = 3,
     PIXEL_ARGB4444 = 4,
     PIXEL_RGB565 = 5,
     PIXEL_A8 = 6,
     PIXEL_DXT5 = 7,
     PIXEL_UNSPECIFIED = 8,
     PIXEL_ARGB2565 = 9,
     PIXEL_BC5 = 11, // DXGI_FORMAT_BC5_UNORM 
     NUM_PIXEL_FORMATS = 12, // (no idea if format=10 exists)
 };
 enum PIXEL_FORMAT_MASKS // 1 << PIXEL_FORMAT
 {
   UNCOMPRESSED_DXTC = (1 << PIXEL_DXT1)
                     | (1 << PIXEL_DXT3)
                     | (1 << PIXEL_DXT5)
   VALID_DXTC_FORMATS = (1 << PIXEL_ARGB8888)
                      | (1 << PIXEL_PalARGB1555DitherFloydSteinberg)
                      | (1 << PIXEL_PalARGB4444DitherFloydSteinberg)
                      | (1 << PIXEL_PalARGB565DitherFloydSteinberg)
   COMPONENT_TEXTURE = (1 << PIXEL_PalARGB2565DitherFloydSteinberg)
 };
 enum mipmap_level_and_flag_type : uint8_t // MIPS_TYPE
 {
   MIPS_NONE = 0x0,
   MIPS_GENERATED = 0x1,
   MIPS_HANDMADE = 0x2, // not supported
   flags_mipmap_mask = 0xF, // level
   flags_unk_0x10 = 0x10,
 };
 struct BLPHeader { // sizeof = 0x494
     u32 magic; // 'BLP2'
     u32 version; // must be 1
     BLPColorEncoding colorEncoding;
     u8 alphaChannelBitDepth; // 0, 1, 8, 4
     BLPPixelFormat preferredFormat;
     mipmap_level_and_flag_type mipmap_level_and_flags;
     u32 width;
     u32 height;
 #define MIPMAP_COUNT 16
 #define PALETTE_SIZE 256
     u32 dataOffsets[MIPMAP_COUNT];
     u32 dataSizes[MIPMAP_COUNT];
     u32 palette[PALETTE_SIZE];
 };


[[Category:Format]]
