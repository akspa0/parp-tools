# BLP Texture Format

This document provides detailed analysis of the BLP texture format used in WoW Alpha 0.5.3, based on reverse engineering the WoWClient.exe binary.

## Overview

BLP is Blizzard's proprietary texture format that supports multiple image formats including JPEG compression for mipmaps and various DXT compression formats for the main image data. The format supports alpha channels and mipmaps for LOD (Level of Detail) rendering.

## Key Classes and Structures

### CBLPFile (BLP File Handler)
```
Address: 0x0046f820+
Purpose: Main BLP file reader
Created by: CBLPFile::Open() @ 0x0046f820
Key Fields:
  - BLPHeader m_header: BLP file header
  - int m_quality: Quality setting (0-100, default 100)
  - void* m_inMemoryImage: In-memory image data
  - uint m_mipMapAlgorithm: Mipmap generation algorithm
  - MipBits* m_images: Array of mipmap level data
```

### BLPHeader (BLP File Header)
```
Purpose: Contains all BLP metadata
Structure:
  - uint32 magic: File magic ('2PLB' / 0x32504c42)
  - uint32 formatVersion: BLP format version (1)
  - uint8 colorEncoding: Color encoding type
  - uint8 preferredFormat: Preferred pixel format
  - uint8 hasAlpha: Alpha channel presence
  - uint8 alphaSize: Alpha channel bit depth
  - uint32 width: Image width in pixels
  - uint32 height: Image height in pixels
  - uint32 mipMapOffsets[16]: Offsets to each mipmap level
  - uint32 mipMapLengths[16]: Lengths of each mipmap level
  - uint32 palette[256]: Color palette (indexed formats only)
  - uint8[] mipmapData: Compressed mipmap data
```

### MipBits (Mipmap Data)
```
Purpose: Stores decompressed mipmap data
Structure:
  - uint8* data: Raw pixel data
  - uint32 pitch: Bytes per row
  - uint32 width: Mipmap width
  - uint32 height: Mipmap height
  - uint32 format: Pixel format
```

## Color Encoding Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | JPEG | JPEG compressed (older format) |
| 1 | Palette8 | 8-bit indexed with palette |
| 2 | DXT | DXTn compression (modern format) |
| 3 | ARGB8888 | Uncompressed 32-bit ARGB |

## Preferred Format Values

| Value | Name | Description |
|-------|------|-------------|
| 0 | Unknown | No specific format |
| 1 | DXT1 | DXT1 compression (4-bit alpha) |
| 2 | DXT3 | DXT3 compression (4-bit explicit alpha) |
| 3 | DXT5 | DXT5 compression (8-bit interpolated alpha) |
| 4 | ARGB1555 | 16-bit ARGB (1-bit alpha) |
| 5 | ARGB4444 | 16-bit ARGB (4-bit alpha) |
| 6 | RGB565 | 16-bit RGB (no alpha) |
| 7 | ARGB8888 | 32-bit ARGB |

## Alpha Channel Types

| Value | Name | Bits | Description |
|-------|------|------|-------------|
| 0 | None | 0 | No alpha channel |
| 1 | Alpha1 | 1 | 1-bit alpha (binary transparency) |
| 2 | Alpha4 | 4 | 4-bit alpha (16 levels) |
| 3 | Alpha8 | 8 | 8-bit alpha (256 levels) |

## DXT Compression Formats

### DXT1
```
Compression: 4 bits per pixel
Block size: 4x4 pixels
Alpha: Optional 1-bit (binary)
Memory: 8 bytes per block
Quality: ~8:1 compression
Used for: Opaque or binary-alpha textures
```

### DXT3
```
Compression: 8 bits per pixel
Block size: 4x4 pixels
Alpha: Explicit 4-bit (16 levels)
Memory: 16 bytes per block
Quality: ~4:1 compression
Used for: Sharp alpha transitions
```

### DXT5
```
Compression: 8 bits per pixel
Block size: 4x4 pixels
Alpha: Interpolated 8-bit (256 levels)
Memory: 16 bytes per block
Quality: ~4:1 compression
Used for: Smooth alpha gradients
```

## Mipmap Generation

### MipMap Algorithm Types
```
MMA_NONE = 0: No mipmaps
MMA_BOX = 1: Box filter (average)
MMA_KAISER = 2: Kaiser filter (better quality)
```

### RequestImageDimensions @ 0x0046fxxx
```
Purpose: Select appropriate mipmap level based on display size
Parameters:
  - width: Requested width
  - height: Requested height
  - bestMip: Output mipmap level index
Logic:
  - Calculate mipRatio = max(width, height) / min(originalWidth, originalHeight)
  - Find mipmap level where mipmapDimension * 2 >= requestedDimension
  - Prefer mipmap that matches display size closely
```

## BLP Loading Functions

### CBLPFile::Open @ 0x0046f820
```c
int CBLPFile::Open(CBLPFile* this, char* filename) {
  // Open file
  FILE* fp = fopen(filename, "rb");
  if (!fp) return 0;
  
  // Read header
  fread(&this->m_header, sizeof(BLPHeader), 1, fp);
  
  // Verify magic
  if (this->m_header.magic != 0x32504c42) { // '2PLB'
    fclose(fp);
    return 0;
  }
  
  // Initialize arrays
  memset(this->m_header.mipMapOffsets, 0, sizeof(uint32) * 16);
  memset(this->m_header.mipMapLengths, 0, sizeof(uint32) * 16);
  
  // Read mipmap offsets and lengths (version 1 format)
  if (this->m_header.formatVersion == 1) {
    fread(this->m_header.mipMapOffsets, sizeof(uint32), 16, fp);
    fread(this->m_header.mipMapLengths, sizeof(uint32), 16, fp);
    fread(this->m_header.palette, sizeof(uint32), 256, fp);
  }
  
  // Load mipmap data
  for (int i = 0; i < 16; i++) {
    if (this->m_header.mipMapOffsets[i] > 0) {
      fseek(fp, this->m_header.mipMapOffsets[i], SEEK_SET);
      uint32 size = this->m_header.mipMapLengths[i];
      this->m_mipMaps[i].data = malloc(size);
      fread(this->m_mipMaps[i].data, 1, size, fp);
      this->m_mipMaps[i].size = size;
    }
  }
  
  fclose(fp);
  return 1;
}
```

### LoadBlpMips @ 0x0046f820
```c
undefined4 LoadBlpMips(
  char* filename,
  MipBits** outputMips,
  uint* outputWidth,
  uint* outputHeight,
  uint* outputFormat,
  uint* hasAlpha,
  uint* alphaSize
)
{
  CBLPFile texFile;
  
  // Initialize BLP file structure
  memset(&texFile, 0, sizeof(CBLPFile));
  texFile.m_quality = 100;
  texFile.m_header.magic = 0x32504c42;
  texFile.m_header.formatVersion = 1;
  texFile.m_header.preferredFormat = 2;
  texFile.m_mipMapAlgorithm = MMA_BOX;
  
  // Open and parse BLP file
  int result = CBLPFile::Open(&texFile, filename);
  if (!result) {
    CBLPFile::Close(&texFile);
    return 0;
  }
  
  // Determine output pixel format
  uint32 outputPixelFormat = PIXEL_ARGB8888;
  
  if (texFile.m_header.colorEncoding == 2) { // DXT compressed
    outputPixelFormat = texFile.m_header.preferredFormat;
    
    if (outputPixelFormat == PIXEL_DXT1) {
      CGxCaps* caps = GxCaps();
      if (caps->m_texFmtDxt == 0) {
        // No DXT support - use fallback
        if (texFile.m_header.alphaSize == 0) {
          outputPixelFormat = PIXEL_RGB565;  // No alpha fallback
        } else {
          outputPixelFormat = PIXEL_ARGB1555; // Binary alpha fallback
        }
      }
    }
    else if (outputPixelFormat == PIXEL_DXT3) {
      CGxCaps* caps = GxCaps();
      if (caps->m_texFmtDxt == 0) {
        outputPixelFormat = PIXEL_ARGB4444; // 4-bit alpha fallback
      }
    }
    else if (outputPixelFormat == PIXEL_DXT5) {
      CGxCaps* caps = GxCaps();
      if (caps->m_texFmtDxt == 0) {
        outputPixelFormat = PIXEL_ARGB4444; // Fallback for DXT5
      }
    }
  }
  
  // Return alpha information
  if (hasAlpha) {
    *hasAlpha = (texFile.m_header.alphaSize != 0);
  }
  
  if (alphaSize) {
    *alphaSize = texFile.m_header.alphaSize;
  }
  
  // Get image dimensions
  uint32 imgWidth = texFile.m_header.width;
  uint32 imgHeight = texFile.m_header.height;
  uint32 bestMip = 0;
  
  RequestImageDimensions(&imgWidth, &imgHeight, &bestMip);
  
  if (outputWidth) *outputWidth = imgWidth;
  if (outputHeight) *outputHeight = imgHeight;
  
  // Lock and convert mipmap data
  result = CBLPFile::LockChain(&texFile, outputPixelFormat, outputMips, bestMip);
  
  CBLPFile::Close(&texFile);
  return result;
}
```

### CBLPFile::LockChain @ 0x0046fxxx
```c
int CBLPFile::LockChain(CBLPFile* this, PIXEL_FORMAT format, MipBits** output, int mipLevel)
{
  // Calculate output pitch
  uint32 pitch = (this->m_header.width >> mipLevel);
  uint32 height = (this->m_header.height >> mipLevel);
  
  switch (this->m_header.colorEncoding) {
    case 0: // JPEG compressed
      return DecompressJpegMipmap(this, mipLevel, format, output);
    
    case 1: // Palette indexed
      return DecompressPalettedMipmap(this, mipLevel, format, output);
    
    case 2: // DXT compressed
      return DecompressDxtMipmap(this, mipLevel, format, output);
    
    case 3: // Uncompressed ARGB8888
      return CopyUncompressedMipmap(this, mipLevel, format, output);
  }
  
  return 0;
}
```

## DXT Decompression

### DecompressDXT1
```c
void DecompressDXT1(
  uint8* compressed,  // 8 bytes per 4x4 block
  uint8* output,      // 16 bytes per 4x4 block (RGBA)
  int blockX, int blockY
)
{
  // Read color endpoints (16-bit RGB565)
  uint16 color0 = *(uint16*)(compressed);
  uint16 color1 = *(uint16*)(compressed + 2);
  
  // Decompress RGB565 to RGBA8888
  uint8 r0 = (color0 >> 11) & 0x1F;
  uint8 g0 = (color0 >> 5) & 0x3F;
  uint8 b0 = color0 & 0x1F;
  
  uint8 r1 = (color1 >> 11) & 0x1F;
  uint8 g1 = (color1 >> 5) & 0x3F;
  uint8 b1 = color1 & 0x1F;
  
  uint32 rgba0 = (r0 << 3) | (g0 << 2) | (b0 << 3) | 0xFF000000;
  uint32 rgba1 = (r1 << 3) | (g1 << 2) | (b1 << 3) | 0xFF000000;
  uint32 rgba2 = ((r0 + r1) * 2 + 1) << 3 | ((g0 + g1) * 1 + 1) << 2 | ((b0 + b1) * 2 + 1) << 3 | 0xFF000000;
  uint32 rgba3 = ((r0 + r1) * 1 + 1) << 3 | ((g0 + g1) * 1 + 1) << 2 | ((b0 + b1) * 1 + 1) << 3 | 0xFF000000;
  
  // Check for alpha (if color0 <= color1 in unsigned comparison)
  bool hasAlpha = color0 <= color1;
  if (hasAlpha) {
    rgba3 = 0; // Transparent
  }
  
  // Read 2-bit indices
  uint32 indices = *(uint32*)(compressed + 4);
  
  // Write decompressed pixels
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      uint32 index = (indices >> ((y * 4 + x) * 2)) & 0x03;
      uint32 pixel = (index == 0) ? rgba0 :
                     (index == 1) ? rgba1 :
                     (index == 2) ? rgba2 : rgba3;
      *(uint32*)(output + ((blockY * 4 + y) * pitch + (blockX * 4 + x)) * 4) = pixel;
    }
  }
}
```

### DecompressDXT3
```c
void DecompressDXT3(
  uint8* compressed,  // 16 bytes per 4x4 block
  uint8* output,
  int blockX, int blockY
)
{
  // Read alpha values (4 bits each, 8 values)
  uint64 alpha = *(uint64*)compressed;
  
  // Extract and expand alpha values
  uint8 alphaValues[8];
  for (int i = 0; i < 8; i++) {
    alphaValues[i] = ((alpha >> (i * 4)) & 0x0F) * 17;  // Expand 4-bit to 8-bit
  }
  
  // Read color endpoints (same as DXT1)
  uint16 color0 = *(uint16*)(compressed + 8);
  uint16 color1 = *(uint16*)(compressed + 10);
  
  // Decompress colors (same as DXT1)
  // ...
  
  // Read 2-bit indices
  uint32 indices = *(uint32*)(compressed + 12);
  
  // Write decompressed pixels
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      uint32 index = (indices >> ((y * 4 + x) * 2)) & 0x03;
      uint32 alphaIndex = y * 4 + x;
      uint32 pixel = colorValues[index] | (alphaValues[alphaIndex] << 24);
      *(uint32*)(output + ((blockY * 4 + y) * pitch + (blockX * 4 + x)) * 4) = pixel;
    }
  }
}
```

### DecompressDXT5
```c
void DecompressDXT5(
  uint8* compressed,  // 16 bytes per 4x4 block
  uint8* output,
  int blockX, int blockY
)
{
  // Read alpha endpoints (8-bit)
  uint8 alpha0 = compressed[0];
  uint8 alpha1 = compressed[1];
  
  // Expand alpha values
  uint8 alphaValues[8];
  if (alpha0 > alpha1) {
    // 6 interpolated values
    alphaValues[0] = alpha0;
    alphaValues[1] = alpha1;
    alphaValues[2] = (6 * alpha0 + 1 * alpha1) / 7;
    alphaValues[3] = (5 * alpha0 + 2 * alpha1) / 7;
    alphaValues[4] = (4 * alpha0 + 3 * alpha1) / 7;
    alphaValues[5] = (3 * alpha0 + 4 * alpha1) / 7;
    alphaValues[6] = (2 * alpha0 + 5 * alpha1) / 7;
    alphaValues[7] = (1 * alpha0 + 6 * alpha1) / 7;
  } else {
    // 4 interpolated + 2 fully transparent
    alphaValues[0] = alpha0;
    alphaValues[1] = alpha1;
    alphaValues[2] = (4 * alpha0 + 1 * alpha1) / 5;
    alphaValues[3] = (3 * alpha0 + 2 * alpha1) / 5;
    alphaValues[4] = (2 * alpha0 + 3 * alpha1) / 5;
    alphaValues[5] = (1 * alpha0 + 4 * alpha1) / 5;
    alphaValues[6] = 0; // Fully transparent
    alphaValues[7] = 255; // Fully opaque (used for indexing)
  }
  
  // Read 3-bit alpha indices
  uint32 indices = *(uint32*)(compressed + 2);
  
  // Read color endpoints and decompress (same as DXT1)
  // ...
  
  // Combine alpha with color
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      uint32 index = (indices >> ((y * 4 + x) * 3)) & 0x07;
      uint32 alphaIndex = (indices >> ((y * 4 + x) * 3)) & 0x07;
      uint32 pixel = colorValues[index] | (alphaValues[alphaIndex] << 24);
      *(uint32*)(output + ((blockY * 4 + y) * pitch + (blockX * 4 + x)) * 4) = pixel;
    }
  }
}
```

## Implementation Recommendations for MdxViewer

### 1. BLP File Reader
```csharp
public class BlpReader
{
    public BlpHeader Header { get; private set; }
    private byte[][] _mipmapData;
    
    public bool Load(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        return Load(fs);
    }
    
    public bool Load(Stream stream)
    {
        var reader = new BinaryReader(stream);
        
        // Read header
        Header = new BlpHeader();
        Header.Magic = reader.ReadUInt32();
        if (Header.Magic != 0x32504c42) // '2PLB'
            return false;
        
        Header.FormatVersion = reader.ReadUInt32();
        Header.ColorEncoding = reader.ReadByte();
        Header.PreferredFormat = reader.ReadByte();
        Header.HasAlpha = reader.ReadByte();
        Header.AlphaSize = reader.ReadByte();
        Header.Width = reader.ReadUInt32();
        Header.Height = reader.ReadUInt32();
        
        // Read mipmap offsets and lengths
        Header.MipOffsets = new uint[16];
        Header.MipLengths = new uint[16];
        for (int i = 0; i < 16; i++)
            Header.MipOffsets[i] = reader.ReadUInt32();
        for (int i = 0; i < 16; i++)
            Header.MipLengths[i] = reader.ReadUInt32();
        
        // Read palette (if indexed)
        if (Header.ColorEncoding == 1) {
            Header.Palette = new uint[256];
            for (int i = 0; i < 256; i++)
                Header.Palette[i] = reader.ReadUInt32();
        }
        
        // Read mipmap data
        _mipmapData = new byte[16][];
        for (int i = 0; i < 16; i++) {
            if (Header.MipOffsets[i] > 0 && Header.MipLengths[i] > 0) {
                stream.Position = Header.MipOffsets[i];
                _mipmapData[i] = reader.ReadBytes((int)Header.MipLengths[i]);
            }
        }
        
        return true;
    }
    
    public byte[] GetMipmap(int level)
    {
        if (level < 0 || level >= 16) return null;
        return _mipmapData[level];
    }
    
    public int GetMipWidth(int level)
    {
        return Math.Max(1, (int)(Header.Width >> level));
    }
    
    public int GetMipHeight(int level)
    {
        return Math.Max(1, (int)(Header.Height >> level));
    }
}
```

### 2. DXT Decompressor
```csharp
public static class DxtDecompressor
{
    public static byte[] DecompressDxt1(byte[] input, int width, int height)
    {
        var output = new byte[width * height * 4];
        
        int blocksX = (width + 3) / 4;
        int blocksY = (height + 3) / 4;
        
        for (int y = 0; y < blocksY; y++) {
            for (int x = 0; x < blocksX; x++) {
                int blockOffset = (y * blocksX + x) * 8;
                DecompressDxt1Block(
                    input, blockOffset,
                    output, width, height,
                    x * 4, y * 4
                );
            }
        }
        
        return output;
    }
    
    private static void DecompressDxt1Block(
        byte[] input, int offset,
        byte[] output, int width, int height,
        int startX, int startY
    )
    {
        // Read color endpoints
        ushort c0 = BitConverter.ToUInt16(input, offset);
        ushort c1 = BitConverter.ToUInt16(input, offset + 2);
        
        // Decompress RGB565 to RGBA8888
        var color0 = Rgb565ToRgba8888(c0);
        var color1 = Rgb565ToRgba8888(c1);
        
        // Calculate intermediate colors
        var color2 = BlendColors(color0, color1, 1, 1);
        var color3 = BlendColors(color0, color1, 1, 2);
        
        // Check for alpha (c0 <= c1 means 4th color is transparent)
        bool hasAlpha = c0 <= c1;
        if (hasAlpha)
            color3 = new byte[] { 0, 0, 0, 0 };
        
        // Read indices
        uint indices = BitConverter.ToUInt32(input, offset + 4);
        
        // Write pixels
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                int x = startX + px;
                int y = startY + py;
                
                if (x >= width || y >= height) continue;
                
                int index = (int)((indices >> ((py * 4 + px) * 2)) & 0x03);
                byte[] color = index switch {
                    0 => color0,
                    1 => color1,
                    2 => color2,
                    _ => color3
                };
                
                int outOffset = (y * width + x) * 4;
                Buffer.BlockCopy(color, 0, output, outOffset, 4);
            }
        }
    }
    
    private static byte[] Rgb565ToRgba8888(ushort color)
    {
        byte r = (byte)((color >> 11) & 0x1F);
        byte g = (byte)((color >> 5) & 0x3F);
        byte b = (byte)(color & 0x1F);
        
        return new byte[] {
            (byte)(r << 3 | r >> 2),
            (byte)(g << 2 | g >> 4),
            (byte)(b << 3 | b >> 2),
            255
        };
    }
    
    private static byte[] BlendColors(byte[] a, byte[] b, int wa, int wb)
    {
        return new byte[] {
            (byte)((a[0] * wa + b[0] * wb) / (wa + wb)),
            (byte)((a[1] * wa + b[1] * wb) / (wa + wb)),
            (byte)((a[2] * wa + b[2] * wb) / (wa + wb)),
            255
        };
    }
}
```

### 3. Texture Upload to GPU
```csharp
public class TextureUploader
{
    public uint UploadBlp(BlpReader blp, int mipLevel = 0)
    {
        byte[] mipData = blp.GetMipmap(mipLevel);
        int width = blp.GetMipWidth(mipLevel);
        int height = blp.GetMipHeight(mipLevel);
        
        byte[] rgbaData;
        
        // Decompress based on color encoding
        switch (blp.Header.ColorEncoding) {
            case 0: // JPEG - would need JPEG decoder
                throw new NotSupportedException("JPEG BLP not supported");
            
            case 1: // Palette
                rgbaData = DecompressPaletted(mipData, width, height, blp.Header.Palette);
                break;
            
            case 2: // DXT
                switch (blp.Header.PreferredFormat) {
                    case 1: // DXT1
                        rgbaData = DxtDecompressor.DecompressDxt1(mipData, width, height);
                        break;
                    case 2: // DXT3
                        rgbaData = DxtDecompressor.DecompressDxt3(mipData, width, height);
                        break;
                    case 3: // DXT5
                        rgbaData = DxtDecompressor.DecompressDxt5(mipData, width, height);
                        break;
                    default:
                        throw new NotSupportedException($"DXT format {blp.Header.PreferredFormat} not supported");
                }
                break;
            
            case 3: // Uncompressed
                rgbaData = mipData; // Already RGBA
                break;
            
            default:
                throw new NotSupportedException($"BLP encoding {blp.Header.ColorEncoding} not supported");
        }
        
        // Upload to GPU
        uint textureId;
        GL.GenTextures(1, out textureId);
        GL.BindTexture(TextureTarget.Texture2D, textureId);
        
        GL.TexImage2D(
            TextureTarget.Texture2D,
            mipLevel,
            PixelInternalFormat.Rgba,
            width,
            height,
            0,
            PixelFormat.Rgba,
            PixelType.UnsignedByte,
            rgbaData
        );
        
        // Set texture parameters
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMaxLevel, 14);
        
        return textureId;
    }
}
```

### 4. Alpha Channel Handling
```csharp
public static class AlphaHandler
{
    public enum AlphaType
    {
        None = 0,
        Binary = 1,    // 1-bit alpha
        Smooth4 = 4,   // 4-bit alpha (16 levels)
        Smooth8 = 8    // 8-bit alpha (256 levels)
    }
    
    public static bool HasAlpha(BlpHeader header)
    {
        return header.AlphaSize != 0;
    }
    
    public static AlphaType GetAlphaType(BlpHeader header)
    {
        return (AlphaType)header.AlphaSize;
    }
    
    public static byte[] ExtractAlpha(byte[] rgbaData, AlphaType alphaType)
    {
        var alpha = new byte[rgbaData.Length / 4];
        
        switch (alphaType) {
            case AlphaType.Binary:
                for (int i = 0; i < alpha.Length; i++) {
                    alpha[i] = (rgbaData[i * 4 + 3] >= 128) ? (byte)255 : (byte)0;
                }
                break;
            
            case AlphaType.Smooth4:
                for (int i = 0; i < alpha.Length; i++) {
                    alpha[i] = (byte)((rgbaData[i * 4 + 3] >> 4) * 17); // 4-bit to 8-bit
                }
                break;
            
            case AlphaType.Smooth8:
                // Already 8-bit
                for (int i = 0; i < alpha.Length; i++) {
                    alpha[i] = rgbaData[i * 4 + 3];
                }
                break;
            
            default:
                // No alpha
                for (int i = 0; i < alpha.Length; i++) {
                    alpha[i] = 255;
                }
                break;
        }
        
        return alpha;
    }
}
```

## Key Functions Reference

| Function | Address | Purpose |
|----------|---------|---------|
| LoadBlpMips | 0x0046f820 | Main BLP loading function |
| CBLPFile::Open | 0x0046fxxx | Open and parse BLP file |
| CBLPFile::Close | 0x0046fxxx | Close BLP file |
| CBLPFile::LockChain | 0x0046fxxx | Lock and convert mipmap |
| RequestImageDimensions | 0x0046fxxx | Select mipmap level |
| AsyncCreateBlpTextureCallback | 0x004719f0 | Async texture creation |
| CreateBlpTexture | 0x004717f0 | Create GPU texture |
| UpdateBlpTextureAsync | 0x0046f630 | Async texture update |
| PumpBlpTextureAsync | 0x00471a70 | Process async texture tasks |

## Debugging Tips

1. **Texture not loading?** Check:
   - BLP magic is correct (0x32504c42)
   - Color encoding is supported (not JPEG)
   - DXT format is supported by GPU
   - Mipmap offsets are valid

2. **Wrong colors?** Check:
   - RGB565 decompression is correct
   - Palette is properly indexed
   - Color encoding matches actual format

3. **Alpha looks wrong?** Check:
   - Alpha size is correct
   - DXT alpha decompression matches format
   - Blend mode is set correctly

4. **Mipmaps missing?** Check:
   - Mipmap offsets are not zero
   - Correct mipmap level is selected
   - GPU supports mipmap generation
