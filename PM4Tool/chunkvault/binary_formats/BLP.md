# BLP (Blizzard Picture) Format

## Overview
BLP is Blizzard's proprietary texture format that stores textures with precalculated mipmaps. The format supports both palettized and compressed (DXT) storage methods with various alpha channel configurations.

## File Structure

### Header
```cpp
struct BLPHeader {
    uint32_t magic;              // Always 'BLP2'
    uint32_t version;           // Always 1
    uint8_t colorEncoding;      // 1=palettized, 2=DXT, 3=ARGB8888, 4=ARGB8888
    uint8_t alphaSize;         // 0, 1, or 8
    uint8_t preferredFormat;   // Compression format hints
    uint8_t hasMips;          // 0=no mipmaps, 1=has mipmaps
    uint32_t width;           // Power of 2
    uint32_t height;          // Power of 2
    uint32_t mipOffsets[16];  // Offsets for each mipmap level
    uint32_t mipSizes[16];    // Sizes for each mipmap level
    union {
        struct {
            uint8_t palette[256][4];  // BGRA palette for palettized textures
        };
        struct {
            uint32_t headerSize;     // For JPEG (unused)
            uint8_t headerData[1020]; // For JPEG (unused)
        };
    };
};
```

## Color Encoding Types
```cpp
enum BLPColorEncoding : uint8_t {
    COLOR_JPEG = 0,      // Not supported
    COLOR_PALETTE = 1,   // Palettized format
    COLOR_DXT = 2,       // DXT compression
    COLOR_ARGB8888 = 3,  // Uncompressed ARGB
    COLOR_ARGB8888_2 = 4 // Alternative ARGB
};
```

## Pixel Formats
```cpp
enum BLPPixelFormat : uint8_t {
    PIXEL_DXT1 = 0,     // DXT1 compression
    PIXEL_DXT3 = 1,     // DXT3 compression
    PIXEL_ARGB8888 = 2, // Uncompressed ARGB
    PIXEL_ARGB1555 = 3, // 16-bit color
    PIXEL_ARGB4444 = 4, // 16-bit color with alpha
    PIXEL_RGB565 = 5,   // 16-bit color no alpha
    PIXEL_A8 = 6,       // Alpha only
    PIXEL_DXT5 = 7,     // DXT5 compression
    PIXEL_UNSPECIFIED = 8,
    PIXEL_ARGB2565 = 9,
    PIXEL_BC5 = 11      // BC5/3Dc/ATI2N compression
};
```

## Implementation Notes

### Compression Types
1. **Palettized**
   - Uses 256-color palette
   - Optional 0-bit, 1-bit, or 8-bit alpha
   - Index data followed by alpha data
   - Common in character textures

2. **DXT Compressed**
   - DXT1 for no/1-bit alpha
   - DXT3 for 4-bit alpha
   - DXT5 for 8-bit alpha
   - Used for most environment textures

3. **Uncompressed**
   - Direct ARGB8888 storage
   - Used in special cases
   - Found in terrain cube maps

### Mipmap Handling
1. **Size Calculation**
   ```cpp
   // DXT1 size calculation
   size = ((width + 3) / 4) * ((height + 3) / 4) * 8;
   
   // DXT3/DXT5 size calculation
   size = ((width + 3) / 4) * ((height + 3) / 4) * 16;
   ```

2. **Validation**
   - Verify power of 2 dimensions
   - Check mipmap level count
   - Validate offset/size pairs
   - Handle incorrect size values

### Best Practices
1. **Loading Strategy**
   - Validate header before reading data
   - Handle each compression type separately
   - Implement proper mipmap chain loading
   - Support all alpha bit depths

2. **Error Handling**
   - Check magic number
   - Validate version
   - Verify compression format
   - Handle unsupported features

3. **Memory Management**
   - Efficient mipmap loading
   - Proper decompression buffers
   - Handle large texture sizes
   - Cache management considerations

### Common Uses
1. **Character Textures**
   - Palettized format
   - Various alpha depths
   - Multiple mipmap levels

2. **Environment Textures**
   - DXT compression
   - Minimal alpha usage
   - Full mipmap chains

3. **Interface Elements**
   - Mixed formats
   - Often without mipmaps
   - Alpha channel common

### Version History
- Original BLP1 format (pre-WoW)
- BLP2 format (WoW release)
- Extended formats (post-Cataclysm)
- Modern format additions (Shadowlands+) 