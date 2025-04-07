# TEX (Texture Blob) Format

## Overview
TEX files store low-resolution textures for efficient distant rendering in World of Warcraft. Each continent has a corresponding TEX file (e.g., Azeroth.tex) that contains optimized versions of textures for quick loading and reduced memory usage.

## File Structure

### Chunk-Based Format
The TEX format uses a chunked structure with required chunk ordering.

### TXVR (Version) Chunk
```cpp
struct TXVR {
    uint32_t version;  // 0 (pre-8.1.5) or 1 (post-8.1.5)
};
```

### TXBT (Blob Texture) Chunk
```cpp
struct SBlobTexture {
    union {
        uint32_t filenameOffset;  // Pre-8.1.5: Offset in TXFN chunk
        uint32_t fileId;         // Post-8.1.5: Direct FileDataID reference
    };
    uint32_t txmdOffset;     // Offset to texture data in TXMD chunk
    uint8_t sizeX;          // Texture width (8,16,32,64)
    uint8_t sizeY;          // Texture height (8,16,32,64)
    uint8_t flags;          // Bit 0-6: numLevels, Bit 7: loaded
    uint8_t format;         // Bits 0-3: dxt_type, Bits 4-7: flags
};
```

### TXFN (Filenames) Chunk
```cpp
// Only present in pre-8.1.5 versions
struct TXFN {
    char filenames[];  // Zero-terminated strings, no file extensions
};
```

### TXMD (Texture Data) Chunks
```cpp
struct TXMD {
    uint8_t textureData[];  // DXT compressed texture data
};
```

## Format Details

### Texture Dimensions
- Width/Height: 8, 16, 32, or 64 pixels
- Special case: Width = Height * 6 (cubemaps)
- Must be power of 2 values
- Maximum size is 64x64

### Compression Types
```cpp
enum DXTType {
    DXT1 = 0,  // No alpha or 1-bit alpha
    DXT3 = 1,  // 4-bit alpha
    DXT5 = 2   // 8-bit alpha
};
```

### Format Flags
```cpp
struct FormatFlags {
    uint4_t dxt_type : 3;     // DXT compression type
    uint4_t alpha_dxt1 : 1;   // DXT1 with alpha
    uint4_t reserved : 1;     // Unused
    uint4_t dxt3_flag : 1;    // Set for DXT3
    uint4_t alpha_flag : 1;   // Set for DXT3/DXT5
    uint4_t unused : 1;       // Unused
};
```

## Implementation Notes

### Version Differences
1. **Pre-8.1.5 (Version 0)**
   - Uses TXFN chunk for filenames
   - Filename offset-based lookup
   - Three-chunk structure

2. **Post-8.1.5 (Version 1)**
   - No TXFN chunk
   - Direct FileDataID references
   - Two-chunk structure

### Mipmap Management
1. **Level Count**
   - Maximum 7 levels
   - Stored in flags field
   - Progressive size reduction
   - Minimum size is 1x1

2. **Data Organization**
   - Sequential mipmap storage
   - DXT block alignment
   - Size-based level access
   - Efficient level selection

### Compression Details
1. **DXT1 Format**
   - 8 bytes per 4x4 block
   - Optional 1-bit alpha
   - RGB565 color encoding
   - Best for opaque textures

2. **DXT3/DXT5 Format**
   - 16 bytes per 4x4 block
   - Explicit alpha encoding
   - Higher quality alpha
   - Used for transparent textures

### Best Practices
1. **Loading Strategy**
   - Check version first
   - Handle both filename methods
   - Validate texture dimensions
   - Implement proper mipmap handling

2. **Error Handling**
   - Verify chunk order
   - Validate offsets
   - Check compression types
   - Handle missing data

3. **Memory Management**
   - Efficient texture loading
   - Proper decompression
   - Mipmap caching
   - Resource cleanup

### Usage Context
1. **Distant Rendering**
   - Low-resolution fallbacks
   - Quick initial loading
   - Memory optimization
   - Performance improvement

2. **Texture Management**
   - Continent-specific files
   - Special effect textures
   - Class-specific resources
   - Dynamic resource loading

### Performance Considerations
1. **Loading Optimization**
   - Batch texture loading
   - Asynchronous decompression
   - Efficient mipmap selection
   - Memory-mapped access

2. **Resource Management**
   - Texture pooling
   - Reference counting
   - Cache optimization
   - Memory budgeting 