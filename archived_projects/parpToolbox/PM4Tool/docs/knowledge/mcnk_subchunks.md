# MCNK Sub-Chunks Documentation

## Overview
MCNK (Map Chunk) is the main chunk type used in ADT files, and its format is also found in Alpha WDT files. Each MCNK chunk contains multiple sub-chunks that define terrain, textures, lighting, and other map features for a specific map chunk.

## MCNK Header
The MCNK chunk begins with a header structure containing:
- Chunk flags
- Indices for location in the map grid (X, Y)
- Number of doodad references
- Number of map object references
- Height information
- Area ID
- Number of layers

## Core Sub-Chunks

### MCVT
- **Purpose**: Terrain height map
- **Content**: 9×9 + 8×8 = 145 float values defining the height of each vertex in the chunk
- **Structure**: Array of 32-bit floats representing height values

### MCCV (Optional)
- **Purpose**: Vertex colors for terrain shading
- **Content**: RGB colors for each vertex (145 vertices)
- **Structure**: Array of RGB triplets (3 bytes per vertex)

### MCNR
- **Purpose**: Normal vectors for terrain lighting
- **Content**: Normal data for each vertex (145 vertices)
- **Structure**: Array of normal vectors (3 bytes per vertex)

### MCLY
- **Purpose**: Texture layer definitions
- **Content**: Information about each texture layer applied to the chunk
- **Structure**: Array of layer structures containing:
  - Texture ID (index into MTEX chunk)
  - Flags (e.g., animation, alpha map usage)
  - Offset to alpha map in MCAL chunk
  - Additional effect information

### MCRF
- **Purpose**: References to doodads and WMOs
- **Content**: Indices into MDDF and MODF chunks for objects in this chunk
- **Structure**: Array of uint32 indices

### MCSH (Optional)
- **Purpose**: Shadow map for the chunk
- **Content**: 64×64 bit array for shadows
- **Structure**: 512 bytes of shadow data

### MCAL
- **Purpose**: Alpha maps for texture blending
- **Content**: Alpha maps for each texture layer (except the base layer)
- **Structure**: Compressed or uncompressed alpha data for each layer

### MCLQ (Pre-WotLK)
- **Purpose**: Liquid data (water, lava, etc.)
- **Content**: Liquid type, height, and flags
- **Structure**: Liquid vertex heights and flags
- **Note**: Replaced by MH2O chunk in WotLK and later

### MCSE (Optional)
- **Purpose**: Sound emitters
- **Content**: References to sound effects in this chunk
- **Structure**: Array of sound emitter data

## Alpha WDT Considerations
- Alpha WDT files may use a subset of these sub-chunks
- The implementation may differ slightly from modern versions
- Focus should be on core sub-chunks (MCVT, MCNR, MCLY, MCAL)
- Flag values and structure sizes might be different

## Internal Structures

### Layer Structure (MCLY)
```
struct MCLY_Entry {
    uint32_t textureID;      // Index into MTEX chunk
    uint32_t flags;          // Flags for the layer
    uint32_t offsetInMCAL;   // Offset to alpha map
    uint32_t effectID;       // Used for additional effects
}
```

### Key Layer Flags
- `FLAG_ANIMATE_45` (0x1) - Animate texture at 45 degrees
- `FLAG_ANIMATE_90` (0x2) - Animate texture at 90 degrees
- `FLAG_ANIMATE_180` (0x4) - Animate texture at 180 degrees
- `FLAG_ANIM_FAST` (0x8) - Faster animation
- `FLAG_ANIM_FASTER` (0x10) - Even faster animation
- `FLAG_ANIM_FASTEST` (0x20) - Fastest animation
- `FLAG_ALPHA` (0x40) - Uses alpha map
- `FLAG_ALPHA_COMPRESSED` (0x80) - Alpha map is compressed
- `FLAG_HOLE` (0x10000) - Marks holes in the terrain

## MCAL (Alpha Map) Chunk

### Overview
The MCAL chunk contains alpha map data that defines how textures blend together in a map chunk. Each MCNK can have multiple alpha maps, one for each texture layer beyond the base layer.

### Structure
- **Signature**: "MCAL"
- **Format**: Raw binary data, no standard chunk header
- **Size**: Variable, depends on format and compression

### Alpha Map Formats

#### Standard Format (2048 bytes)
- Uses 4 bits per alpha value
- Two values packed per byte
- Total size: 2048 bytes (64×64÷2)
- Values range from 0-15, scaled to 0-255 for rendering

#### Big Alpha Format (4096 bytes)
- Uses 8 bits per alpha value
- One value per byte
- Total size: 4096 bytes (64×64)
- Values range from 0-255

#### Compressed Format (Variable size)
- Uses run-length encoding (RLE)
- Format:
  ```
  For each byte:
    bit 7: Fill flag (0=individual values, 1=repeated value)
    bits 0-6: Count of values
    Following byte(s): Value(s) to use
  ```
- Decompresses to 4096 bytes (64×64)

### Relationship with MCLY
The MCLY (texture layer) chunk contains flags that determine how to interpret the MCAL data:
```csharp
[Flags]
public enum MCLYFlags
{
    UseAlpha = 0x1,          // Layer uses alpha map
    UseBigAlpha = 0x2,       // Use 8-bit alpha values
    CompressedAlpha = 0x4,   // Alpha map is RLE compressed
    // ... other flags ...
}
```

### Data Layout
1. **Grid Structure**:
   - 64×64 grid of alpha values
   - Represents texture blending intensity
   - Last row/column duplicates second-to-last for continuity

2. **Memory Layout**:
   ```
   Standard Format:
   Byte 0: [Alpha0_Low 4 bits][Alpha1_High 4 bits]
   Byte 1: [Alpha2_Low 4 bits][Alpha3_High 4 bits]
   ...

   Big Alpha Format:
   Byte 0: Alpha0 (8 bits)
   Byte 1: Alpha1 (8 bits)
   ...

   Compressed Format:
   Byte 0: [Fill_Flag 1 bit][Count 7 bits]
   Byte 1: Value to repeat or first value
   [Additional values if not fill mode]
   ...
   ```

### Implementation Notes

1. **Reading Standard Format**:
   ```csharp
   // For each byte in the 2048-byte buffer:
   byte value = buffer[i];
   byte alpha1 = (byte)((255 * (value & 0x0f)) / 0x0f);
   byte alpha2 = (byte)((255 * (value & 0xf0)) / 0xf0);
   ```

2. **Reading Big Alpha Format**:
   ```csharp
   // Direct copy of 4096 bytes
   Array.Copy(buffer, alphaMap, 4096);
   ```

3. **Decompressing RLE Format**:
   ```csharp
   while (outputIndex < 4096)
   {
       byte control = input[inputIndex++];
       bool fill = (control & 0x80) != 0;
       int count = control & 0x7F;
       
       if (fill)
       {
           byte value = input[inputIndex++];
           for (int i = 0; i < count; i++)
               output[outputIndex++] = value;
       }
       else
       {
           for (int i = 0; i < count; i++)
               output[outputIndex++] = input[inputIndex++];
       }
   }
   ```

### Error Handling
1. Validate chunk size against expected format
2. Check for buffer overruns during decompression
3. Verify final decompressed size is 4096 bytes
4. Handle missing or corrupt data gracefully
5. Use empty alpha maps (all zeros) as fallback

### Performance Tips
1. Pre-allocate buffers for decompression
2. Use buffer pooling for large ADT files
3. Consider lazy loading for memory efficiency
4. Cache decompressed results when possible
5. Use SIMD operations for bulk processing if available

## Resources
- [wowdev wiki ADT/MCNK](https://wowdev.wiki/ADT/MCNK)
- Warcraft.NET implementation in `/docs/libs/Warcraft.NET/Warcraft.NET/Files/ADT/Chunks/` 