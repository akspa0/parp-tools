# MCSH Shadow Map Format

## Overview

MCSH (MapChunk SHadow) stores baked shadow data for terrain chunks as a compressed bitmap.

**Compressed Size**: 512 bytes (64 × 8)  
**Uncompressed Size**: 4096 bytes (64 × 64)  
**Encoding**: 1 bit per shadow value, unpacked to 8-bit grayscale

---

## Format Specification

### Compressed Format (On-Disk)
```
512 bytes = 64 rows × 8 bytes per row
Each byte contains 8 bits representing 8 shadow values
```

### Uncompressed Format (In-Memory)
```
4096 bytes = 64 × 64 shadow map
Each byte is a grayscale intensity:
  - 0x00 (0)   = Fully shadowed (black)
  - 0x55 (85)  = Fully lit (white)
```

---

## Decoding Algorithm

### From noggit-red (MapChunk.cpp)

```cpp
// Input: compressed_shadow_map (512 bytes)
// Output: _shadow_map (4096 bytes)

uint8_t* p = _shadow_map;
uint8_t* c = compressed_shadow_map;

for (int i = 0; i < 64 * 8; ++i)  // 512 iterations
{
    for (int b = 0x01; b != 0x100; b <<= 1)  // 8 bits per byte
    {
        *p++ = ((*c) & b) ? 85 : 0;  // Unpack bit → 0 or 85
    }
    c++;
}

// Edge fixing (optional - handles last row/column artifacts)
if (!header_flags.flags.do_not_fix_alpha_map)
{
    for (std::size_t i(0); i < 64; ++i)
    {
        _shadow_map[i * 64 + 63] = _shadow_map[i * 64 + 62];  // Right edge
        _shadow_map[63 * 64 + i] = _shadow_map[62 * 64 + i];  // Bottom edge
    }
    _shadow_map[63 * 64 + 63] = _shadow_map[62 * 64 + 62];  // Bottom-right corner
}
```

### Bit Unpacking Explanation

Each compressed byte expands to 8 shadow values:
```
Compressed byte: 0b10110010
                    │││││││└─ Bit 0 → shadow[0] = 0
                    ││││││└── Bit 1 → shadow[1] = 85
                    │││││└─── Bit 2 → shadow[2] = 0
                    ││││└──── Bit 3 → shadow[3] = 0
                    │││└───── Bit 4 → shadow[4] = 85
                    ││└────── Bit 5 → shadow[5] = 85
                    │└─────── Bit 6 → shadow[6] = 0
                    └──────── Bit 7 → shadow[7] = 85
```

---

## C# Implementation

### Decoder

```csharp
public static class McshDecoder
{
    /// <summary>
    /// Decodes compressed MCSH shadow map data.
    /// </summary>
    /// <param name="compressed">512-byte compressed shadow map</param>
    /// <returns>4096-byte uncompressed shadow map (64×64)</returns>
    public static byte[] Decode(byte[] compressed)
    {
        if (compressed.Length != 512)
            throw new ArgumentException("MCSH data must be 512 bytes", nameof(compressed));
        
        var uncompressed = new byte[4096];
        int outputIndex = 0;
        
        // Unpack each byte into 8 shadow values
        foreach (byte compressedByte in compressed)
        {
            for (int bit = 0; bit < 8; bit++)
            {
                int mask = 1 << bit;
                uncompressed[outputIndex++] = (compressedByte & mask) != 0 ? (byte)85 : (byte)0;
            }
        }
        
        return uncompressed;
    }
    
    /// <summary>
    /// Decodes and applies edge fixing.
    /// </summary>
    public static byte[] DecodeWithEdgeFix(byte[] compressed)
    {
        var uncompressed = Decode(compressed);
        
        // Fix last row and column by copying second-to-last
        for (int i = 0; i < 64; i++)
        {
            uncompressed[i * 64 + 63] = uncompressed[i * 64 + 62];  // Right edge
            uncompressed[63 * 64 + i] = uncompressed[62 * 64 + i];  // Bottom edge
        }
        uncompressed[63 * 64 + 63] = uncompressed[62 * 64 + 62];  // Corner
        
        return uncompressed;
    }
    
    /// <summary>
    /// Converts 64×64 byte array to 2D array for easier access.
    /// </summary>
    public static byte[][] To2DArray(byte[] shadowMap)
    {
        var result = new byte[64][];
        for (int y = 0; y < 64; y++)
        {
            result[y] = new byte[64];
            Array.Copy(shadowMap, y * 64, result[y], 0, 64);
        }
        return result;
    }
}
```

### CSV Export

```csharp
// In McnkShadowExtractor.cs
public static List<McnkShadowEntry> ExtractShadows(WdtAlphaScanner wdt)
{
    var entries = new List<McnkShadowEntry>();
    
    foreach (var adtPath in wdt.AdtPaths)
    {
        var (row, col) = ParseTileCoordinates(adtPath);
        
        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);
        
        // Read all MCNK chunks
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                // Find MCSH subchunk within MCNK
                var mcshData = FindMcshInChunk(br, chunkY, chunkX);
                if (mcshData == null) continue;
                
                // Decode shadow map
                var shadowMap = McshDecoder.Decode(mcshData);
                
                // Convert to hex string for CSV (or base64 for compactness)
                var shadowString = BitConverter.ToString(shadowMap).Replace("-", "");
                
                entries.Add(new McnkShadowEntry(
                    wdt.MapName,
                    row,
                    col,
                    chunkY,
                    chunkX,
                    shadowString
                ));
            }
        }
    }
    
    return entries;
}
```

---

## Integration with Existing Tools

### AlphaWdtAnalyzer
Already has `McnkShadowExtractor.cs` - just needs decoder:

```csharp
// Update existing extractor to decode MCSH
private static string ExtractShadowData(BinaryReader br, long mcshOffset, int mcshSize)
{
    if (mcshSize != 512)
        throw new InvalidDataException($"MCSH size should be 512, got {mcshSize}");
    
    br.BaseStream.Position = mcshOffset;
    var compressed = br.ReadBytes(512);
    
    // Decode to 64×64
    var uncompressed = McshDecoder.DecodeWithEdgeFix(compressed);
    
    // Convert to compact string representation
    // Option 1: Hex string (8192 chars)
    return BitConverter.ToString(uncompressed).Replace("-", "");
    
    // Option 2: Base64 (5461 chars - more compact)
    // return Convert.ToBase64String(uncompressed);
    
    // Option 3: Intensity digits 0-9,A-F (4096 chars)
    // return string.Concat(uncompressed.Select(b => (b / 17).ToString("X")));
}
```

### CSV Format Options

**Option 1: Hex String (Current - 8192 chars)**
```csv
MapName,TileRow,TileCol,ChunkY,ChunkX,ShadowHex
Azeroth,30,30,0,0,0055005500550000555555005555000055550055...
```

**Option 2: Base64 (Compact - 5461 chars)**
```csv
MapName,TileRow,TileCol,ChunkY,ChunkX,ShadowBase64
Azeroth,30,30,0,0,AFUAVQBVAAAAVVVVAFVVVQAAVVUAVQ==...
```

**Option 3: Intensity Digits (Human-Readable - 4096 chars)**
```csv
MapName,TileRow,TileCol,ChunkY,ChunkX,ShadowMap
Azeroth,30,30,0,0,0505050000555500555500050550...
```

Recommend **Option 3** - most compact while human-readable. Each character represents shadow intensity:
- `0` = Fully shadowed (0x00)
- `5` = Fully lit (0x55)

---

## Viewer Rendering

### JavaScript Canvas Rendering

```javascript
class ShadowMapLayer {
    async loadTile(row, col) {
        const url = `/overlays/${this.version}/${this.mapName}/shadow_map/tile_r${row}_c${col}.json`;
        const response = await fetch(url);
        if (!response.ok) return null;
        
        const data = await response.json();
        
        // Create canvas for entire tile (1024×1024 for 16×16 chunks)
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        
        // Draw each chunk's shadow map
        data.chunks.forEach(chunk => {
            const chunkPixelX = chunk.chunkX * 64;
            const chunkPixelY = chunk.chunkY * 64;
            
            // Draw 64×64 shadow map
            const imageData = ctx.createImageData(64, 64);
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const index = (y * 64 + x) * 4;
                    const shadow = chunk.shadowData[y][x];  // 0-85
                    
                    // Convert to alpha channel (0 = opaque black, 85 = transparent)
                    imageData.data[index + 0] = 0;      // R
                    imageData.data[index + 1] = 0;      // G
                    imageData.data[index + 2] = 0;      // B
                    imageData.data[index + 3] = 255 - (shadow * 3);  // A (invert)
                }
            }
            
            ctx.putImageData(imageData, chunkPixelX, chunkPixelY);
        });
        
        // Convert canvas to Leaflet ImageOverlay
        const bounds = this.calculateTileBounds(row, col);
        const overlay = L.imageOverlay(canvas.toDataURL(), bounds, {
            opacity: 0.5,
            interactive: false
        });
        
        return overlay;
    }
}
```

---

## Visual Examples

### Shadow Intensity Values
```
0x00 (  0) ████████ Fully shadowed
0x11 ( 17) ███████░ 
0x22 ( 34) ██████░░
0x33 ( 51) █████░░░
0x44 ( 68) ████░░░░
0x55 ( 85) ███░░░░░ Fully lit
```

### CSV Encoding (Intensity Digits)
```
Chunk with diagonal shadow:
0000000000000000
5000000000000000
5500000000000000
5550000000000000
5555000000000000
5555500000000000
5555550000000000
5555555000000000
...

Encoded: 000000000000000050000000000000005500000000000000555000000000000055550...
```

---

## Implementation Priority

1. **Decoder (McshDecoder.cs)** - Simple bit unpacking
2. **CSV Export** - Update McnkShadowExtractor to decode MCSH
3. **Overlay Builder** - Convert CSV → JSON for viewer
4. **UI Layer** - Canvas-based rendering in viewer
5. **Toggle Control** - Add checkbox to sidebar

**Estimated Time**: 2-3 hours (now that decoding is clear!)

---

## Testing

### Validation
1. Extract MCSH from test map
2. Decode to 64×64
3. Verify output is 4096 bytes
4. Check values are only 0x00 or 0x55
5. Visual inspection in viewer should show baked shadows

### Expected Results
- Dark areas under trees, buildings
- Light areas in open fields
- Edge artifacts fixed on last row/column

---

## References

- **noggit-red**: `lib/noggit-red/src/noggit/MapChunk.cpp` lines 235-264
- **WoW File Formats**: https://wowdev.wiki/ADT/v18#MCSH
- **AlphaWdtAnalyzer**: `AlphaWdtAnalyzer.Core/Terrain/McnkShadowExtractor.cs`

---

## Notes

### Why 0 and 85?
- **0x00**: Fully shadowed (black)
- **0x55 (85)**: Fully lit (white, or 85/255 = 33% intensity)
- These are the only two values used in Alpha WoW
- Later expansions may use more gradients

### Edge Fixing
The last row and column are duplicated from row/column 62 to fix rendering artifacts. This is optional but recommended for visual quality.

### Performance
- Decoding is fast (512 bytes → 4096 bytes)
- Canvas rendering may be slow for many tiles
- Consider: Only render visible tiles, use image caching
