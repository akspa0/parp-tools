# C004: AHDR

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Terrain file header for ADT v23 format. Contains global information about the terrain tile and offsets to other chunks.

## Structure
```csharp
struct AHDR
{
    uint32 flags;                // Global flags for this terrain tile
    uint32 mcnkOffset;           // Offset to the ACNK chunks
    uint32 mtexOffset;           // Offset to the ATEX chunk (textures)
    uint32 mmdxOffset;           // Offset to the ADOO chunk (doodads)
    uint32 mmidOffset;           // Offset to the model indices
    uint32 mwmoOffset;           // Offset to the WMO filenames
    uint32 mwidOffset;           // Offset to the WMO indices
    uint32 mddfOffset;           // Offset to the doodad placement info
    uint32 modfOffset;           // Offset to the WMO placement info
    uint32 mtdaOffset;           // Offset to liquid/fog terrain data (WoD addition)
    uint32 fogDataOffset;        // Offset to fog data (WoD addition)
    uint32 vertexColorsOffset;   // Offset to ACVT chunk (WoD addition)
    uint32 shadowMapOffset;      // Offset to ABSH chunk (WoD addition)
    uint32 acnkSize;             // Size of each ACNK chunk (WoD addition)
    uint8 skyboxId;              // Skybox ID to use for this tile (WoD addition)
    uint8 reserved[3];           // Padding
    float waterLevel;            // Water level for this tile
    float[8] lowQualityTextureMapping; // Low quality texture mapping (expanded in WoD)
    uint32 layerCount;           // Number of texture layers in this tile
    uint32 lodLevel;             // Level of detail settings
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32 | Global flags for this terrain tile |
| mcnkOffset | uint32 | Offset to the ACNK chunks |
| mtexOffset | uint32 | Offset to the ATEX chunk (textures) |
| mmdxOffset | uint32 | Offset to the ADOO chunk (doodads) |
| mmidOffset | uint32 | Offset to the model indices |
| mwmoOffset | uint32 | Offset to the WMO filenames |
| mwidOffset | uint32 | Offset to the WMO indices |
| mddfOffset | uint32 | Offset to the doodad placement info |
| modfOffset | uint32 | Offset to the WMO placement info |
| mtdaOffset | uint32 | Offset to liquid/fog terrain data (WoD addition) |
| fogDataOffset | uint32 | Offset to fog data (WoD addition) |
| vertexColorsOffset | uint32 | Offset to ACVT chunk (WoD addition) |
| shadowMapOffset | uint32 | Offset to ABSH chunk (WoD addition) |
| acnkSize | uint32 | Size of each ACNK chunk (WoD addition) |
| skyboxId | uint8 | Skybox ID to use for this tile (WoD addition) |
| reserved | uint8[3] | Padding bytes |
| waterLevel | float | Water level for this tile |
| lowQualityTextureMapping | float[8] | Low quality texture mapping (expanded in WoD) |
| layerCount | uint32 | Number of texture layers in this tile |
| lodLevel | uint32 | Level of detail settings |

## Dependencies
- References all other chunks via offsets

## Implementation Notes
- Size: Variable (larger than v22 due to additional fields)
- New fields added in WoD (v23):
  - mtdaOffset: Points to liquid/fog terrain data
  - fogDataOffset: Points to fog data
  - vertexColorsOffset: Points to ACVT chunk
  - shadowMapOffset: Points to ABSH chunk
  - acnkSize: Size of each ACNK chunk
  - skyboxId: ID of skybox to use for this tile
  - Expanded lowQualityTextureMapping from 4 to 8 values
- Similar to AHDR in ADT v22, but with more fields for enhanced visual features
- Used to locate all other chunks in the file via offsets

## Implementation Example
```csharp
[Flags]
public enum AHDRFlags
{
    None = 0,
    HasATEX = 0x1,           // Has ATEX (texture) information
    HasADOO = 0x2,           // Has ADOO (doodad) information
    HasLiquidData = 0x4,     // Has liquid data
    HasVertexColors = 0x8,   // Has vertex coloring (ACVT chunk)
    HasShadowMap = 0x10,     // Has shadow map (ABSH chunk)
    HasFogData = 0x20,       // Has fog data
    IsHighResolution = 0x40  // Uses high-resolution terrain
}

public class AHDR
{
    // Basic properties from v22
    public AHDRFlags Flags { get; set; }
    public uint ACNKOffset { get; set; }
    public uint ATEXOffset { get; set; }
    public uint ADOOOffset { get; set; }
    public uint ModelIndicesOffset { get; set; }
    public uint WMOFilenamesOffset { get; set; }
    public uint WMOIndicesOffset { get; set; }
    public uint DoodadPlacementOffset { get; set; }
    public uint WMOPlacementOffset { get; set; }
    
    // New properties in v23 (WoD)
    public uint LiquidFogDataOffset { get; set; }
    public uint FogDataOffset { get; set; }
    public uint VertexColorsOffset { get; set; }
    public uint ShadowMapOffset { get; set; }
    public uint ACNKSize { get; set; }
    public byte SkyboxId { get; set; }
    public byte[] Reserved { get; set; } = new byte[3];
    public float WaterLevel { get; set; }
    public float[] LowQualityTextureMapping { get; set; } = new float[8]; // Expanded in WoD
    public uint LayerCount { get; set; }
    public uint LodLevel { get; set; }
    
    // Helper properties
    public bool HasTextureInformation => (Flags & AHDRFlags.HasATEX) != 0;
    public bool HasDoodadInformation => (Flags & AHDRFlags.HasADOO) != 0;
    public bool HasLiquidData => (Flags & AHDRFlags.HasLiquidData) != 0;
    public bool HasVertexColors => (Flags & AHDRFlags.HasVertexColors) != 0;
    public bool HasShadowMap => (Flags & AHDRFlags.HasShadowMap) != 0;
    public bool HasFogData => (Flags & AHDRFlags.HasFogData) != 0;
    public bool IsHighResolution => (Flags & AHDRFlags.IsHighResolution) != 0;
    
    // Try to get the ACVT chunk if it exists
    public ACVT GetVertexColors(BinaryReader reader)
    {
        if (!HasVertexColors || VertexColorsOffset == 0)
            return null;
            
        long currentPosition = reader.BaseStream.Position;
        reader.BaseStream.Position = VertexColorsOffset;
        
        // Read ACVT chunk
        var acvt = new ACVT(reader);
        
        reader.BaseStream.Position = currentPosition;
        return acvt;
    }
    
    // Try to get the ABSH chunk if it exists
    public ABSH GetShadowMap(BinaryReader reader)
    {
        if (!HasShadowMap || ShadowMapOffset == 0)
            return null;
            
        long currentPosition = reader.BaseStream.Position;
        reader.BaseStream.Position = ShadowMapOffset;
        
        // Read ABSH chunk
        var absh = new ABSH(reader);
        
        reader.BaseStream.Position = currentPosition;
        return absh;
    }
    
    // Get skybox information
    public string GetSkyboxName()
    {
        // Map skybox ID to actual skybox names
        switch (SkyboxId)
        {
            case 0: return "Default";
            case 1: return "Nagrand";
            case 2: return "Hellfire";
            case 3: return "Terokkar";
            case 4: return "Shattrath";
            case 5: return "Shadowmoon";
            case 6: return "Zangarmarsh";
            case 7: return "Blade's Edge";
            // Add additional skyboxes for WoD and later
            default: return $"Unknown ({SkyboxId})";
        }
    }
}
```

## Usage Context
The AHDR chunk serves as the primary header for an ADT v23 terrain tile, providing global information about the tile and allowing the parser to locate all other chunks via offsets. Compared to the v22 version, the v23 AHDR adds support for additional visual features introduced in Warlords of Draenor, including vertex coloring (ACVT), shadow mapping (ABSH), improved fog effects, and skybox selection.

The new fields specifically support atmospheric and lighting enhancements that were a major focus in WoD's graphical upgrade. The vertex colors chunk (ACVT) and blend shadow chunk (ABSH) work together to create more realistic terrain appearance, especially at distance. The skybox ID allows specific sky visuals to be assigned to individual terrain tiles. 