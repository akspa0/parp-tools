# W002: MPHD

## Type
WDT Chunk

## Source
WDT.md

## Description
The MPHD (Map Header) chunk contains general information about the map, including various flags that affect how the map is rendered and how the player interacts with it.

## Structure
```csharp
struct MPHD
{
    /*0x00*/ uint32_t flags;              // Map flags
    /*0x04*/ uint32_t lq_texture;         // Low-quality texture layer (legacy, unused in modern clients)
    /*0x08*/ uint32_t layerCount;         // Number of map layers (used in multi-level maps)
    /*0x0C*/ uint32_t pad;                // Padding or unknown data
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32_t | Bitfield of map flags determining various map properties |
| lq_texture | uint32_t | Low-quality texture layer index (legacy, unused in modern clients) |
| layerCount | uint32_t | Number of map layers (used in dungeons with multiple levels) |
| pad | uint32_t | Padding or unknown data |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | HAS_WMO_ONLY | Map contains only WMO objects, no terrain |
| 0x2 | HAS_NO_TERRAIN | No terrain data (often used with HAS_WMO_ONLY) |
| 0x4 | HAS_FLIGHT_BOUNDS | Map has flight boundaries defined |
| 0x8 | HAS_VERTEX_COLOR | Terrain uses vertex coloring |
| 0x10 | HAS_MCCV | Map uses MCCV chunks for vertex coloring |
| 0x20 | HAS_MULTI_LAYERS | Map has multiple layers (dungeons with multiple floors) |
| 0x40 | USE_GLOBAL_MAP_OBJ | Use global map objects |
| 0x80 | HAS_LIQUIDS | Map has liquid data (water, lava, etc.) |
| 0x100 | IS_INTERIOR | Map is an interior area (no sky showing) |
| 0x200 | DISABLE_LOD | Disables LOD for this map (forces full detail) |
| 0x400 | DISABLE_WATER | Disables water rendering |
| 0x800 | USE_NEW_WATER | Uses new water rendering system |
| 0x1000 | UNUSED_FLAG_1 | Unused flag |
| 0x2000 | DONT_LOAD_WDLS | Don't load WDL files (no distant terrain) |
| 0x4000 | HAS_FLAT_MODE | Map has a flat mode (like Orgrimmar in Cataclysm) |
| 0x8000 | IS_DEVELOPMENT | Map is in development stage |

## Dependencies
- MVER (W001) - Version information determines exact format of MPHD

## Implementation Notes
- The MPHD chunk is required in all WDT files
- Flags determine how the map is rendered and what features are available
- Multiple layers are used for dungeons with multiple floors
- Some flags are specific to certain client versions
- In newer versions, additional flags may be present

## Implementation Example
```csharp
public class MPHD : IChunk
{
    [Flags]
    public enum MapFlags : uint
    {
        None = 0,
        HasWmoOnly = 0x1,
        HasNoTerrain = 0x2,
        HasFlightBounds = 0x4,
        HasVertexColor = 0x8,
        HasMCCV = 0x10,
        HasMultiLayers = 0x20,
        UseGlobalMapObj = 0x40,
        HasLiquids = 0x80,
        IsInterior = 0x100,
        DisableLOD = 0x200,
        DisableWater = 0x400,
        UseNewWater = 0x800,
        UnusedFlag1 = 0x1000,
        DontLoadWDLs = 0x2000,
        HasFlatMode = 0x4000,
        IsDevelopment = 0x8000
    }
    
    public MapFlags Flags { get; set; }
    public uint LqTexture { get; set; }
    public uint LayerCount { get; set; }
    public uint Padding { get; set; }
    
    public void Parse(BinaryReader reader)
    {
        Flags = (MapFlags)reader.ReadUInt32();
        LqTexture = reader.ReadUInt32();
        LayerCount = reader.ReadUInt32();
        Padding = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write((uint)Flags);
        writer.Write(LqTexture);
        writer.Write(LayerCount);
        writer.Write(Padding);
    }
    
    // Helper methods for checking flags
    public bool IsWmoOnly => (Flags & MapFlags.HasWmoOnly) != 0;
    public bool HasNoTerrain => (Flags & MapFlags.HasNoTerrain) != 0;
    public bool HasFlightBounds => (Flags & MapFlags.HasFlightBounds) != 0;
    public bool UseVertexColors => (Flags & MapFlags.HasVertexColor) != 0;
    public bool HasMultipleLayers => (Flags & MapFlags.HasMultiLayers) != 0;
    public bool HasLiquids => (Flags & MapFlags.HasLiquids) != 0;
    public bool IsInterior => (Flags & MapFlags.IsInterior) != 0;
    public bool ShouldLoadWDLs => (Flags & MapFlags.DontLoadWDLs) == 0;
}
```

## Version Information
- Present in all versions of WDT files
- Some flags may have different meanings in different versions
- Version 22+ files may have expanded flag definitions
- Structure size remains constant across versions

## Map Types
Based on the flags, WDT files can define different types of maps:
- **Standard Map**: Regular outdoor terrain with ADT files (no special flags)
- **WMO-Only Map**: Interior areas built only with WMO objects (HasWmoOnly, HasNoTerrain)
- **Hybrid Map**: Areas with both terrain and major WMO structures
- **Multi-Layer Map**: Dungeons with multiple floors (HasMultiLayers)

## Usage Context
The MPHD chunk is essential for determining how a map should be handled by the client. It affects:
- Whether terrain data should be loaded
- How liquids are rendered
- Whether the sky is visible
- If WDL files should be loaded for distant terrain
- Whether the map has multiple layers (like in multi-level dungeons)
- Special rendering modes for the map

The flags in this chunk guide the client in determining which features to enable or disable when rendering the map. 