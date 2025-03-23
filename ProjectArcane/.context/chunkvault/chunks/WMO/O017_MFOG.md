# MFOG - WMO Fog

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MFOG chunk defines fog volumes within the WMO. These fog volumes create atmospheric effects that add depth and ambiance to the scene. Each fog entry defines properties such as color, density, and the 3D volume where the fog is applied. Fog effects can be used to simulate weather conditions, mystical environments, or to help mask the transition between visible and non-visible areas.

## Structure

```csharp
public struct MFOG
{
    public SMOFog[] fogs; // Array of fog definitions
}

public struct SMOFog
{
    public uint fogId;         // Fog ID (used for lookup/reference)
    public Vector3 position;   // Position of the fog volume center
    public float smallerRadius; // Smaller radius (inner radius for transition)
    public float largerRadius;  // Larger radius (outer reach of the fog)
    public float fogEnd;       // End of the fog (distance-based edge)
    public float fogStartScalar; // Multiplier for fog start distance (fogEnd * fogStartScalar = fogStart)
    public uint flags;         // Bit flags controlling fog behavior
    public float color;        // Fog color (0BGR format)
    public float unknownFloat1; // Unknown floating point value
    public float unknownFloat2; // Unknown floating point value
    public Vector3 position2;  // Secondary position (appears to be legacy or reserved)
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | fogId | uint | Unique identifier for this fog volume |
| 0x04 | position | Vector3 | Center position of the fog volume (x, y, z) |
| 0x10 | smallerRadius | float | Inner radius of the fog effect (creates transition zone) |
| 0x14 | largerRadius | float | Outer radius of the fog effect (maximum extent) |
| 0x18 | fogEnd | float | End distance for the fog effect |
| 0x1C | fogStartScalar | float | Multiplier for fog start distance (fogEnd * fogStartScalar = fogStart) |
| 0x20 | flags | uint | Bit flags controlling fog behavior and rendering |
| 0x24 | color | float | Fog color in 0BGR format packed into a float |
| 0x28 | unknownFloat1 | float | Unknown parameter (possibly related to fog density) |
| 0x2C | unknownFloat2 | float | Unknown parameter (possibly related to additional effects) |
| 0x30 | position2 | Vector3 | Secondary position (appears to be unused in modern WMOs) |

## Fog Flags

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x01 | FOG_FLAG_ENABLED | When set, the fog is enabled and will be rendered |
| 0x10 | FOG_FLAG_CUSTOM_INTERIOR | Used for interior fog settings in certain environments |
| 0x20 | FOG_FLAG_CUSTOM_EXTERIOR | Used for exterior fog settings in certain environments |

## Dependencies
- **MOHD**: Contains the number of fog definitions, which determines the array size.

## Implementation Notes
- Each fog definition is 48 bytes in size.
- The color value is stored as a 32-bit value (packed into a float) with BGR color channels and possibly an alpha component.
- The fogStartScalar is multiplied by fogEnd to determine where the fog starts to appear. This creates a gradient effect where fog gradually increases in density.
- The two radius values (smallerRadius and largerRadius) define a spherical volume with a transition zone.
- Position and position2 appear to define the center of the fog volume, though position2 might be legacy or reserved for special cases.
- The fogId can be used for lookup or reference purposes, especially when applying fog effects to specific groups.
- Fog can be placed in specific areas to create localized atmospheric effects.

## Implementation Example

```csharp
public class MFOGChunk : IWmoChunk
{
    public string ChunkId => "MFOG";
    public List<FogDefinition> Fogs { get; set; } = new List<FogDefinition>();

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate how many fog definitions are in the chunk
        int count = (int)(size / 48); // Each fog definition is 48 bytes
        
        for (int i = 0; i < count; i++)
        {
            FogDefinition fog = new FogDefinition
            {
                FogId = reader.ReadUInt32(),
                Position = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                SmallerRadius = reader.ReadSingle(),
                LargerRadius = reader.ReadSingle(),
                FogEnd = reader.ReadSingle(),
                FogStartScalar = reader.ReadSingle(),
                Flags = reader.ReadUInt32(),
                Color = reader.ReadSingle(),
                UnknownFloat1 = reader.ReadSingle(),
                UnknownFloat2 = reader.ReadSingle(),
                Position2 = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                }
            };
            Fogs.Add(fog);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate size (48 bytes per fog definition)
        uint dataSize = (uint)(Fogs.Count * 48);
        writer.Write(dataSize);
        
        // Write all fog definitions
        foreach (var fog in Fogs)
        {
            writer.Write(fog.FogId);
            writer.Write(fog.Position.X);
            writer.Write(fog.Position.Y);
            writer.Write(fog.Position.Z);
            writer.Write(fog.SmallerRadius);
            writer.Write(fog.LargerRadius);
            writer.Write(fog.FogEnd);
            writer.Write(fog.FogStartScalar);
            writer.Write(fog.Flags);
            writer.Write(fog.Color);
            writer.Write(fog.UnknownFloat1);
            writer.Write(fog.UnknownFloat2);
            writer.Write(fog.Position2.X);
            writer.Write(fog.Position2.Y);
            writer.Write(fog.Position2.Z);
        }
    }
    
    public class FogDefinition
    {
        public uint FogId { get; set; }
        public Vector3 Position { get; set; }
        public float SmallerRadius { get; set; }
        public float LargerRadius { get; set; }
        public float FogEnd { get; set; }
        public float FogStartScalar { get; set; }
        public uint Flags { get; set; }
        public float Color { get; set; }
        public float UnknownFloat1 { get; set; }
        public float UnknownFloat2 { get; set; }
        public Vector3 Position2 { get; set; }
        
        public FogDefinition()
        {
            Position = new Vector3();
            Position2 = new Vector3();
        }
    }
    
    public class Vector3
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
}
```

## Validation Requirements
- The number of fog definitions must match the expected count (chunk size should be a multiple of 48 bytes).
- The fogStartScalar should typically be between 0 and 1 (as it represents a percentage of fogEnd).
- The color value should represent a valid color.
- The smallerRadius should be less than or equal to largerRadius.
- The flags should contain valid bit combinations.
- Position coordinates should be within a reasonable range for the WMO model's dimensions.

## Usage Context
- **Atmospheric Effects:** Fog adds atmosphere and depth to scenes, creating a more immersive environment.
- **Distance Cues:** Fog provides visual cues about distance, enhancing the player's perception of space.
- **Visual Transition:** Fog can mask the transition between visible and non-visible areas or different levels of detail.
- **Themed Environments:** Different fog colors and densities can be used to create specific moods or environments.
- **Weather Simulation:** Fog can be part of a weather system, simulating misty conditions or smoke effects.
- **Performance Optimization:** Fog can be used to reduce visual detail in the distance, potentially improving performance. 