# MOLR - WMO Group Light References

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOLR chunk contains references to light sources that affect the WMO group. Each entry is a 16-bit index into the MOLT chunk in the root file, identifying which lights from the global light list should illuminate this particular group. This system allows multiple groups to share the same light definitions while allowing each group to be affected by a specific subset of lights that are relevant to its position and function within the overall WMO.

## Structure

```csharp
public struct MOLR
{
    public ushort[] lightRefList; // Array of indices into the MOLT chunk in the root file
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | lightRefList | ushort[] | Array of 16-bit indices referencing light definitions in the root file's MOLT chunk. Each index occupies 2 bytes, and the number of references can be calculated as chunk size / 2. |

## Dependencies
- **MOLT**: In the root file, contains the actual light definitions referenced by these indices.
- **MOGP**: The flag 0x00000200 (HasLights) in the MOGP header indicates that this chunk is present.

## Implementation Notes
- The size of the chunk should be a multiple of 2 bytes (each reference is a ushort).
- Some WMO groups may reference more lights than a typical graphics card can render simultaneously. The client likely implements some form of light culling or prioritization.
- Not all lights in the root file will necessarily be referenced by any given group.
- In some World of Warcraft client versions, particularly 10.1.5 and later, this chunk is internally referred to as "m_doodadOverrideLightRefList", suggesting it may have additional functionality related to overriding lighting for doodads.
- The name MOLR likely stands for "Map Object Light References".
- Light references are typically processed during rendering to determine which lights affect visible surfaces, with potential culling based on distance or visibility.
- When a WMO group doesn't have the HasLights flag set, this chunk will not be present, and the group will likely use default lighting or ambient lighting only.

## Implementation Example

```csharp
public class MOLRChunk : IWmoGroupChunk
{
    public string ChunkId => "MOLR";
    public List<ushort> LightReferences { get; private set; } = new List<ushort>();

    public void Parse(BinaryReader reader, long size)
    {
        // Calculate the number of light references
        int referenceCount = (int)(size / 2);
        
        // Read all light references
        for (int i = 0; i < referenceCount; i++)
        {
            ushort lightIndex = reader.ReadUInt16();
            LightReferences.Add(lightIndex);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 2 != 0)
        {
            throw new InvalidDataException("MOLR chunk size is not a multiple of 2 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var lightIndex in LightReferences)
        {
            writer.Write(lightIndex);
        }
    }
}
```

## Usage Context
- Light references define which light sources from the root file affect this particular group.
- During rendering, these references are used to determine which lights need to be applied to the surfaces in the group.
- The lighting system creates visual depth and atmosphere within the WMO model.
- For performance reasons, the client might need to prioritize or cull some lights if a group references more lights than the rendering system can handle simultaneously.
- The lighting information is crucial for creating realistic interior spaces and properly illuminated exteriors in the game world. 