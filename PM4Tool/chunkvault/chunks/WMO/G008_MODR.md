# MODR - WMO Group Doodad References

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MODR chunk contains references to doodads (decorative M2 models) that are placed within the WMO group. Each entry is a 16-bit index into the MODD chunk in the root file, identifying which doodad instances from the global doodad list should appear in this particular group. This system allows the WMO to efficiently reference decorative objects without duplicating their placement information in each group file.

## Structure

```csharp
public struct MODR
{
    public ushort[] doodadRefList; // Array of indices into the MODD chunk in the root file
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | doodadRefList | ushort[] | Array of 16-bit indices referencing doodad instances in the root file's MODD chunk. Each index occupies 2 bytes, and the number of references can be calculated as chunk size / 2. |

## Dependencies
- **MODD**: In the root file, contains the actual doodad definitions referenced by these indices.
- **MODS**: In the root file, defines doodad sets that can be shown or hidden as a group.
- **MODN**: In the root file, contains the filenames of the M2 models used as doodads.
- **MOGP**: The flag 0x00000800 (HasDoodads) in the MOGP header indicates that this chunk is present.

## Implementation Notes
- The size of the chunk should be a multiple of 2 bytes (each reference is a ushort).
- When processing doodad references, they need to be filtered based on the currently active doodad set. Only doodads that belong to the active set (as specified in their MODD entry) should be rendered.
- Not all doodads in the root file will necessarily be referenced by any given group.
- Doodad references need to be processed differently from the main WMO geometry, as they represent separate M2 models that need to be loaded and positioned within the WMO.
- The doodad references can represent a wide variety of decorative objects, from furniture and light fixtures to vegetation and signage.
- When a WMO group doesn't have the HasDoodads flag set, this chunk will not be present, and the group will not have any decorative doodads.

## Implementation Example

```csharp
public class MODRChunk : IWmoGroupChunk
{
    public string ChunkId => "MODR";
    public List<ushort> DoodadReferences { get; private set; } = new List<ushort>();

    public void Parse(BinaryReader reader, long size)
    {
        // Calculate the number of doodad references
        int referenceCount = (int)(size / 2);
        
        // Read all doodad references
        for (int i = 0; i < referenceCount; i++)
        {
            ushort doodadIndex = reader.ReadUInt16();
            DoodadReferences.Add(doodadIndex);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 2 != 0)
        {
            throw new InvalidDataException("MODR chunk size is not a multiple of 2 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var doodadIndex in DoodadReferences)
        {
            writer.Write(doodadIndex);
        }
    }
    
    // Helper method to get only doodads that belong to a specific doodad set
    public List<ushort> GetDoodadsForSet(uint setId, List<MODDChunk.DoodadDefinition> doodadDefinitions)
    {
        return DoodadReferences
            .Where(index => index < doodadDefinitions.Count && doodadDefinitions[index].Set == setId)
            .ToList();
    }
}
```

## Usage Context
- Doodad references identify which decorative objects from the root file should appear in this particular group.
- During rendering, these references are used to determine which M2 models need to be loaded and positioned within the WMO group.
- The doodad system adds visual variety and detail to WMO environments without requiring these details to be built into the main model geometry.
- Doodads can be grouped into sets that can be shown or hidden together, allowing different variations of the same WMO.
- Doodads are critical for creating realistic and detailed environments, as they add small-scale details and variation that would be inefficient to model directly in the WMO geometry. 