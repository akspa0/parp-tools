# MOPR - WMO Portal References

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOPR chunk defines the relationship between portals and groups in the WMO. It creates a mapping that specifies which portals connect which groups, enabling the engine to determine visibility between different sections of the world model. This system is crucial for efficient rendering and occlusion culling.

## Structure

```csharp
public struct MOPR
{
    public SMOPortalReference[] portalReferences; // Array of portal reference structures
}

public struct SMOPortalReference
{
    public ushort portalIndex;  // Index into the MOPT array
    public ushort groupIndex;   // Index into the MOGI array
    public short side;          // Side of the portal this reference is for (-1 or 1)
    public ushort unknown;      // Padding or reserved
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | portalIndex | ushort | Index into the MOPT array, identifying which portal this reference relates to |
| 0x02 | groupIndex | ushort | Index into the MOGI array, identifying which group this portal reference belongs to |
| 0x04 | side | short | Indicates which side of the portal this reference represents. Values are typically -1 or 1 |
| 0x06 | unknown | ushort | Unused or padding value, typically set to 0 |

## Dependencies
- **MOHD**: Contains the number of portal references, which determines the array size
- **MOPT**: Contains the portal definitions that are referenced by portalIndex
- **MOGI**: Contains the group information that is referenced by groupIndex

## Implementation Notes
- Each portal reference entry is 8 bytes in size.
- Portal references create the connectivity graph between WMO groups through portals.
- Each physical portal (from MOPT) typically has two portal references, one for each side of the portal.
- The side field (-1 or 1) indicates which face of the portal this reference is for. When a viewer is on the "positive" side of a portal (side = 1), they can see through to the connected group. Similarly for the "negative" side (side = -1).
- Portal references are used by the game engine to perform portal-based visibility determination during rendering.
- A group can have multiple portal references, connecting it to several other groups.

## Implementation Example

```csharp
public class MOPRChunk : IWmoChunk
{
    public string ChunkId => "MOPR";
    public List<PortalReference> PortalReferences { get; set; } = new List<PortalReference>();

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate how many portal references are in the chunk
        int count = (int)(size / 8); // Each portal reference is 8 bytes
        
        for (int i = 0; i < count; i++)
        {
            PortalReference reference = new PortalReference
            {
                PortalIndex = reader.ReadUInt16(),
                GroupIndex = reader.ReadUInt16(),
                Side = reader.ReadInt16(),
                Unknown = reader.ReadUInt16()
            };
            PortalReferences.Add(reference);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate size (8 bytes per portal reference)
        uint dataSize = (uint)(PortalReferences.Count * 8);
        writer.Write(dataSize);
        
        // Write all portal references
        foreach (var reference in PortalReferences)
        {
            writer.Write(reference.PortalIndex);
            writer.Write(reference.GroupIndex);
            writer.Write(reference.Side);
            writer.Write(reference.Unknown);
        }
    }
    
    public class PortalReference
    {
        public ushort PortalIndex { get; set; }
        public ushort GroupIndex { get; set; }
        public short Side { get; set; }
        public ushort Unknown { get; set; }
    }
}
```

## Validation Requirements
- The number of portal references must match the expected count based on the MOHD chunk.
- Each portalIndex must be a valid index into the MOPT array.
- Each groupIndex must be a valid index into the MOGI array.
- The side value should typically be either -1 or 1.
- Portal references should form a consistent network of connections between groups.
- Each portal should typically have two references (one for each side).

## Usage Context
- **Visibility Determination:** Portal references are essential for determining which parts of the WMO are visible from a given viewpoint.
- **Rendering Optimization:** By only rendering groups that are potentially visible through portals, the game can significantly optimize performance.
- **Spatial Organization:** Portal references define how the different sections of a WMO are connected, creating the logical structure of the model.
- **Occlusion Culling:** The portal system allows for effective occlusion culling, where groups that cannot be seen through any portal are excluded from rendering.
- **Level Design:** For game designers, portals and their references provide a way to control visibility and flow within complex structures, helping to manage rendering load and create a sense of space. 