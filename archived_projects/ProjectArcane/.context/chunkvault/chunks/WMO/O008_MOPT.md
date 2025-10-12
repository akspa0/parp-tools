# O008: MOPT

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOPT (Map Object PorTal) chunk defines the portals that connect different groups within a WMO. Each portal is a convex polygon that serves as a boundary between two groups, allowing the engine to determine visibility between connected spaces. Portals are essential for efficient rendering of complex interior spaces, as they enable the engine to render only what might be visible from the player's position.

## Structure
```csharp
struct MOPT
{
    SMOPortal[] portals;  // Array of portal definitions
};

struct SMOPortal
{
    /*0x00*/ uint16_t portalId;      // Portal identifier
    /*0x02*/ uint16_t groupId;       // Group this portal leads to
    /*0x04*/ int16_t vertRefs[4];    // Indices into MOPV vertex list (-1 if unused)
    /*0x0C*/ C3Vector normal;        // Normal vector pointing into the target group
    /*0x18*/ float unknown;          // Unknown, might be a plane distance
};
```

## Properties

### SMOPortal Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | portalId | uint16_t | Unique identifier for this portal |
| 0x02 | groupId | uint16_t | ID of the group this portal connects to |
| 0x04 | vertRefs | int16_t[4] | Indices to vertices in MOPV chunk (-1 if unused) |
| 0x0C | normal | C3Vector | Normal vector pointing into the connected group |
| 0x18 | unknown | float | Unknown value, possibly plane distance |

## Dependencies
- MOHD: The nPortals field indicates how many portal definitions should be present
- MOPV: Contains the vertex coordinates referenced by vertRefs indices
- MOGI: References portals by index range (portalStart/portalCount)

## Implementation Notes
- Each portal definition is 28 bytes (0x1C)
- Portals are always defined as convex polygons with 3 or 4 vertices
- The vertRefs array contains indices into the MOPV vertex array
- A value of -1 in vertRefs indicates an unused vertex slot (for triangular portals)
- The normal vector points into the group that the portal connects to
- Portals are bidirectional - each connection between groups requires two portals
- Portal vertices should be arranged in counter-clockwise order when viewed from the side the normal points to
- The portalId field is a unique identifier within the WMO
- The groupId field identifies which group the portal connects to, not the group it's in
- The MOGI chunk references portals through portalStart and portalCount fields

## Implementation Example
```csharp
public class MOPT : IChunk
{
    public List<Portal> Portals { get; private set; }
    
    public MOPT()
    {
        Portals = new List<Portal>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many portals we expect
        int portalCount = (int)(size / 0x1C); // Each portal is 28 bytes
        
        Portals.Clear();
        
        for (int i = 0; i < portalCount; i++)
        {
            Portal portal = new Portal();
            
            portal.Id = reader.ReadUInt16();
            portal.GroupId = reader.ReadUInt16();
            
            // Read vertex references
            portal.VertexReferences = new short[4];
            for (int j = 0; j < 4; j++)
            {
                portal.VertexReferences[j] = reader.ReadInt16();
            }
            
            // Read normal vector
            portal.Normal = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            portal.Unknown = reader.ReadSingle();
            
            Portals.Add(portal);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (Portal portal in Portals)
        {
            writer.Write(portal.Id);
            writer.Write(portal.GroupId);
            
            // Write vertex references
            for (int i = 0; i < 4; i++)
            {
                writer.Write(portal.VertexReferences[i]);
            }
            
            // Write normal vector
            writer.Write(portal.Normal.X);
            writer.Write(portal.Normal.Y);
            writer.Write(portal.Normal.Z);
            
            writer.Write(portal.Unknown);
        }
    }
    
    public Portal GetPortal(int index)
    {
        if (index >= 0 && index < Portals.Count)
        {
            return Portals[index];
        }
        
        throw new IndexOutOfRangeException($"Portal index {index} is out of range. Valid range: 0-{Portals.Count - 1}");
    }
    
    public List<Portal> GetPortalsForGroup(int groupId)
    {
        return Portals.Where(p => p.GroupId == groupId).ToList();
    }
}

public class Portal
{
    public ushort Id { get; set; }
    public ushort GroupId { get; set; }
    public short[] VertexReferences { get; set; }
    public Vector3 Normal { get; set; }
    public float Unknown { get; set; }
    
    // Helper properties
    public int VertexCount => VertexReferences.Count(v => v >= 0);
    public bool IsTriangle => VertexCount == 3;
    public bool IsQuad => VertexCount == 4;
    
    public Portal()
    {
        // Initialize with defaults
        Id = 0;
        GroupId = 0;
        VertexReferences = new short[4] { -1, -1, -1, -1 };
        Normal = new Vector3(0, 0, 1);
        Unknown = 0;
    }
    
    // Helper methods for portal operations
    public void SetAsTriangle(short v1, short v2, short v3)
    {
        VertexReferences[0] = v1;
        VertexReferences[1] = v2;
        VertexReferences[2] = v3;
        VertexReferences[3] = -1;
    }
    
    public void SetAsQuad(short v1, short v2, short v3, short v4)
    {
        VertexReferences[0] = v1;
        VertexReferences[1] = v2;
        VertexReferences[2] = v3;
        VertexReferences[3] = v4;
    }
}
```

## Validation Requirements
- The number of portal definitions should match the nPortals field in the MOHD chunk
- Each portal should have at least 3 valid vertex references
- Vertex references should be valid indices into the MOPV vertex array
- The normal vector should be normalized (length = 1.0)
- Group IDs should be valid indices into the group array
- Portal vertices should define a convex polygon

## Usage Context
The MOPT chunk is crucial for the WMO's portal-based visibility system:

1. **Visibility Determination**: Portals define where visibility can flow between groups
2. **Culling**: The engine uses portals to determine which groups need to be rendered
3. **Group Connectivity**: Portals establish the topological connections between groups
4. **Spatial Optimization**: Portal culling significantly improves rendering performance

The portal system works as follows:
1. The engine identifies which group contains the camera
2. It checks which portals in that group are potentially visible
3. For each visible portal, it adds the connected group to the render queue
4. This process continues recursively until all potentially visible groups are identified
5. Only groups determined to be potentially visible are rendered

This approach is particularly effective for complex indoor environments where traditional view frustum culling alone would be inefficient. By rendering only what can potentially be seen through open portals, the engine can maintain high performance even in detailed interior spaces.

Portals are placed at natural boundaries between areas, such as:
- Doorways between rooms
- Windows that allow visibility between spaces
- Archways or open passages
- Any location where visibility can flow between separate groups

Each portal is paired with a corresponding portal in the connected group, with normals pointing in opposite directions. This bidirectional relationship allows visibility testing regardless of which side the camera is on. 