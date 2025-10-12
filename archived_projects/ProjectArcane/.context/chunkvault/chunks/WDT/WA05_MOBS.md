# WA05: MOBS

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MOBS (Map Object BSP) chunk contains Binary Space Partitioning (BSP) tree data for map objects in the Alpha WDT format. BSP trees are used for visibility determination and efficient rendering of complex 3D scenes. This chunk stores the spatial subdivision information that helps the game engine decide which objects or portions of objects to render based on the player's viewpoint.

## Structure
```csharp
struct MOBS
{
    struct BSPNode
    {
        /*0x00*/ uint32_t plane_type;      // Type of splitting plane (0: YZ, 1: XZ, 2: XY)
        /*0x04*/ float plane_distance;     // Distance from origin to splitting plane
        /*0x08*/ uint32_t children[2];     // Indices to child nodes or geometry
        /*0x10*/ Vector3 bbox_min;         // Minimum corner of bounding box
        /*0x1C*/ Vector3 bbox_max;         // Maximum corner of bounding box
        /*0x28*/ uint32_t flags;           // Node flags
    };

    BSPNode[] nodes;                       // Array of BSP nodes
}

struct Vector3
{
    float x, y, z;
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nodes | BSPNode[] | Array of BSP nodes defining the spatial partitioning |

### BSPNode Properties
| Name | Type | Description |
|------|------|-------------|
| plane_type | uint32_t | Type of splitting plane (0: YZ, 1: XZ, 2: XY) |
| plane_distance | float | Distance from origin to splitting plane |
| children | uint32_t[2] | Indices to child nodes or references to geometry |
| bbox_min | Vector3 | Minimum corner of node's bounding box |
| bbox_max | Vector3 | Maximum corner of node's bounding box |
| flags | uint32_t | Node flags for special behavior |

## BSP Node Type Values
| Value | Name | Description |
|-------|------|-------------|
| 0 | YZ_PLANE | Splitting plane is perpendicular to X axis (YZ plane) |
| 1 | XZ_PLANE | Splitting plane is perpendicular to Y axis (XZ plane) |
| 2 | XY_PLANE | Splitting plane is perpendicular to Z axis (XY plane) |

## BSP Child Node Interpretation
The values in the `children` array have special interpretation:
- If the high bit (0x80000000) is set, then the value refers to geometry (triangle indices)
- If the high bit is clear, the value is an index to another BSP node
- The first child (index 0) is for the negative side of the plane
- The second child (index 1) is for the positive side of the plane

## Dependencies
- MAOI (WA02) - Contains the actual geometry data referenced by the BSP tree
- MAOT (WA01) - Indirectly related, as it indexes the MAOI entries that contain BSP references
- MAOH (WA03) - May contain global settings affecting BSP traversal

## Implementation Notes
- The BSP tree is used to efficiently determine which parts of the map are visible from a given viewpoint
- During rendering, the tree is traversed based on the camera position relative to splitting planes
- Child nodes are processed front-to-back or back-to-front depending on the rendering algorithm (painter's algorithm vs. z-buffer)
- The bounding boxes (bbox_min and bbox_max) help with frustum culling and optimization
- Triangle references in the BSP tree correspond to triangle indices in the associated MAOI object
- This chunk is present only when the map contains 3D objects with complex visibility requirements

## Implementation Example
```csharp
public class MOBS : IChunk
{
    public class BSPNode
    {
        public uint PlaneType { get; set; }
        public float PlaneDistance { get; set; }
        public uint[] Children { get; set; } = new uint[2];
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public uint Flags { get; set; }
        
        // Helper properties
        public bool IsLeafNodeLeft => (Children[0] & 0x80000000) != 0;
        public bool IsLeafNodeRight => (Children[1] & 0x80000000) != 0;
        public uint LeftChildIndex => Children[0] & 0x7FFFFFFF;
        public uint RightChildIndex => Children[1] & 0x7FFFFFFF;
        
        // Helper method to determine which side of the plane a point is on
        public int ClassifyPoint(Vector3 point)
        {
            float coordinate;
            
            switch (PlaneType)
            {
                case 0: // YZ plane (X normal)
                    coordinate = point.X;
                    break;
                case 1: // XZ plane (Y normal)
                    coordinate = point.Y;
                    break;
                case 2: // XY plane (Z normal)
                    coordinate = point.Z;
                    break;
                default:
                    return 0; // On plane by default
            }
            
            if (coordinate < PlaneDistance)
                return -1; // Negative side
            else if (coordinate > PlaneDistance)
                return 1;  // Positive side
            else
                return 0;  // On plane
        }
    }
    
    public List<BSPNode> Nodes { get; private set; } = new List<BSPNode>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many nodes are in the chunk
        int nodeCount = (int)(size / 40); // Each node is 40 bytes
        
        for (int i = 0; i < nodeCount; i++)
        {
            BSPNode node = new BSPNode
            {
                PlaneType = reader.ReadUInt32(),
                PlaneDistance = reader.ReadSingle(),
                Children = new uint[] { reader.ReadUInt32(), reader.ReadUInt32() },
                BoundingBoxMin = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                BoundingBoxMax = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Flags = reader.ReadUInt32()
            };
            
            Nodes.Add(node);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var node in Nodes)
        {
            writer.Write(node.PlaneType);
            writer.Write(node.PlaneDistance);
            writer.Write(node.Children[0]);
            writer.Write(node.Children[1]);
            writer.Write(node.BoundingBoxMin.X);
            writer.Write(node.BoundingBoxMin.Y);
            writer.Write(node.BoundingBoxMin.Z);
            writer.Write(node.BoundingBoxMax.X);
            writer.Write(node.BoundingBoxMax.Y);
            writer.Write(node.BoundingBoxMax.Z);
            writer.Write(node.Flags);
        }
    }
    
    // Helper method to traverse the BSP tree using a visitor pattern
    public void TraverseTree(int nodeIndex, IBSPVisitor visitor, Vector3 viewpoint)
    {
        if (nodeIndex >= Nodes.Count)
            return;
            
        BSPNode node = Nodes[nodeIndex];
        int side = node.ClassifyPoint(viewpoint);
        
        // Determine which child to process first based on viewpoint
        int firstChild = (side >= 0) ? 1 : 0;
        int secondChild = (side >= 0) ? 0 : 1;
        
        // Process first child
        if (!visitor.VisitNode(node, firstChild))
            return;
            
        uint firstChildIndex = node.Children[firstChild];
        if ((firstChildIndex & 0x80000000) == 0)
            TraverseTree((int)firstChildIndex, visitor, viewpoint);
        else
            visitor.VisitLeaf(firstChildIndex & 0x7FFFFFFF);
            
        // Process second child
        if (!visitor.VisitNode(node, secondChild))
            return;
            
        uint secondChildIndex = node.Children[secondChild];
        if ((secondChildIndex & 0x80000000) == 0)
            TraverseTree((int)secondChildIndex, visitor, viewpoint);
        else
            visitor.VisitLeaf(secondChildIndex & 0x7FFFFFFF);
    }
}

// Interface for BSP tree traversal using visitor pattern
public interface IBSPVisitor
{
    bool VisitNode(MOBS.BSPNode node, int childIndex);
    void VisitLeaf(uint geometryIndex);
}

public struct Vector3
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
}
```

## Version Information
- Present only in the Alpha version of the WDT format
- In later versions, BSP data was moved into individual WMO model files
- The approach to visibility determination evolved in later versions to include more sophisticated techniques

## Rendering Implications
The BSP data in MOBS significantly impacts rendering:
1. **Occlusion Culling**: Allows the engine to skip rendering objects that are hidden behind others
2. **View Frustum Culling**: Using bounding boxes to determine if objects are outside the camera's view
3. **Rendering Order**: Enables correct depth sorting for transparent objects
4. **Level of Detail**: Can incorporate different detail levels based on distance from viewer

## Architectural Significance
The MOBS chunk highlights the Alpha WDT format's approach to scene management:

1. **Integrated Scene Graph**: World geometry and visibility data contained in a single file
2. **Centralized Rendering Logic**: Visibility determination handled at the map level
3. **Performance Optimization**: BSP trees provided efficient rendering for 1999-era hardware

This contrasts with the modern approach where:
- BSP data is contained within individual WMO files
- Multiple visibility techniques are used together (BSP, portals, occlusion culling)
- Rendering decisions are more distributed across different file types and systems 