# MOBN - WMO Group BSP Tree

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOBN (Map Object BsP Nodes) chunk contains the binary space partitioning (BSP) tree structure used for collision detection and spatial organization of the WMO group. Each node in the tree represents a division of space, allowing for efficient spatial queries such as ray casting, collision detection, and visibility determination. The BSP tree divides the 3D space of the WMO group into convex subspaces, creating a hierarchical structure that can be traversed to find specific triangles or determine if a point is inside the model.

## Structure

```csharp
public struct MOBN
{
    public MOBNNode[] nodes; // Array of BSP tree nodes
}

public struct MOBNNode
{
    public ushort flags;        // Bit flags (see Flags table)
    public short negChild;      // Index of child node on negative side of plane
    public short posChild;      // Index of child node on positive side of plane
    public ushort nFaces;       // Number of triangle faces
    public uint faceStart;      // Index of the first triangle index in MOBR
    public float planeDist;     // Distance of the splitting plane from origin
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | flags | ushort | Bit flags defining the node type and plane (see Flags table) |
| 0x02 | negChild | short | Index of child node on negative side of plane, or -1 if no child |
| 0x04 | posChild | short | Index of child node on positive side of plane, or -1 if no child |
| 0x06 | nFaces | ushort | Number of triangle faces referenced by this node |
| 0x08 | faceStart | uint | Index of the first triangle index in the MOBR chunk |
| 0x0C | planeDist | float | Distance of the splitting plane from the origin (0,0,0) |

## Flags (Offset 0x00)

| Value | Name | Description |
|-------|------|-------------|
| 0x0 | FLAG_X_AXIS | Node splits space along YZ-plane (X-axis normal) |
| 0x1 | FLAG_Y_AXIS | Node splits space along XZ-plane (Y-axis normal) |
| 0x2 | FLAG_Z_AXIS | Node splits space along XY-plane (Z-axis normal) |
| 0x3 | FLAG_AXIS_MASK | Mask to extract the axis flags |
| 0x4 | FLAG_LEAF | Node is a leaf node (contains triangles, no children) |
| 0xFFFF | FLAG_NO_CHILD | Indicates no child node (-1 in child indices) |

## Dependencies
- **MOBR**: Contains the indices of triangles referenced by BSP leaf nodes
- **MOVI**: Contains the actual triangle indices that MOBR references
- **MOVT**: Contains the vertex positions needed to define the triangles
- **MOGP**: The flag 0x00000001 (HasBSP) in the MOGP header indicates that this chunk is present

## Implementation Notes
- Each node is 16 bytes (0x10) in size.
- The flags field combines both the axis information (bits 0-1) and the leaf flag (bit 2).
- For internal nodes (non-leaves), the axis flags (0-2) indicate which axis the splitting plane is perpendicular to.
- The planeDist value defines where along the chosen axis the splitting plane is located, relative to the origin (0,0,0) of the whole WMO.
- Leaf nodes (flag 0x4 set) contain triangle indices and have no children.
- The BSP tree is typically traversed from the root (node 0), with the root's bounding box being the entire WMO group's bounding box from the MOGP header.
- The negChild points to the node on the negative side of the splitting plane, while posChild points to the node on the positive side.
- A value of -1 (0xFFFF) for negChild or posChild indicates no child exists on that side.
- The combination of faceStart and nFaces defines a range of indices in the MOBR chunk that contains the triangles for a leaf node.
- Some BSP trees can be quite large, with leaf nodes containing up to 2100 faces in older versions, with possible limits of 8192 faces in newer versions.
- When traversing the BSP tree for collision detection or ray casting, you recursively check which side of the splitting plane your ray or point falls on, and follow that path through the tree.

## Implementation Example

```csharp
public class MOBNChunk : IWmoGroupChunk
{
    public string ChunkId => "MOBN";
    public List<BSPNode> Nodes { get; private set; } = new List<BSPNode>();

    public void Parse(BinaryReader reader, long size)
    {
        // Each node is 16 bytes
        int nodeCount = (int)(size / 16);
        
        for (int i = 0; i < nodeCount; i++)
        {
            BSPNode node = new BSPNode
            {
                Flags = reader.ReadUInt16(),
                NegativeChildIndex = reader.ReadInt16(),
                PositiveChildIndex = reader.ReadInt16(),
                NumFaces = reader.ReadUInt16(),
                FaceStartIndex = reader.ReadUInt32(),
                PlaneDistance = reader.ReadSingle()
            };
            
            Nodes.Add(node);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 16 != 0)
        {
            throw new InvalidDataException("MOBN chunk size is not a multiple of 16 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var node in Nodes)
        {
            writer.Write(node.Flags);
            writer.Write(node.NegativeChildIndex);
            writer.Write(node.PositiveChildIndex);
            writer.Write(node.NumFaces);
            writer.Write(node.FaceStartIndex);
            writer.Write(node.PlaneDistance);
        }
    }
    
    // Helper method to determine if a point is inside the BSP tree
    public bool IsPointInside(Vector3 point, List<ushort> mobrIndices, List<ushort> moviIndices, List<Vector3> vertices)
    {
        return IsPointInsideNode(0, point, mobrIndices, moviIndices, vertices);
    }
    
    private bool IsPointInsideNode(int nodeIndex, Vector3 point, List<ushort> mobrIndices, List<ushort> moviIndices, List<Vector3> vertices)
    {
        if (nodeIndex < 0 || nodeIndex >= Nodes.Count)
            return false;
            
        BSPNode node = Nodes[nodeIndex];
        
        // If this is a leaf node, check if the point is inside any of its triangles
        if ((node.Flags & 0x4) != 0)
        {
            // Check triangles in this leaf
            for (uint i = 0; i < node.NumFaces; i++)
            {
                uint mobrIndex = node.FaceStartIndex + i;
                if (mobrIndex >= mobrIndices.Count)
                    continue;
                    
                ushort triangleIndex = mobrIndices[(int)mobrIndex];
                // Get the three vertices of this triangle from MOVI and MOVT
                // Then check if the point is inside this triangle
                // This is a simplified example, actual triangle containment testing would be more complex
            }
            
            return false; // Not inside any triangle in this leaf
        }
        
        // Internal node, determine which side of the splitting plane the point is on
        int axis = node.Flags & 0x3;
        float pointValue = 0;
        
        switch (axis)
        {
            case 0: pointValue = point.X; break; // X-axis
            case 1: pointValue = point.Y; break; // Y-axis
            case 2: pointValue = point.Z; break; // Z-axis
        }
        
        // Check which side of the plane the point is on
        if (pointValue <= node.PlaneDistance)
        {
            // Negative side
            if (node.NegativeChildIndex != -1)
                return IsPointInsideNode(node.NegativeChildIndex, point, mobrIndices, moviIndices, vertices);
        }
        else
        {
            // Positive side
            if (node.PositiveChildIndex != -1)
                return IsPointInsideNode(node.PositiveChildIndex, point, mobrIndices, moviIndices, vertices);
        }
        
        return false;
    }
    
    public class BSPNode
    {
        public ushort Flags { get; set; }
        public short NegativeChildIndex { get; set; }
        public short PositiveChildIndex { get; set; }
        public ushort NumFaces { get; set; }
        public uint FaceStartIndex { get; set; }
        public float PlaneDistance { get; set; }
        
        public bool IsLeaf => (Flags & 0x4) != 0;
        public int PlaneType => Flags & 0x3;
    }
}
```

## Usage Context
- The BSP tree is used for efficient collision detection between players, NPCs, or objects and the WMO geometry.
- During rendering, the BSP tree can be used for visibility determination and occlusion culling.
- The tree structure enables fast ray casting for line-of-sight checks or projectile collision detection.
- For physics simulations, the BSP tree helps determine when and where objects collide with the WMO geometry.
- The BSP structure is essential for large, complex WMOs with many triangles, as it provides logarithmic search times instead of linear checks against all triangles. 