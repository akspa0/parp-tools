# CLID - MDX Collision Shapes Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The CLID (Collision Shapes) chunk defines collision geometry used for hit detection, pathfinding, and physical interactions. Unlike the detailed visual geometry in the GEOS chunk, collision shapes are simplified geometric primitives (spheres, boxes, cylinders) that approximate the model's volume for efficient collision detection. These shapes are used for gameplay mechanics such as unit selection, projectile impacts, and ability targeting.

## Structure

```csharp
public struct CLID
{
    /// <summary>
    /// Array of collision shape definitions
    /// </summary>
    // MDLCOLLISIONSHAPE shapes[numShapes] follows
}

public struct MDLCOLLISIONSHAPE
{
    /// <summary>
    /// Type of collision shape
    /// </summary>
    public uint type;
    
    /// <summary>
    /// Number of vertices in the shape
    /// </summary>
    public uint vertexCount;
    
    /// <summary>
    /// Bone ID this shape is attached to (-1 for none)
    /// </summary>
    public int boneId;
    
    /// <summary>
    /// Vertices defining the shape
    /// </summary>
    // Vector3 vertices[vertexCount] follows
    
    /// <summary>
    /// Radius of the shape (for spheres, cylinders)
    /// </summary>
    public float radius;
    
    /// <summary>
    /// Additional parameters based on shape type
    /// </summary>
    // Additional data follows based on shape type
}
```

## Properties

### MDLCOLLISIONSHAPE Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | type | uint | Type of collision shape (see Collision Shape Types) |
| 0x04 | vertexCount | uint | Number of vertices in the collision shape |
| 0x08 | boneId | int | ID of bone this shape is attached to (-1 if none) |
| 0x0C | vertices | Vector3[] | Array of vertices defining the shape |
| varies | radius | float | Radius of shape (for sphere, cylinder) |
| varies | ... | ... | Additional type-specific data |

## Collision Shape Types

| Value | Type | Description | Vertex Usage |
|-------|------|-------------|--------------|
| 0 | Sphere | Spherical collision volume | 1 vertex (center) + radius |
| 1 | Box | Box-shaped collision volume | 2 vertices (min, max corners) |
| 2 | Cylinder | Cylindrical collision volume | 2 vertices (base, top) + radius |
| 3 | Plane | Flat collision surface | 3+ vertices (defining a plane or convex shape) |
| 4 | Mesh | Simplified triangle mesh | Multiple vertices forming triangles |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Basic sphere, box, and plane shapes |
| 1300-1500 (WoW Alpha) | Added cylinder and simplified mesh types |

## Dependencies
- BONE - Collision shapes can be attached to bones via boneId
- GEOS - Visual geometry that collision shapes approximate
- SEQS - Animation sequences that may affect collision shapes attached to bones

## Implementation Notes
- Collision shapes should be simpler than visual geometry for efficient collision detection
- Shapes attached to bones via boneId will transform with the bone during animations
- The shape type determines how the vertices and radius are interpreted:
  - Sphere: 1 vertex (center) plus radius
  - Box: 2 vertices defining opposite corners of the box (min/max)
  - Cylinder: 2 vertices for the base and top, plus radius
  - Plane: 3+ vertices defining a flat surface or convex shape
  - Mesh: Multiple vertices defining a simplified collision mesh
- Multiple collision shapes can be combined to create complex collision volumes
- Collision shapes are not rendered but used for gameplay mechanics
- Shapes with boneId = -1 are in model space and don't move with animations
- For units, a cylinder is commonly used for pathfinding and selection
- For buildings, boxes and planes are typically used for collision
- Spheres are often used for approximate/quick collision tests
- Proper collision handling requires transforming shapes according to model position and rotation

## Usage Context
Collision shapes in MDX models are used for:
- Unit selection and targeting
- Pathfinding and obstacle detection
- Projectile hit detection
- Ability targeting and area effects
- Item pickup ranges
- Trigger volumes for gameplay events
- Physics interactions between units
- Line-of-sight calculations
- Defining unit "footprint" for placement
- Optimizing spatial queries in the game world

## Implementation Example

```csharp
public class CLIDChunk : IMdxChunk
{
    public string ChunkId => "CLID";
    
    public List<MdxCollisionShape> Shapes { get; private set; } = new List<MdxCollisionShape>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear existing shapes
        Shapes.Clear();
        
        // Read shapes until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var shape = new MdxCollisionShape();
            
            // Read basic shape properties
            shape.Type = reader.ReadUInt32();
            shape.VertexCount = reader.ReadUInt32();
            shape.BoneId = reader.ReadInt32();
            
            // Read vertices
            shape.Vertices = new Vector3[shape.VertexCount];
            for (int i = 0; i < shape.VertexCount; i++)
            {
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                shape.Vertices[i] = new Vector3(x, y, z);
            }
            
            // Read radius for sphere and cylinder types
            if (shape.Type == 0 || shape.Type == 2) // Sphere or Cylinder
            {
                shape.Radius = reader.ReadSingle();
            }
            
            // Read additional data based on shape type
            if (shape.Type == 4) // Mesh
            {
                shape.FaceCount = reader.ReadUInt32();
                shape.Indices = new int[shape.FaceCount * 3]; // Triangles
                
                for (int i = 0; i < shape.FaceCount * 3; i++)
                {
                    shape.Indices[i] = reader.ReadInt32();
                }
            }
            
            Shapes.Add(shape);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var shape in Shapes)
        {
            // Write basic shape properties
            writer.Write(shape.Type);
            writer.Write(shape.VertexCount);
            writer.Write(shape.BoneId);
            
            // Write vertices
            for (int i = 0; i < shape.VertexCount; i++)
            {
                writer.Write(shape.Vertices[i].X);
                writer.Write(shape.Vertices[i].Y);
                writer.Write(shape.Vertices[i].Z);
            }
            
            // Write radius for sphere and cylinder types
            if (shape.Type == 0 || shape.Type == 2) // Sphere or Cylinder
            {
                writer.Write(shape.Radius);
            }
            
            // Write additional data based on shape type
            if (shape.Type == 4) // Mesh
            {
                writer.Write(shape.FaceCount);
                
                for (int i = 0; i < shape.FaceCount * 3; i++)
                {
                    writer.Write(shape.Indices[i]);
                }
            }
        }
    }
    
    /// <summary>
    /// Performs a ray intersection test against all collision shapes
    /// </summary>
    /// <param name="rayOrigin">Origin of the ray</param>
    /// <param name="rayDirection">Direction of the ray (normalized)</param>
    /// <param name="boneTransforms">Array of bone transformation matrices</param>
    /// <returns>Distance to intersection or null if no hit</returns>
    public float? RayIntersection(Vector3 rayOrigin, Vector3 rayDirection, Matrix4x4[] boneTransforms)
    {
        float? closestHit = null;
        
        foreach (var shape in Shapes)
        {
            // Get transformation for this shape
            Matrix4x4 transform = Matrix4x4.Identity;
            if (shape.BoneId >= 0 && shape.BoneId < boneTransforms.Length)
            {
                transform = boneTransforms[shape.BoneId];
            }
            
            // Test intersection based on shape type
            float? hit = null;
            
            switch (shape.Type)
            {
                case 0: // Sphere
                    Vector3 center = Vector3.Transform(shape.Vertices[0], transform);
                    hit = IntersectRaySphere(rayOrigin, rayDirection, center, shape.Radius);
                    break;
                    
                case 1: // Box
                    Vector3 min = Vector3.Transform(shape.Vertices[0], transform);
                    Vector3 max = Vector3.Transform(shape.Vertices[1], transform);
                    hit = IntersectRayBox(rayOrigin, rayDirection, min, max);
                    break;
                    
                case 2: // Cylinder
                    Vector3 base_ = Vector3.Transform(shape.Vertices[0], transform);
                    Vector3 top = Vector3.Transform(shape.Vertices[1], transform);
                    hit = IntersectRayCylinder(rayOrigin, rayDirection, base_, top, shape.Radius);
                    break;
                    
                // Other shape types...
            }
            
            // Update closest hit
            if (hit.HasValue && (!closestHit.HasValue || hit.Value < closestHit.Value))
            {
                closestHit = hit;
            }
        }
        
        return closestHit;
    }
    
    /// <summary>
    /// Transforms all collision shapes by the given model matrix
    /// </summary>
    /// <param name="modelMatrix">Model transformation matrix</param>
    /// <param name="boneTransforms">Array of bone transformations</param>
    /// <returns>Transformed collision shapes</returns>
    public List<TransformedCollisionShape> GetTransformedShapes(Matrix4x4 modelMatrix, Matrix4x4[] boneTransforms)
    {
        var result = new List<TransformedCollisionShape>();
        
        foreach (var shape in Shapes)
        {
            var transformed = new TransformedCollisionShape();
            transformed.Type = shape.Type;
            transformed.Radius = shape.Radius;
            
            // Apply bone transform if attached to a bone
            Matrix4x4 transform = modelMatrix;
            if (shape.BoneId >= 0 && shape.BoneId < boneTransforms.Length)
            {
                transform = Matrix4x4.Multiply(boneTransforms[shape.BoneId], modelMatrix);
            }
            
            // Transform vertices
            transformed.Vertices = new Vector3[shape.Vertices.Length];
            for (int i = 0; i < shape.Vertices.Length; i++)
            {
                transformed.Vertices[i] = Vector3.Transform(shape.Vertices[i], transform);
            }
            
            result.Add(transformed);
        }
        
        return result;
    }
    
    // Ray intersection helper methods
    private float? IntersectRaySphere(Vector3 rayOrigin, Vector3 rayDirection, Vector3 center, float radius)
    {
        Vector3 oc = rayOrigin - center;
        float a = Vector3.Dot(rayDirection, rayDirection);
        float b = 2.0f * Vector3.Dot(oc, rayDirection);
        float c = Vector3.Dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0)
        {
            return null; // No intersection
        }
        
        float t = (-b - MathF.Sqrt(discriminant)) / (2.0f * a);
        if (t < 0)
        {
            t = (-b + MathF.Sqrt(discriminant)) / (2.0f * a);
        }
        
        if (t < 0)
        {
            return null; // Intersection behind ray origin
        }
        
        return t;
    }
    
    private float? IntersectRayBox(Vector3 rayOrigin, Vector3 rayDirection, Vector3 min, Vector3 max)
    {
        float tMin = float.MinValue;
        float tMax = float.MaxValue;
        
        // Check intersection with each pair of planes
        for (int i = 0; i < 3; i++)
        {
            if (Math.Abs(rayDirection[i]) < float.Epsilon)
            {
                // Ray is parallel to slab, check if origin is contained in slab
                if (rayOrigin[i] < min[i] || rayOrigin[i] > max[i])
                {
                    return null;
                }
            }
            else
            {
                // Compute intersection with slab
                float ood = 1.0f / rayDirection[i];
                float t1 = (min[i] - rayOrigin[i]) * ood;
                float t2 = (max[i] - rayOrigin[i]) * ood;
                
                if (t1 > t2)
                {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }
                
                tMin = Math.Max(tMin, t1);
                tMax = Math.Min(tMax, t2);
                
                if (tMin > tMax)
                {
                    return null;
                }
            }
        }
        
        if (tMin > 0)
        {
            return tMin;
        }
        else if (tMax > 0)
        {
            return tMax;
        }
        
        return null;
    }
    
    private float? IntersectRayCylinder(Vector3 rayOrigin, Vector3 rayDirection, Vector3 base_, Vector3 top, float radius)
    {
        // Calculate cylinder axis
        Vector3 axis = Vector3.Normalize(top - base_);
        float height = Vector3.Distance(base_, top);
        
        // Calculate nearest approach of ray to cylinder axis
        Vector3 oc = rayOrigin - base_;
        float rayDotAxis = Vector3.Dot(rayDirection, axis);
        float ocDotAxis = Vector3.Dot(oc, axis);
        
        Vector3 rayPerp = rayDirection - axis * rayDotAxis;
        Vector3 ocPerp = oc - axis * ocDotAxis;
        
        float perpDotPerp = Vector3.Dot(rayPerp, rayPerp);
        
        if (Math.Abs(perpDotPerp) < float.Epsilon)
        {
            // Ray is parallel to cylinder axis
            float d = Vector3.Dot(ocPerp, ocPerp);
            if (d > radius * radius)
            {
                return null; // No intersection
            }
            
            // Ray inside cylinder, find intersection with end caps
            float t1 = (-ocDotAxis) / rayDotAxis;
            float t2 = (height - ocDotAxis) / rayDotAxis;
            
            if (t1 > t2)
            {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }
            
            if (t2 < 0)
            {
                return null; // Cylinder behind ray
            }
            
            return t1 > 0 ? t1 : t2;
        }
        
        // Calculate intersection with infinite cylinder
        float ocPerpDotRayPerp = Vector3.Dot(ocPerp, rayPerp);
        float a = perpDotPerp;
        float b = 2.0f * ocPerpDotRayPerp;
        float c = Vector3.Dot(ocPerp, ocPerp) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0)
        {
            return null; // No intersection with infinite cylinder
        }
        
        // Calculate intersection points
        float t = (-b - MathF.Sqrt(discriminant)) / (2.0f * a);
        if (t < 0)
        {
            t = (-b + MathF.Sqrt(discriminant)) / (2.0f * a);
        }
        
        if (t < 0)
        {
            return null; // Intersection behind ray origin
        }
        
        // Check if intersection point is within cylinder height
        float hitAxis = ocDotAxis + t * rayDotAxis;
        if (hitAxis < 0 || hitAxis > height)
        {
            // Intersection with infinite cylinder outside height range
            // Check for intersection with end caps
            float t1 = (-ocDotAxis) / rayDotAxis;
            float t2 = (height - ocDotAxis) / rayDotAxis;
            
            if (t1 > 0)
            {
                Vector3 p = oc + rayDirection * t1;
                if (Vector3.Dot(p - axis * ocDotAxis, p - axis * ocDotAxis) <= radius * radius)
                {
                    return t1;
                }
            }
            
            if (t2 > 0)
            {
                Vector3 p = oc + rayDirection * t2;
                if (Vector3.Dot(p - axis * (ocDotAxis + height), p - axis * (ocDotAxis + height)) <= radius * radius)
                {
                    return t2;
                }
            }
            
            return null;
        }
        
        return t;
    }
}

public class MdxCollisionShape
{
    public uint Type { get; set; }
    public uint VertexCount { get; set; }
    public int BoneId { get; set; }
    public Vector3[] Vertices { get; set; }
    public float Radius { get; set; }
    
    // Additional data for mesh type
    public uint FaceCount { get; set; }
    public int[] Indices { get; set; }
    
    /// <summary>
    /// Gets a friendly name for the shape type
    /// </summary>
    public string TypeName
    {
        get
        {
            switch (Type)
            {
                case 0: return "Sphere";
                case 1: return "Box";
                case 2: return "Cylinder";
                case 3: return "Plane";
                case 4: return "Mesh";
                default: return $"Unknown ({Type})";
            }
        }
    }
    
    /// <summary>
    /// Checks if the shape is attached to a bone
    /// </summary>
    public bool IsAttachedToBone => BoneId >= 0;
    
    /// <summary>
    /// Calculates the bounding box of the shape
    /// </summary>
    public (Vector3 Min, Vector3 Max) GetBoundingBox()
    {
        if (Vertices == null || Vertices.Length == 0)
        {
            return (Vector3.Zero, Vector3.Zero);
        }
        
        Vector3 min = Vertices[0];
        Vector3 max = Vertices[0];
        
        for (int i = 1; i < Vertices.Length; i++)
        {
            min = Vector3.Min(min, Vertices[i]);
            max = Vector3.Max(max, Vertices[i]);
        }
        
        // Expand by radius for sphere and cylinder
        if (Type == 0 || Type == 2) // Sphere or Cylinder
        {
            min -= new Vector3(Radius);
            max += new Vector3(Radius);
        }
        
        return (min, max);
    }
}

public class TransformedCollisionShape
{
    public uint Type { get; set; }
    public Vector3[] Vertices { get; set; }
    public float Radius { get; set; }
    
    // Additional methods for collision detection with the transformed shape
}
``` 