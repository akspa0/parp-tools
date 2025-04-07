# ACDO - Chunk Doodad/Object Definitions

## Type
ADT v23 ACNK Subchunk

## Source
Referenced from `ADT_v23.md`

## Description
The ACDO (Chunk Doodad/Object) subchunk contains information about the placement of models (both M2 doodads and WMO objects) within a specific terrain chunk. This subchunk defines the position, rotation, scale, and other properties of each model instance in the chunk. It serves a similar function to the MCRD subchunk in ADT v18 but with an updated structure and additional properties.

## Structure

```csharp
public struct ACDO
{
    public uint modelId;       // Index into the ADOO chunk
    public float position[3];  // X, Y, Z coordinates
    public float rotation[3];  // Rotation angles in radians (X, Y, Z)
    public float scale[3];     // Scale factors for X, Y, Z axes
    public uint flags;         // Various object flags
    public uint uniqueId;      // Unique identifier for this instance
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| modelId | uint | Index into the ADOO chunk's model filename array |
| position | float[3] | Position coordinates (X, Y, Z) of the model in world space |
| rotation | float[3] | Rotation angles in radians around X, Y, and Z axes |
| scale | float[3] | Scale factors for the model along X, Y, and Z axes |
| flags | uint | Bit flags controlling various aspects of the model (see table below) |
| uniqueId | uint | Unique identifier for this specific model instance |

### Object Flags

| Flag | Value | Description |
|------|-------|-------------|
| BIODOME | 0x01 | Used in biodomes (special interior spaces) |
| SHRUBBERY | 0x02 | Object is shrub-like vegetation, affected by shaders differently |
| ANIMATE | 0x04 | Object should be animated |
| BILLBOARD | 0x08 | Object should always face the camera |
| NO_SHADOW | 0x10 | Object doesn't cast shadows |
| NO_CULLING | 0x20 | Object is never culled by distance |
| LIGHT_SOURCE | 0x40 | Object emits light (used for post-processing effects) |
| NO_COLLISION | 0x80 | Object doesn't have collision |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk
- ADOO (C005) - Contains the model filenames referenced by modelId

## Implementation Notes

1. The ACDO subchunk defines multiple model placements within a single terrain chunk, with each entry having a fixed size of 32 bytes.

2. The number of objects in the subchunk can be calculated by dividing the subchunk size (minus the header) by 32.

3. Unlike the v18 format, which has separate chunks for M2 doodads (MDDF) and WMO objects (MODF), the v23 format combines these into a single ACDO subchunk per terrain chunk.

4. The modelId refers to an index in the ADOO chunk, which contains the actual filenames for M2 and WMO models.

5. The position is relative to the world space, not the chunk space, meaning absolute coordinates are used.

6. The rotation array specifies rotation angles in radians around the X, Y, and Z axes, applied in that order.

7. The scale array allows for non-uniform scaling along each axis, unlike the v18 format which only supported uniform scaling.

8. The uniqueId field provides a way to specifically identify individual model instances, which is useful for scripting, phasing, and other game mechanics.

9. The flags field controls various rendering properties and behaviors of the model.

## Implementation Example

```csharp
public class AcdoSubchunk
{
    // Constants
    private const int ACDO_ENTRY_SIZE = 32; // Size of each ACDO entry in bytes
    
    // Flags enumeration
    public static class ObjectFlags
    {
        public const uint BIODOME = 0x01;
        public const uint SHRUBBERY = 0x02;
        public const uint ANIMATE = 0x04;
        public const uint BILLBOARD = 0x08;
        public const uint NO_SHADOW = 0x10;
        public const uint NO_CULLING = 0x20;
        public const uint LIGHT_SOURCE = 0x40;
        public const uint NO_COLLISION = 0x80;
    }
    
    // Class to represent a single object placement
    public class ObjectDefinition
    {
        public uint ModelId { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Rotation { get; set; }
        public Vector3 Scale { get; set; }
        public uint Flags { get; set; }
        public uint UniqueId { get; set; }
        
        // Helper properties
        public bool IsBiodome => (Flags & ObjectFlags.BIODOME) != 0;
        public bool IsShrubbery => (Flags & ObjectFlags.SHRUBBERY) != 0;
        public bool IsAnimated => (Flags & ObjectFlags.ANIMATE) != 0;
        public bool IsBillboard => (Flags & ObjectFlags.BILLBOARD) != 0;
        public bool HasNoShadow => (Flags & ObjectFlags.NO_SHADOW) != 0;
        public bool HasNoCulling => (Flags & ObjectFlags.NO_CULLING) != 0;
        public bool IsLightSource => (Flags & ObjectFlags.LIGHT_SOURCE) != 0;
        public bool HasNoCollision => (Flags & ObjectFlags.NO_COLLISION) != 0;
    }
    
    // List of object definitions in this chunk
    public List<ObjectDefinition> Objects { get; private set; } = new List<ObjectDefinition>();
    
    public AcdoSubchunk()
    {
    }
    
    public void Load(BinaryReader reader, uint size)
    {
        int objectCount = (int)size / ACDO_ENTRY_SIZE;
        
        Objects.Clear();
        
        for (int i = 0; i < objectCount; i++)
        {
            ObjectDefinition obj = new ObjectDefinition();
            
            obj.ModelId = reader.ReadUInt32();
            
            // Read position
            float posX = reader.ReadSingle();
            float posY = reader.ReadSingle();
            float posZ = reader.ReadSingle();
            obj.Position = new Vector3(posX, posY, posZ);
            
            // Read rotation
            float rotX = reader.ReadSingle();
            float rotY = reader.ReadSingle();
            float rotZ = reader.ReadSingle();
            obj.Rotation = new Vector3(rotX, rotY, rotZ);
            
            // Read scale
            float scaleX = reader.ReadSingle();
            float scaleY = reader.ReadSingle();
            float scaleZ = reader.ReadSingle();
            obj.Scale = new Vector3(scaleX, scaleY, scaleZ);
            
            obj.Flags = reader.ReadUInt32();
            obj.UniqueId = reader.ReadUInt32();
            
            Objects.Add(obj);
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ACDO".ToCharArray());
        writer.Write(Objects.Count * ACDO_ENTRY_SIZE);
        
        foreach (ObjectDefinition obj in Objects)
        {
            writer.Write(obj.ModelId);
            
            // Write position
            writer.Write(obj.Position.X);
            writer.Write(obj.Position.Y);
            writer.Write(obj.Position.Z);
            
            // Write rotation
            writer.Write(obj.Rotation.X);
            writer.Write(obj.Rotation.Y);
            writer.Write(obj.Rotation.Z);
            
            // Write scale
            writer.Write(obj.Scale.X);
            writer.Write(obj.Scale.Y);
            writer.Write(obj.Scale.Z);
            
            writer.Write(obj.Flags);
            writer.Write(obj.UniqueId);
        }
    }
    
    // Add a new object to the chunk
    public void AddObject(uint modelId, Vector3 position, Vector3 rotation, Vector3 scale, uint flags, uint uniqueId)
    {
        ObjectDefinition obj = new ObjectDefinition
        {
            ModelId = modelId,
            Position = position,
            Rotation = rotation,
            Scale = scale,
            Flags = flags,
            UniqueId = uniqueId
        };
        
        Objects.Add(obj);
    }
    
    // Remove an object by its unique ID
    public bool RemoveObject(uint uniqueId)
    {
        return Objects.RemoveAll(obj => obj.UniqueId == uniqueId) > 0;
    }
    
    // Find all objects of a specific model type
    public List<ObjectDefinition> FindObjectsByModelId(uint modelId)
    {
        return Objects.FindAll(obj => obj.ModelId == modelId);
    }
    
    // Find an object by its unique ID
    public ObjectDefinition FindObjectByUniqueId(uint uniqueId)
    {
        return Objects.Find(obj => obj.UniqueId == uniqueId);
    }
    
    // Convert degrees to radians for rotation input
    public Vector3 DegreesToRadians(Vector3 degrees)
    {
        return new Vector3(
            degrees.X * (float)Math.PI / 180.0f,
            degrees.Y * (float)Math.PI / 180.0f,
            degrees.Z * (float)Math.PI / 180.0f
        );
    }
    
    // Get a transformation matrix for an object
    public Matrix4x4 GetTransformationMatrix(ObjectDefinition obj)
    {
        // Create rotation matrix
        Matrix4x4 rotationX = Matrix4x4.CreateRotationX(obj.Rotation.X);
        Matrix4x4 rotationY = Matrix4x4.CreateRotationY(obj.Rotation.Y);
        Matrix4x4 rotationZ = Matrix4x4.CreateRotationZ(obj.Rotation.Z);
        
        // Create scale matrix
        Matrix4x4 scale = Matrix4x4.CreateScale(obj.Scale);
        
        // Create translation matrix
        Matrix4x4 translation = Matrix4x4.CreateTranslation(obj.Position);
        
        // Combine transformations
        return scale * rotationX * rotationY * rotationZ * translation;
    }
}
```

## Usage Context

The ACDO subchunk is essential for populating the World of Warcraft environment with various decorative and interactive elements:

1. **Environmental Decorations**: Defines the placement of trees, rocks, bushes, and other natural elements that give the world its detailed appearance.

2. **Architecture and Structures**: Places buildings, ruins, walls, bridges, and other constructed elements in the world.

3. **Props and Furniture**: Positions smaller objects like furniture, tools, weapons, and other props that add realism and storytelling elements.

4. **Local Culling**: By organizing object placements into per-chunk ACDO subchunks, the game can efficiently load and render only the objects in visible terrain chunks.

5. **Unique Object Identification**: The uniqueId field allows specific instances to be referenced by game scripts, phased content, or quest mechanics.

6. **Advanced Object Behavior**: The flags field enables specialized rendering and gameplay behaviors for different object types.

The ACDO subchunk in v23 represents an evolution from the separate MDDF and MODF chunks in v18, consolidating all model placements into per-chunk collections with enhanced properties like non-uniform scaling and unique identifiers. This approach aligns with v23's general theme of organizing data more locally within chunk containers, potentially improving memory access patterns during rendering. Though never used in a retail release, these experimental changes provide insight into Blizzard's exploration of alternative terrain data organization during the Cataclysm beta development period. 