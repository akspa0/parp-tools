# ACDO - Chunk Doodad/Object Definitions

## Type
ADT v22 ACNK Subchunk

## Source
Referenced from `ADT_v22.md`

## Description
The ACDO (Chunk Doodad/Object) subchunk contains placement information for models (both M2 doodads and WMO objects) within a specific terrain chunk. Unlike the v18 format which uses separate MDDF and MODF chunks for M2 and WMO models respectively, the v22 format consolidates both model types into a single ACDO subchunk, simplifying the object placement system.

## Structure

```csharp
public struct ACDO
{
    public int modelID;          // Index into ADOO array for the model filename
    public float position[3];    // XYZ position for the model placement
    public float rotation[3];    // XYZ rotation angles in degrees
    public float scale[3];       // XYZ scale factors
    public uint uniqueId;        // Unique identifier for this object instance
    // Additional data may follow (name/doodadsets?)
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| modelID | int | Index into the ADOO array for the model to use |
| position | float[3] | XYZ position coordinates in world space |
| rotation | float[3] | XYZ rotation angles in degrees |
| scale | float[3] | XYZ scale factors for the model |
| uniqueId | uint | Unique identifier for this object instance |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk
- ADOO (C005) - Contains model filenames referenced by modelID

## Implementation Notes

1. The ACDO subchunk combines the functionality of both MDDF (M2 doodad placement) and MODF (WMO object placement) from the v18 format.

2. Each ACDO entry references a model in the ADOO chunk by using the modelID index.

3. Unlike v18 where scale is specified as a single value, v22 uses separate scale factors for X, Y, and Z, allowing for non-uniform scaling.

4. The position values use the same coordinate system as MDDF/MODF in v18.

5. The uniqueId should be unique across all object instances in the entire map.

6. The v22 documentation mentions that additional data may follow each ACDO entry, possibly related to doodad sets or additional naming information, but the exact format is not specified.

7. The optional size of the ACDO chunk is listed as 0x38 (56 bytes) in the documentation, which allows for the defined fields plus some additional data.

## Implementation Example

```csharp
public class AcdoSubChunk
{
    public class ObjectDefinition
    {
        public int ModelID { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Rotation { get; set; }
        public Vector3 Scale { get; set; }
        public uint UniqueId { get; set; }
        public byte[] AdditionalData { get; set; } // Any additional data that might follow
        
        // Helper to convert to world space matrix
        public Matrix4x4 GetPlacementMatrix()
        {
            // Constants for coordinate conversion
            const float TILESIZE = 533.33333f;
            
            // Create translation matrix
            float worldX = 32 * TILESIZE - Position.X;
            float worldY = Position.Y;
            float worldZ = 32 * TILESIZE - Position.Z;
            
            // Combine transforms: apply rotations then translation
            Matrix4x4 matrix = Matrix4x4.Identity;
            
            // Apply standard coordinate system transformation
            matrix *= Matrix4x4.CreateRotationX(MathHelper.ToRadians(90));
            matrix *= Matrix4x4.CreateRotationY(MathHelper.ToRadians(90));
            
            // Apply translation
            matrix *= Matrix4x4.CreateTranslation(worldX, worldY, worldZ);
            
            // Apply rotations in correct order
            matrix *= Matrix4x4.CreateRotationY(MathHelper.ToRadians(Rotation.Y - 270));
            matrix *= Matrix4x4.CreateRotationZ(MathHelper.ToRadians(-Rotation.X));
            matrix *= Matrix4x4.CreateRotationX(MathHelper.ToRadians(Rotation.Z - 90));
            
            // Apply scaling
            matrix *= Matrix4x4.CreateScale(Scale.X, Scale.Y, Scale.Z);
            
            return matrix;
        }
    }
    
    // Collection of all object definitions in this chunk
    public List<ObjectDefinition> Objects { get; private set; } = new List<ObjectDefinition>();
    
    public AcdoSubChunk()
    {
    }
    
    public void Load(BinaryReader reader, long size)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        // Read object definitions until we reach the end of the chunk
        while (reader.BaseStream.Position + 56 <= endPosition) // Each object is at least 56 bytes
        {
            var objDef = new ObjectDefinition();
            
            // Read the basic object data
            objDef.ModelID = reader.ReadInt32();
            
            // Read position, rotation, scale vectors
            objDef.Position = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            objDef.Rotation = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            objDef.Scale = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            objDef.UniqueId = reader.ReadUInt32();
            
            // Calculate how much additional data might be available
            long remainingBytes = endPosition - reader.BaseStream.Position;
            
            // Check if there's additional data (documentation mentions possible doodad sets)
            if (remainingBytes > 0 && remainingBytes < 56) // Not enough for another full object
            {
                objDef.AdditionalData = reader.ReadBytes((int)remainingBytes);
                break; // We've read all the data
            }
            
            Objects.Add(objDef);
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ACDO".ToCharArray());
        
        // Calculate size: 56 bytes per object + additional data length
        int totalSize = 0;
        foreach (var obj in Objects)
        {
            totalSize += 56; // Base object size
            if (obj.AdditionalData != null)
                totalSize += obj.AdditionalData.Length;
        }
        
        writer.Write(totalSize);
        
        // Write each object
        foreach (var obj in Objects)
        {
            writer.Write(obj.ModelID);
            
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
            
            writer.Write(obj.UniqueId);
            
            // Write any additional data
            if (obj.AdditionalData != null)
                writer.Write(obj.AdditionalData);
        }
    }
    
    // Helper to get the model filename for an object from the ADOO chunk
    public string GetModelFilename(int objectIndex, AdooChunk adooChunk)
    {
        if (objectIndex < 0 || objectIndex >= Objects.Count)
            throw new ArgumentOutOfRangeException(nameof(objectIndex));
            
        if (adooChunk == null)
            throw new ArgumentNullException(nameof(adooChunk));
            
        int modelId = Objects[objectIndex].ModelID;
        return adooChunk.GetFilename(modelId);
    }
}
```

## Usage Context

The ACDO subchunk is responsible for populating the terrain with various decorative objects and structures. This includes:

1. **Environmental Doodads**: Trees, rocks, bushes, grass clumps, fallen logs, and other small details that enhance the environment's visual richness.

2. **World Structures**: Buildings, bridges, towers, walls, and other larger architectural elements that define locations in the game world.

3. **Props and Decorations**: Furniture, signposts, lanterns, fences, and other objects that add character to areas.

The v22 format's approach to object placement represents an interesting evolution from v18, with several key differences:

1. **Unified Model Types**: Both M2 doodads and WMO objects are handled by the same chunk type, simplifying the object placement system.

2. **Per-Axis Scaling**: Unlike v18 which uses uniform scaling, v22 allows for non-uniform scaling along each axis, enabling more flexible model transformations.

3. **Localized Placement**: Objects are defined within the specific ACNK chunk where they appear, rather than in global lists, potentially improving loading performance for large maps.

This unified approach to object placement suggests that Blizzard was experimenting with more efficient ways to organize world data in the Cataclysm beta. While ultimately not used in the final release, this design offers insights into alternative approaches to terrain decoration that might have influenced later developments in World of Warcraft's environmental design. 