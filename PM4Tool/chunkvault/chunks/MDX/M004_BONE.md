# BONE - MDX Bone Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The BONE chunk defines the skeletal structure of the MDX model. Each bone serves as a transform node in the skeletal hierarchy and can influence vertex positions through skinning. Bones contain animation data for translation, rotation, and scaling, allowing for complex articulated animations.

## Structure

```csharp
public struct BONE
{
    /// <summary>
    /// Array of bone definitions
    /// </summary>
    // Array of MDLBONE structures, count determined by numBones in MODL chunk
}

public struct MDLBONE
{
    /// <summary>
    /// Base object structure (name, IDs, flags)
    /// </summary>
    public MDLGENOBJECT baseObject;
    
    /// <summary>
    /// Bone's geoset ID (-1 if not associated with a specific geoset)
    /// </summary>
    public uint geosetId;
    
    /// <summary>
    /// Geoset animation ID (-1 if not associated with a specific geoset animation)
    /// </summary>
    public uint geosetAnimId;
    
    /// <summary>
    /// Translation animation track
    /// </summary>
    // MDLKEYTRACK<Vector3> translationTrack follows
    
    /// <summary>
    /// Rotation animation track
    /// </summary>
    // MDLKEYTRACK<Quaternion> rotationTrack follows
    
    /// <summary>
    /// Scaling animation track
    /// </summary>
    // MDLKEYTRACK<Vector3> scalingTrack follows
}
```

## Properties

### BONE Chunk
The BONE chunk itself has no properties beyond the bone array. Its size is determined by the number of MDLBONE structures it contains, which should match the numBones value in the MODL chunk.

### MDLBONE Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | baseObject | MDLGENOBJECT | Base object containing name, IDs, and flags |
| varies | geosetId | uint | Geoset this bone is associated with (0xFFFFFFFF = none) |
| varies+4 | geosetAnimId | uint | Geoset animation this bone is associated with (0xFFFFFFFF = none) |
| varies+8 | ... | ... | Animation tracks (translation, rotation, scaling) |

## Bone Flags (from MDLGENOBJECT)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | DontInheritTranslation | Don't inherit translation from parent bone |
| 1 | DontInheritRotation | Don't inherit rotation from parent bone |
| 2 | DontInheritScaling | Don't inherit scaling from parent bone |
| 3 | Billboarded | Bone is always oriented toward the camera |
| 4 | BillboardedLockX | Bone is billboarded, but locked on X axis |
| 5 | BillboardedLockY | Bone is billboarded, but locked on Y axis |
| 6 | BillboardedLockZ | Bone is billboarded, but locked on Z axis |
| 7 | CameraAnchored | Bone's transformation is relative to the camera |
| 8-31 | Reserved | Reserved for future use, typically set to 0 |

## Animation Tracks

Each bone contains three animation tracks:

1. **Translation Track**: Keyframes define bone position over time
   - Structure: MDLKEYTRACK<Vector3>
   - Contains position values (x, y, z)

2. **Rotation Track**: Keyframes define bone orientation over time
   - Structure: MDLKEYTRACK<Quaternion>
   - Contains rotation quaternions (x, y, z, w)
   - In WoW versions (1300-1500), may use compressed quaternions

3. **Scaling Track**: Keyframes define bone scale over time
   - Structure: MDLKEYTRACK<Vector3>
   - Contains scale values (x, y, z)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | May use compressed quaternions for rotation tracks |

## Dependencies
- MODL - Provides the number of bones that should be in the bone array
- MDLGENOBJECT - Base structure for bone properties
- MDLKEYTRACK - Structure for animation tracks
- GEOS - Referenced by geosetId
- GEOA - Referenced by geosetAnimId

## Implementation Notes
- Bones form a hierarchy defined by the parentId field in the baseObject
- A parentId of 0xFFFFFFFF indicates a root bone (no parent)
- The numBones field in the MODL chunk must match the actual number of bones in this chunk
- The translation, rotation, and scaling tracks provide the bone's transformation over time
- If a track has no keyframes, the bone has no animation for that property
- The billboarded flags make bones (and attached geometry) always face the camera
- Bones can be associated with specific geosets or geoset animations through the ID fields
- The model's vertex skinning data (in the GEOS chunk) references bones by their index in this array

## Usage Context
The BONE chunk provides:
- The skeletal structure of the model
- Transformation data for animating model parts
- Hierarchical relationships between model components
- Support for vertex skinning/deformation
- Camera-facing or billboarded elements

## Skeletal Animation System
The bone system works as follows:
- Bones are organized in a hierarchical tree
- Each bone's transform is relative to its parent (modified by inheritance flags)
- Animation tracks define how bones move over time
- Vertices can be influenced by multiple bones through skinning weights
- The combination of bone transformations and skinning creates complex deformations

## Implementation Example

```csharp
public class BONEChunk : IMdxChunk
{
    public string ChunkId => "BONE";
    
    public List<MdxBone> Bones { get; private set; } = new List<MdxBone>();
    
    public void Parse(BinaryReader reader, long size, uint version)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        // Clear any existing bones
        Bones.Clear();
        
        // Read bones until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var bone = new MdxBone();
            
            // Parse the base object (MDLGENOBJECT)
            bone.ParseBaseObject(reader);
            
            // Read geoset and geoset animation IDs
            bone.GeosetId = reader.ReadUInt32();
            bone.GeosetAnimId = reader.ReadUInt32();
            
            // Parse translation track
            bone.TranslationTrack = new MdxKeyTrack<Vector3>();
            bone.TranslationTrack.Parse(reader, r => new Vector3(
                r.ReadSingle(),
                r.ReadSingle(),
                r.ReadSingle()
            ));
            
            // Parse rotation track
            bone.RotationTrack = new MdxKeyTrack<Quaternion>();
            
            // Handle different quaternion formats based on version
            if (version >= 1300 && version <= 1500) // WoW versions
            {
                bone.RotationTrack.Parse(reader, r =>
                {
                    // For WoW versions, this might use compressed quaternions
                    // This is a simplified example - compression would need specific handling
                    return new Quaternion(
                        r.ReadSingle(),
                        r.ReadSingle(),
                        r.ReadSingle(),
                        r.ReadSingle()
                    );
                });
            }
            else // WC3 versions
            {
                bone.RotationTrack.Parse(reader, r => new Quaternion(
                    r.ReadSingle(),
                    r.ReadSingle(),
                    r.ReadSingle(),
                    r.ReadSingle()
                ));
            }
            
            // Parse scaling track
            bone.ScalingTrack = new MdxKeyTrack<Vector3>();
            bone.ScalingTrack.Parse(reader, r => new Vector3(
                r.ReadSingle(),
                r.ReadSingle(),
                r.ReadSingle()
            ));
            
            Bones.Add(bone);
        }
    }
    
    public void Write(BinaryWriter writer, uint version)
    {
        foreach (var bone in Bones)
        {
            // Write the base object (MDLGENOBJECT)
            bone.WriteBaseObject(writer);
            
            // Write geoset and geoset animation IDs
            writer.Write(bone.GeosetId);
            writer.Write(bone.GeosetAnimId);
            
            // Write translation track
            bone.TranslationTrack.Write(writer, (w, v) =>
            {
                w.Write(v.X);
                w.Write(v.Y);
                w.Write(v.Z);
            });
            
            // Write rotation track
            if (version >= 1300 && version <= 1500) // WoW versions
            {
                bone.RotationTrack.Write(writer, (w, q) =>
                {
                    // For WoW versions, this might use compressed quaternions
                    // This is a simplified example - compression would need specific handling
                    w.Write(q.X);
                    w.Write(q.Y);
                    w.Write(q.Z);
                    w.Write(q.W);
                });
            }
            else // WC3 versions
            {
                bone.RotationTrack.Write(writer, (w, q) =>
                {
                    w.Write(q.X);
                    w.Write(q.Y);
                    w.Write(q.Z);
                    w.Write(q.W);
                });
            }
            
            // Write scaling track
            bone.ScalingTrack.Write(writer, (w, v) =>
            {
                w.Write(v.X);
                w.Write(v.Y);
                w.Write(v.Z);
            });
        }
    }
}

public class MdxBone : MdxGenericObject
{
    // Specific bone properties
    public uint GeosetId { get; set; }
    public uint GeosetAnimId { get; set; }
    
    // Animation tracks
    public MdxKeyTrack<Vector3> TranslationTrack { get; set; }
    public MdxKeyTrack<Quaternion> RotationTrack { get; set; }
    public MdxKeyTrack<Vector3> ScalingTrack { get; set; }
    
    // Bone-specific flag accessors
    public bool Billboarded => (Flags & 0x8) != 0;
    public bool BillboardedLockX => (Flags & 0x10) != 0;
    public bool BillboardedLockY => (Flags & 0x20) != 0;
    public bool BillboardedLockZ => (Flags & 0x40) != 0;
    public bool CameraAnchored => (Flags & 0x80) != 0;
    
    public bool HasNoGeoset => GeosetId == 0xFFFFFFFF;
    public bool HasNoGeosetAnim => GeosetAnimId == 0xFFFFFFFF;
    
    // Implementation of abstract methods from MdxGenericObject
    public override void ParseAnimationData(BinaryReader reader, uint version)
    {
        GeosetId = reader.ReadUInt32();
        GeosetAnimId = reader.ReadUInt32();
        
        // Parse animation tracks (implementation omitted for brevity)
    }
    
    public override void WriteAnimationData(BinaryWriter writer, uint version)
    {
        writer.Write(GeosetId);
        writer.Write(GeosetAnimId);
        
        // Write animation tracks (implementation omitted for brevity)
    }
    
    // Helper methods for bone transformations
    public Matrix4x4 GetTransformAtTime(uint time, uint sequenceIndex, Dictionary<uint, uint> globalSequences)
    {
        // Calculate bone transform at the given time
        // This would involve evaluating the translation, rotation, and scaling tracks
        // and combining them into a transformation matrix
        // Implementation omitted for brevity
        return Matrix4x4.Identity;
    }
}
``` 