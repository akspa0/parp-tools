# TXAN (Texture Animation)

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The TXAN chunk defines animations for material textures, allowing for dynamic effects such as scrolling, rotation, and scaling of texture coordinates. These animations are often used to create effects like flowing water, pulsing energy, spinning radar displays, and animated UI elements. Texture animations operate independently of mesh animations, making them a powerful tool for adding visual detail to static or animated models.

## Structure
The TXAN chunk contains a collection of texture animation definitions, each linked to specific material layers defined in the MTLS chunk.

```csharp
public class TXAN
{
    public string Magic { get; set; } // "TXAN"
    public int Size { get; set; }     // Size of the chunk data in bytes
    public List<MDLTEXTUREANM> TextureAnimations { get; set; }
}

// Texture animation structure
public class MDLTEXTUREANM : MDLGENOBJECT
{
    public uint InclusiveSize { get; set; }    // Size including this header
    public uint MaterialID { get; set; }       // Index of material in MTLS chunk
    public uint LayerID { get; set; }          // Layer ID within material (0-based)
    
    // Animation tracks (optional, presence determined by flags in MDLGENOBJECT)
    public MDLKEYTRACK TranslationTrack { get; set; }  // Translation keyframes (UV offset)
    public MDLKEYTRACK RotationTrack { get; set; }     // Rotation keyframes (UV rotation)
    public MDLKEYTRACK ScalingTrack { get; set; }      // Scaling keyframes (UV scale)
}
```

## Properties

### TXAN Chunk
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Magic | char[4] | "TXAN" |
| 0x04 | Size | uint32 | Size of the chunk data in bytes |
| 0x08 | TextureAnimations | MDLTEXTUREANM[] | Array of texture animation definitions |

### MDLTEXTUREANM Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | InclusiveSize | uint32 | Total size of this texture animation definition |
| 0x04 | ObjectId | uint32 | Unique ID for this object |
| 0x08 | ParentId | uint32 | Parent object's ID (0 for no parent) |
| 0x0C | Flags | uint32 | Animation flags (determines which tracks exist) |
| 0x10 | MaterialID | uint32 | Index of material in the MTLS chunk |
| 0x14 | LayerID | uint32 | Layer index within material (0-based) |
| var | TranslationTrack | MDLKEYTRACK | Translation animation track (if flag set) |
| var | RotationTrack | MDLKEYTRACK | Rotation animation track (if flag set) |
| var | ScalingTrack | MDLKEYTRACK | Scaling animation track (if flag set) |

## Animation Flags

The flags field in the MDLGENOBJECT parent structure determines which animation tracks are present:

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x00000001 | TRANSLATION | Has translation (UV offset) track |
| 0x00000002 | ROTATION | Has rotation track |
| 0x00000004 | SCALING | Has scaling track |
| 0x00000010 | NOTLOOPED | Animation should not loop |

## Animation Tracks

Each animation track contains keyframes that define how the texture transformation changes over time:

1. **Translation Track**: Defines UV coordinate offsets (Vector3, but only X and Y used)
2. **Rotation Track**: Defines rotation angle in radians around UV center
3. **Scaling Track**: Defines U and V scale factors (Vector3, but only X and Y used)

## Version Differences

| Version | Differences |
|---------|-------------|
| All | The TXAN chunk has maintained a consistent format across MDX versions |
| Early Models | May use simpler animations with only translation |
| Advanced Models | May combine multiple transformation types for complex effects |

## Dependencies
- **MTLS**: Defines the materials and layers that texture animations are applied to
- **TEXS**: Defines the textures being animated
- **SEQS**: Animations may be linked to specific animation sequences
- **GLBS**: May use global sequences for continuous animation

## Implementation Notes

1. **Transformation Order**: When multiple transforms are applied, the order is typically:
   - Scale
   - Rotate
   - Translate
2. **Pivot Point**: Rotations are performed around UV coordinate (0.5, 0.5) by default
3. **Animation Timing**:
   - Texture animations can be tied to specific sequence timelines or use global sequences
   - They can loop independently of the model's animation
4. **Interpolation**: Linear interpolation is typically used between keyframes
5. **Layer Specificity**: Animations target specific layers within materials, allowing for multi-layered effects
6. **UV Wrapping**: When UVs are animated beyond the 0-1 range, they typically wrap around (modulo 1.0)

## Usage Context

Texture animations are commonly used for:

1. **Environmental Effects**: Flowing water, lava, clouds, fog
2. **Energy Effects**: Pulsing magic, electricity, force fields
3. **Mechanical Elements**: Spinning gears, radar screens, computer displays
4. **Status Effects**: Glowing auras, elemental overlays
5. **UI Elements**: Progress indicators, highlighting effects
6. **Creature Features**: Blinking eyes, color shifting skin

## Implementation Example

```csharp
public class TextureAnimationSystem
{
    private List<TextureAnimation> textureAnimations = new List<TextureAnimation>();
    
    public class TextureAnimation
    {
        public uint MaterialID { get; set; }
        public uint LayerID { get; set; }
        public uint Flags { get; set; }
        
        // Animation tracks
        public KeyTrack<Vector3> TranslationTrack { get; set; }
        public KeyTrack<float> RotationTrack { get; set; }
        public KeyTrack<Vector3> ScalingTrack { get; set; }
    }
    
    public class KeyTrack<T>
    {
        public List<KeyFrame<T>> KeyFrames { get; set; } = new List<KeyFrame<T>>();
        public int GlobalSequenceID { get; set; } = -1;
    }
    
    public class KeyFrame<T>
    {
        public int Time { get; set; }
        public T Value { get; set; }
    }
    
    public void ParseTXAN(BinaryReader reader)
    {
        // Read chunk header
        string magic = new string(reader.ReadChars(4));
        if (magic != "TXAN")
            throw new Exception("Invalid TXAN chunk header");
            
        int chunkSize = reader.ReadInt32();
        long startPosition = reader.BaseStream.Position;
        
        textureAnimations.Clear();
        
        // Read texture animations until we've consumed the entire chunk
        while (reader.BaseStream.Position < startPosition + chunkSize)
        {
            uint inclusiveSize = reader.ReadUInt32();
            long animStartPosition = reader.BaseStream.Position - 4; // Include size field
            
            TextureAnimation anim = new TextureAnimation();
            
            // Read object data (MDLGENOBJECT)
            uint objectId = reader.ReadUInt32();
            uint parentId = reader.ReadUInt32();
            uint flags = reader.ReadUInt32();
            
            anim.Flags = flags;
            
            // Read texture animation specific data
            anim.MaterialID = reader.ReadUInt32();
            anim.LayerID = reader.ReadUInt32();
            
            // Read animation tracks if present
            if ((flags & 0x00000001) != 0) // Has translation
            {
                anim.TranslationTrack = ReadKeyTrack<Vector3>(reader, ReadVector3);
            }
            
            if ((flags & 0x00000002) != 0) // Has rotation
            {
                anim.RotationTrack = ReadKeyTrack<float>(reader, reader => reader.ReadSingle());
            }
            
            if ((flags & 0x00000004) != 0) // Has scaling
            {
                anim.ScalingTrack = ReadKeyTrack<Vector3>(reader, ReadVector3);
            }
            
            textureAnimations.Add(anim);
            
            // Skip to the end of this animation definition
            reader.BaseStream.Position = animStartPosition + inclusiveSize;
        }
        
        // Verify we read the expected amount of data
        long endPosition = reader.BaseStream.Position;
        if (endPosition - startPosition != chunkSize)
            throw new Exception($"TXAN chunk size mismatch. Expected {chunkSize}, read {endPosition - startPosition}");
    }
    
    private KeyTrack<T> ReadKeyTrack<T>(BinaryReader reader, Func<BinaryReader, T> valueReader)
    {
        KeyTrack<T> track = new KeyTrack<T>();
        
        // Read track header
        uint trackType = reader.ReadUInt32(); // Linear = 1, Hermite = 2, Bezier = 3
        uint globalSequenceID = reader.ReadUInt32();
        uint keyFrameCount = reader.ReadUInt32();
        
        track.GlobalSequenceID = globalSequenceID != 0xFFFFFFFF ? (int)globalSequenceID : -1;
        
        // Read keyframes
        for (int i = 0; i < keyFrameCount; i++)
        {
            int time = reader.ReadInt32();
            T value = valueReader(reader);
            
            // For Hermite and Bezier curves, read additional control points
            if (trackType >= 2) // Hermite or Bezier
            {
                // Skip inTan and outTan values
                for (int j = 0; j < 2; j++)
                {
                    valueReader(reader); // Skip tangent values
                }
            }
            
            track.KeyFrames.Add(new KeyFrame<T> { Time = time, Value = value });
        }
        
        return track;
    }
    
    private Vector3 ReadVector3(BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        return new Vector3(x, y, z);
    }
    
    public void WriteTXAN(BinaryWriter writer)
    {
        if (textureAnimations.Count == 0)
            return; // Skip if no texture animations
        
        // Calculate chunk size first
        int chunkSize = 0;
        foreach (var anim in textureAnimations)
        {
            // Start with base size (inclusive size + object ID, parent ID, flags + material ID + layer ID)
            int animSize = 4 + 12 + 8;
            
            // Add sizes for tracks
            if ((anim.Flags & 0x00000001) != 0 && anim.TranslationTrack != null)
                animSize += CalculateTrackSize<Vector3>(anim.TranslationTrack);
                
            if ((anim.Flags & 0x00000002) != 0 && anim.RotationTrack != null)
                animSize += CalculateTrackSize<float>(anim.RotationTrack);
                
            if ((anim.Flags & 0x00000004) != 0 && anim.ScalingTrack != null)
                animSize += CalculateTrackSize<Vector3>(anim.ScalingTrack);
                
            chunkSize += animSize;
        }
        
        // Write chunk header
        writer.Write("TXAN".ToCharArray());
        writer.Write(chunkSize);
        
        // Write each texture animation
        foreach (var anim in textureAnimations)
        {
            // Calculate size for this animation
            int animSize = 4 + 12 + 8; // Base size
            
            if ((anim.Flags & 0x00000001) != 0 && anim.TranslationTrack != null)
                animSize += CalculateTrackSize<Vector3>(anim.TranslationTrack);
                
            if ((anim.Flags & 0x00000002) != 0 && anim.RotationTrack != null)
                animSize += CalculateTrackSize<float>(anim.RotationTrack);
                
            if ((anim.Flags & 0x00000004) != 0 && anim.ScalingTrack != null)
                animSize += CalculateTrackSize<Vector3>(anim.ScalingTrack);
            
            // Write animation header
            writer.Write(animSize);
            writer.Write(0U); // Object ID (typically 0)
            writer.Write(0U); // Parent ID (typically 0)
            writer.Write(anim.Flags);
            
            // Write texture animation specific data
            writer.Write(anim.MaterialID);
            writer.Write(anim.LayerID);
            
            // Write tracks if present
            if ((anim.Flags & 0x00000001) != 0 && anim.TranslationTrack != null)
                WriteKeyTrack(writer, anim.TranslationTrack, WriteVector3);
                
            if ((anim.Flags & 0x00000002) != 0 && anim.RotationTrack != null)
                WriteKeyTrack(writer, anim.RotationTrack, (w, v) => w.Write(v));
                
            if ((anim.Flags & 0x00000004) != 0 && anim.ScalingTrack != null)
                WriteKeyTrack(writer, anim.ScalingTrack, WriteVector3);
        }
    }
    
    private int CalculateTrackSize<T>(KeyTrack<T> track)
    {
        // Track header (type, global sequence ID, keyframe count)
        int size = 12;
        
        // Each keyframe
        int keyframeSize = sizeof(int); // Time
        
        if (typeof(T) == typeof(float))
            keyframeSize += sizeof(float);
        else if (typeof(T) == typeof(Vector3))
            keyframeSize += sizeof(float) * 3;
            
        size += track.KeyFrames.Count * keyframeSize;
        
        return size;
    }
    
    private void WriteKeyTrack<T>(BinaryWriter writer, KeyTrack<T> track, Action<BinaryWriter, T> valueWriter)
    {
        // Write track header
        writer.Write(1U); // Track type (1 = Linear)
        writer.Write(track.GlobalSequenceID >= 0 ? (uint)track.GlobalSequenceID : 0xFFFFFFFF);
        writer.Write((uint)track.KeyFrames.Count);
        
        // Write keyframes
        foreach (var keyFrame in track.KeyFrames)
        {
            writer.Write(keyFrame.Time);
            valueWriter(writer, keyFrame.Value);
        }
    }
    
    private void WriteVector3(BinaryWriter writer, Vector3 vector)
    {
        writer.Write(vector.X);
        writer.Write(vector.Y);
        writer.Write(vector.Z);
    }
    
    // Get texture transformation matrix at a specific time
    public Matrix4x4 GetTextureMatrix(uint materialID, uint layerID, int time, List<int> globalSequenceTimes)
    {
        // Find the matching texture animation
        TextureAnimation anim = textureAnimations.FirstOrDefault(a => 
            a.MaterialID == materialID && a.LayerID == layerID);
            
        if (anim == null)
            return Matrix4x4.Identity; // No animation, return identity matrix
            
        // Calculate transforms
        Vector3 translation = Vector3.Zero;
        float rotation = 0.0f;
        Vector3 scaling = new Vector3(1.0f, 1.0f, 1.0f);
        
        // Get translation
        if ((anim.Flags & 0x00000001) != 0 && anim.TranslationTrack != null)
        {
            translation = GetKeyFrameValueAtTime(anim.TranslationTrack, time, globalSequenceTimes);
        }
        
        // Get rotation
        if ((anim.Flags & 0x00000002) != 0 && anim.RotationTrack != null)
        {
            rotation = GetKeyFrameValueAtTime(anim.RotationTrack, time, globalSequenceTimes);
        }
        
        // Get scaling
        if ((anim.Flags & 0x00000004) != 0 && anim.ScalingTrack != null)
        {
            scaling = GetKeyFrameValueAtTime(anim.ScalingTrack, time, globalSequenceTimes);
        }
        
        // Construct transformation matrix (applied in order: scale, rotate, translate)
        // 1. Create center translation (move pivot to 0.5, 0.5)
        Matrix4x4 centerTranslation = Matrix4x4.CreateTranslation(-0.5f, -0.5f, 0.0f);
        Matrix4x4 uncenterTranslation = Matrix4x4.CreateTranslation(0.5f, 0.5f, 0.0f);
        
        // 2. Create scale matrix
        Matrix4x4 scaleMatrix = Matrix4x4.CreateScale(scaling.X, scaling.Y, 1.0f);
        
        // 3. Create rotation matrix
        Matrix4x4 rotationMatrix = Matrix4x4.CreateRotationZ(rotation);
        
        // 4. Create translation matrix
        Matrix4x4 translationMatrix = Matrix4x4.CreateTranslation(translation.X, translation.Y, 0.0f);
        
        // 5. Combine transforms: center -> scale -> rotate -> uncenter -> translate
        return centerTranslation * 
               scaleMatrix * 
               rotationMatrix * 
               uncenterTranslation * 
               translationMatrix;
    }
    
    // Get interpolated value from a key track at specified time
    private T GetKeyFrameValueAtTime<T>(KeyTrack<T> track, int time, List<int> globalSequenceTimes)
    {
        if (track.KeyFrames.Count == 0)
            return default(T);
            
        if (track.KeyFrames.Count == 1)
            return track.KeyFrames[0].Value;
            
        // Handle global sequence if specified
        if (track.GlobalSequenceID >= 0 && globalSequenceTimes != null && 
            track.GlobalSequenceID < globalSequenceTimes.Count)
        {
            int globalTime = globalSequenceTimes[track.GlobalSequenceID];
            if (globalTime > 0)
            {
                time = time % globalTime; // Loop global sequence
            }
        }
        
        // Find keyframes that bracket the requested time
        KeyFrame<T> prevFrame = track.KeyFrames[0];
        KeyFrame<T> nextFrame = track.KeyFrames[track.KeyFrames.Count - 1];
        
        for (int i = 0; i < track.KeyFrames.Count - 1; i++)
        {
            if (track.KeyFrames[i].Time <= time && track.KeyFrames[i + 1].Time > time)
            {
                prevFrame = track.KeyFrames[i];
                nextFrame = track.KeyFrames[i + 1];
                break;
            }
        }
        
        // If time is before first keyframe or after last keyframe
        if (time <= prevFrame.Time)
            return prevFrame.Value;
            
        if (time >= nextFrame.Time)
            return nextFrame.Value;
            
        // Interpolate between keyframes
        float factor = (float)(time - prevFrame.Time) / (nextFrame.Time - prevFrame.Time);
        
        if (typeof(T) == typeof(float))
        {
            float prevValue = (float)(object)prevFrame.Value;
            float nextValue = (float)(object)nextFrame.Value;
            return (T)(object)(prevValue + (nextValue - prevValue) * factor);
        }
        else if (typeof(T) == typeof(Vector3))
        {
            Vector3 prevValue = (Vector3)(object)prevFrame.Value;
            Vector3 nextValue = (Vector3)(object)nextFrame.Value;
            
            return (T)(object)new Vector3(
                prevValue.X + (nextValue.X - prevValue.X) * factor,
                prevValue.Y + (nextValue.Y - prevValue.Y) * factor,
                prevValue.Z + (nextValue.Z - prevValue.Z) * factor
            );
        }
        
        return default(T);
    }
}
``` 