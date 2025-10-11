# FAFX (Facial Effects)

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The FAFX chunk defines facial animations for character models, enabling expressions, lip-syncing, and other facial movements. This chunk contains a specialized animation system specifically for facial features, allowing for complex character emotions without requiring full skeletal animations. This system is particularly important for close-up character interactions, cinematics, and dialogue sequences.

## Structure
The FAFX chunk contains a collection of facial animation definitions, including facial keyframes, expression maps, and morph target data.

```csharp
public class FAFX
{
    public string Magic { get; set; } // "FAFX"
    public int Size { get; set; }     // Size of the chunk data in bytes
    public List<MDLFACIALFX> FacialEffects { get; set; }
}

// Facial effect structure
public class MDLFACIALFX : MDLGENOBJECT
{
    public uint InclusiveSize { get; set; }       // Size including this header
    public string Name { get; set; }              // Name of the facial effect
    public uint TargetGeosetID { get; set; }      // Geoset that this effect applies to
    public List<MDLFACIALKEYTRANS> KeyTransforms { get; set; } // Key transformations
    
    // Animation tracks (optional, presence determined by flags in MDLGENOBJECT)
    public List<MDLMORPHTRACK> MorphTracks { get; set; } // Morph animation tracks
}

// Facial key transformation
public struct MDLFACIALKEYTRANS
{
    public ushort VertexIndex { get; set; }      // Index of target vertex
    public Vector3 Translation { get; set; }     // Displacement vector for morphing
}

// Morph track for animating facial features
public class MDLMORPHTRACK
{
    public string MorphName { get; set; }         // Name of morph (e.g., "smile", "frown")
    public MDLKEYTRACK MorphKeyframes { get; set; } // Animation keyframes
}
```

## Properties

### FAFX Chunk
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Magic | char[4] | "FAFX" |
| 0x04 | Size | uint32 | Size of the chunk data in bytes |
| 0x08 | FacialEffects | MDLFACIALFX[] | Array of facial effect definitions |

### MDLFACIALFX Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | InclusiveSize | uint32 | Total size of this facial effect definition |
| 0x04 | ObjectId | uint32 | Unique ID for this object |
| 0x08 | ParentId | uint32 | Parent object's ID (0 for no parent) |
| 0x0C | Flags | uint32 | Animation flags |
| 0x10 | NameLength | uint32 | Length of name string |
| 0x14 | Name | char[] | Name of facial effect |
| var | TargetGeosetID | uint32 | ID of target geoset |
| var | KeyTransformCount | uint32 | Number of key transformations |
| var | KeyTransforms | MDLFACIALKEYTRANS[] | Array of key transformations |
| var | MorphTracks | MDLMORPHTRACK[] | Array of morph animation tracks |

### MDLFACIALKEYTRANS Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | VertexIndex | uint16 | Index of vertex in target geoset |
| 0x02 | TranslationX | float | X displacement |
| 0x06 | TranslationY | float | Y displacement |
| 0x0A | TranslationZ | float | Z displacement |

## Animation Flags

The flags field in the MDLGENOBJECT parent structure determines various facial animation properties:

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x00000001 | MORPH | Has morph animation tracks |
| 0x00000010 | NOTLOOPED | Animation should not loop |
| 0x00001000 | ABSOLUTE | Use absolute transforms rather than relative |

## Common Facial Expressions

The FAFX chunk often defines standard facial expressions:

1. **Basic Emotions**: Smile, frown, anger, surprise, fear, disgust
2. **Mouth Shapes**: For phonemes - A, E, I, O, U, consonant positions
3. **Eye Movements**: Blink, squint, widen
4. **Brow Movements**: Raise brows, furrow brows
5. **Combined Expressions**: Laugh, cry, shout, whisper

## Version Differences

| Version | Differences |
|---------|-------------|
| Early MDX | Basic facial controls with limited vertex transformations |
| Later MDX | More sophisticated expression blending and phoneme mapping |
| Some Models | May not include FAFX chunk if facial animation is not needed |

## Dependencies
- **GEOS**: Defines the geosets that facial effects are applied to
- **SEQS**: Facial animations are often synchronized with animation sequences
- **BONE**: Some facial effects may be tied to bone animations for jaw/head movement

## Implementation Notes

1. **Morph Targets**: The facial system uses morph targets (blend shapes) where vertices are displaced to create expressions
2. **Expression Blending**: Multiple expressions can be blended for complex facial animations
3. **Vertex Selection**: Only specific vertices (those defining facial features) are affected
4. **Animation Timing**:
   - Facial animations are typically tied to dialogue and cinematic timing
   - Can be synchronized with audio for lip-syncing
5. **Weighting**: Vertices may be moved partially toward a target position using weights
6. **Real-time Adjustment**: Some implementations allow dynamic adjustment for runtime expression control

## Usage Context

Facial animations are essential for:

1. **Character Dialogue**: Lip synchronization with spoken dialogue
2. **Emotional Expression**: Conveying character emotions in cinematics
3. **Reaction Shots**: Character reactions to events in the game world
4. **Ambient Animation**: Adding life to idle characters or NPCs
5. **Player Feedback**: Visual responses to player actions or choices

## Implementation Example

```csharp
public class FacialAnimationSystem
{
    private List<FacialEffect> facialEffects = new List<FacialEffect>();
    
    public class FacialEffect
    {
        public string Name { get; set; }
        public uint TargetGeosetID { get; set; }
        public List<FacialKeyTransform> KeyTransforms { get; set; } = new List<FacialKeyTransform>();
        public List<MorphTrack> MorphTracks { get; set; } = new List<MorphTrack>();
    }
    
    public struct FacialKeyTransform
    {
        public ushort VertexIndex { get; set; }
        public Vector3 Translation { get; set; }
    }
    
    public class MorphTrack
    {
        public string Name { get; set; }
        public KeyTrack<float> Keyframes { get; set; }
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
    
    public void ParseFAFX(BinaryReader reader)
    {
        // Read chunk header
        string magic = new string(reader.ReadChars(4));
        if (magic != "FAFX")
            throw new Exception("Invalid FAFX chunk header");
            
        int chunkSize = reader.ReadInt32();
        long startPosition = reader.BaseStream.Position;
        
        facialEffects.Clear();
        
        // Read facial effects until we've consumed the entire chunk
        while (reader.BaseStream.Position < startPosition + chunkSize)
        {
            uint inclusiveSize = reader.ReadUInt32();
            long effectStartPosition = reader.BaseStream.Position - 4; // Include size field
            
            FacialEffect effect = new FacialEffect();
            
            // Read object data (MDLGENOBJECT)
            uint objectId = reader.ReadUInt32();
            uint parentId = reader.ReadUInt32();
            uint flags = reader.ReadUInt32();
            
            // Read facial effect name
            uint nameLength = reader.ReadUInt32();
            effect.Name = new string(reader.ReadChars((int)nameLength)).TrimEnd('\0');
            
            // Read target geoset ID
            effect.TargetGeosetID = reader.ReadUInt32();
            
            // Read key transformations
            uint keyTransformCount = reader.ReadUInt32();
            for (int i = 0; i < keyTransformCount; i++)
            {
                ushort vertexIndex = reader.ReadUInt16();
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                
                effect.KeyTransforms.Add(new FacialKeyTransform
                {
                    VertexIndex = vertexIndex,
                    Translation = new Vector3(x, y, z)
                });
            }
            
            // Read morph tracks if present
            if ((flags & 0x00000001) != 0) // Has morph tracks
            {
                uint morphTrackCount = reader.ReadUInt32();
                for (int i = 0; i < morphTrackCount; i++)
                {
                    // Read morph name
                    uint morphNameLength = reader.ReadUInt32();
                    string morphName = new string(reader.ReadChars((int)morphNameLength)).TrimEnd('\0');
                    
                    // Read morph keyframe track
                    KeyTrack<float> keyTrack = ReadKeyTrack(reader);
                    
                    effect.MorphTracks.Add(new MorphTrack
                    {
                        Name = morphName,
                        Keyframes = keyTrack
                    });
                }
            }
            
            facialEffects.Add(effect);
            
            // Skip to the end of this facial effect definition
            reader.BaseStream.Position = effectStartPosition + inclusiveSize;
        }
        
        // Verify we read the expected amount of data
        long endPosition = reader.BaseStream.Position;
        if (endPosition - startPosition != chunkSize)
            throw new Exception($"FAFX chunk size mismatch. Expected {chunkSize}, read {endPosition - startPosition}");
    }
    
    private KeyTrack<float> ReadKeyTrack(BinaryReader reader)
    {
        KeyTrack<float> track = new KeyTrack<float>();
        
        // Read track header
        uint trackType = reader.ReadUInt32(); // Linear = 1, Hermite = 2, Bezier = 3
        uint globalSequenceID = reader.ReadUInt32();
        uint keyFrameCount = reader.ReadUInt32();
        
        track.GlobalSequenceID = globalSequenceID != 0xFFFFFFFF ? (int)globalSequenceID : -1;
        
        // Read keyframes
        for (int i = 0; i < keyFrameCount; i++)
        {
            int time = reader.ReadInt32();
            float value = reader.ReadSingle();
            
            // For Hermite and Bezier curves, read additional control points
            if (trackType >= 2) // Hermite or Bezier
            {
                // Skip inTan and outTan values
                reader.ReadSingle(); // inTan
                reader.ReadSingle(); // outTan
            }
            
            track.KeyFrames.Add(new KeyFrame<float> { Time = time, Value = value });
        }
        
        return track;
    }
    
    public void WriteFAFX(BinaryWriter writer)
    {
        if (facialEffects.Count == 0)
            return; // Skip if no facial effects
        
        // Calculate chunk size first
        int chunkSize = 0;
        foreach (var effect in facialEffects)
        {
            // Start with base size (inclusive size + object ID, parent ID, flags)
            int effectSize = 4 + 12;
            
            // Add name size
            effectSize += 4 + effect.Name.Length + 1; // +1 for null terminator
            
            // Add target geoset ID
            effectSize += 4;
            
            // Add key transforms
            effectSize += 4 + (effect.KeyTransforms.Count * 14); // 2 bytes for index + 12 bytes for vector
            
            // Add morph tracks if present
            if (effect.MorphTracks.Count > 0)
            {
                effectSize += 4; // Morph track count
                
                foreach (var track in effect.MorphTracks)
                {
                    // Morph name
                    effectSize += 4 + track.Name.Length + 1;
                    
                    // Keyframe track
                    effectSize += 12; // Track header
                    effectSize += track.Keyframes.KeyFrames.Count * 8; // Time + value
                }
            }
            
            chunkSize += effectSize;
        }
        
        // Write chunk header
        writer.Write("FAFX".ToCharArray());
        writer.Write(chunkSize);
        
        // Write each facial effect
        foreach (var effect in facialEffects)
        {
            // Calculate effect size
            int effectSize = 4 + 12; // inclusive size + object ID, parent ID, flags
            effectSize += 4 + effect.Name.Length + 1;
            effectSize += 4; // target geoset ID
            effectSize += 4 + (effect.KeyTransforms.Count * 14);
            
            if (effect.MorphTracks.Count > 0)
            {
                effectSize += 4;
                
                foreach (var track in effect.MorphTracks)
                {
                    effectSize += 4 + track.Name.Length + 1;
                    effectSize += 12;
                    effectSize += track.Keyframes.KeyFrames.Count * 8;
                }
            }
            
            // Write effect header
            writer.Write(effectSize);
            writer.Write(0U); // Object ID (typically auto-generated)
            writer.Write(0U); // Parent ID (typically 0)
            
            // Set appropriate flags
            uint flags = 0;
            if (effect.MorphTracks.Count > 0)
                flags |= 0x00000001; // Has morph tracks
                
            writer.Write(flags);
            
            // Write name
            writer.Write(effect.Name.Length + 1); // +1 for null terminator
            writer.Write(effect.Name.ToCharArray());
            writer.Write((byte)0); // Null terminator
            
            // Write target geoset ID
            writer.Write(effect.TargetGeosetID);
            
            // Write key transforms
            writer.Write((uint)effect.KeyTransforms.Count);
            foreach (var transform in effect.KeyTransforms)
            {
                writer.Write(transform.VertexIndex);
                writer.Write(transform.Translation.X);
                writer.Write(transform.Translation.Y);
                writer.Write(transform.Translation.Z);
            }
            
            // Write morph tracks if present
            if (effect.MorphTracks.Count > 0)
            {
                writer.Write((uint)effect.MorphTracks.Count);
                
                foreach (var track in effect.MorphTracks)
                {
                    // Write morph name
                    writer.Write((uint)(track.Name.Length + 1));
                    writer.Write(track.Name.ToCharArray());
                    writer.Write((byte)0); // Null terminator
                    
                    // Write keyframe track
                    writer.Write(1U); // Track type (1 = Linear)
                    writer.Write(track.Keyframes.GlobalSequenceID >= 0 ? 
                        (uint)track.Keyframes.GlobalSequenceID : 0xFFFFFFFF);
                    writer.Write((uint)track.Keyframes.KeyFrames.Count);
                    
                    foreach (var keyFrame in track.Keyframes.KeyFrames)
                    {
                        writer.Write(keyFrame.Time);
                        writer.Write(keyFrame.Value);
                    }
                }
            }
        }
    }
    
    // Apply facial animations to vertex data
    public void ApplyFacialAnimation(string effectName, List<Vector3> vertices, uint geosetID, int time, List<int> globalSequenceTimes)
    {
        // Find facial effect by name
        FacialEffect effect = facialEffects.FirstOrDefault(e => 
            e.Name.Equals(effectName, StringComparison.OrdinalIgnoreCase) && 
            e.TargetGeosetID == geosetID);
            
        if (effect == null)
            return; // Effect not found
            
        // Apply key transformations
        foreach (var transform in effect.KeyTransforms)
        {
            if (transform.VertexIndex < vertices.Count)
            {
                // Calculate morph weight from animation
                float weight = 1.0f;
                
                // If we have a morph track for this effect, apply weight
                foreach (var morphTrack in effect.MorphTracks)
                {
                    if (morphTrack.Name.Equals("weight", StringComparison.OrdinalIgnoreCase))
                    {
                        weight = GetKeyFrameValueAtTime(morphTrack.Keyframes, time, globalSequenceTimes);
                        break;
                    }
                }
                
                // Apply weighted transformation
                vertices[transform.VertexIndex] += transform.Translation * weight;
            }
        }
    }
    
    // Blend multiple facial expressions
    public void BlendFacialExpressions(List<FacialBlend> blends, List<Vector3> vertices, uint geosetID, int time, List<int> globalSequenceTimes)
    {
        // Create a copy of original vertices to work from
        List<Vector3> originalVertices = new List<Vector3>(vertices);
        
        // Apply each facial blend
        foreach (var blend in blends)
        {
            // Find facial effect
            FacialEffect effect = facialEffects.FirstOrDefault(e => 
                e.Name.Equals(blend.EffectName, StringComparison.OrdinalIgnoreCase) && 
                e.TargetGeosetID == geosetID);
                
            if (effect == null)
                continue;
                
            // Apply key transformations with blend weight
            foreach (var transform in effect.KeyTransforms)
            {
                if (transform.VertexIndex < vertices.Count)
                {
                    // Calculate morph weight from animation if available
                    float animWeight = 1.0f;
                    
                    foreach (var morphTrack in effect.MorphTracks)
                    {
                        if (morphTrack.Name.Equals("weight", StringComparison.OrdinalIgnoreCase))
                        {
                            animWeight = GetKeyFrameValueAtTime(morphTrack.Keyframes, time, globalSequenceTimes);
                            break;
                        }
                    }
                    
                    // Apply weighted transformation (animation weight * blend weight)
                    vertices[transform.VertexIndex] += transform.Translation * animWeight * blend.Weight;
                }
            }
        }
    }
    
    // Get interpolated value from a key track at specified time
    private float GetKeyFrameValueAtTime(KeyTrack<float> track, int time, List<int> globalSequenceTimes)
    {
        if (track.KeyFrames.Count == 0)
            return 0.0f;
            
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
        KeyFrame<float> prevFrame = track.KeyFrames[0];
        KeyFrame<float> nextFrame = track.KeyFrames[track.KeyFrames.Count - 1];
        
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
        return prevFrame.Value + (nextFrame.Value - prevFrame.Value) * factor;
    }
    
    // Helper class for blending multiple facial expressions
    public class FacialBlend
    {
        public string EffectName { get; set; }
        public float Weight { get; set; }
    }
}
``` 