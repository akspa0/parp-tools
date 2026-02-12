# MDLKEYTRACK - MDX Key Track Animation Structure

## Type
MDX Common Structure

## Source
MDX_index.md

## Description
The MDLKEYTRACK structure is a template-based system used for defining animation tracks in MDX. It provides a flexible mechanism for animating various properties (translation, rotation, scaling, color, etc.) over time. Each key track consists of a series of keyframes with timestamps and values, along with interpolation information.

## Structure

```csharp
public struct MDLKEYTRACK<T>
{
    /// <summary>
    /// Number of keyframes in the track
    /// </summary>
    public uint numKeys;
    
    /// <summary>
    /// Interpolation type for the keyframes
    /// </summary>
    public uint interpolationType;
    
    /// <summary>
    /// Global sequence ID (-1 if not part of a global sequence)
    /// </summary>
    public uint globalSequenceId;
    
    /// <summary>
    /// Array of keyframes
    /// </summary>
    // MDLKEYFRAME<T> keyframes[numKeys] follows
}

public struct MDLKEYFRAME<T>
{
    /// <summary>
    /// Time of the keyframe in milliseconds
    /// </summary>
    public uint time;
    
    /// <summary>
    /// Value at this keyframe (type varies)
    /// </summary>
    public T value;
    
    /// <summary>
    /// In-tangent for Hermite/Bezier interpolation (optional)
    /// </summary>
    // T inTan; (only for Hermite/Bezier)
    
    /// <summary>
    /// Out-tangent for Hermite/Bezier interpolation (optional)
    /// </summary>
    // T outTan; (only for Hermite/Bezier)
}
```

## Properties (MDLKEYTRACK)

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | numKeys | uint | Number of keyframes in the track |
| 0x04 | interpolationType | uint | Interpolation method (see Interpolation Types) |
| 0x08 | globalSequenceId | uint | Global sequence ID (0xFFFFFFFF = not part of a global sequence) |
| 0x0C | ... | ... | Array of MDLKEYFRAME structures follows |

## Properties (MDLKEYFRAME)

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | time | uint | Time of the keyframe in milliseconds |
| 0x04 | value | T | Value at this keyframe (type depends on the animated property) |
| varies | inTan | T | In-tangent (only present for interpolation types 2 and 3) |
| varies | outTan | T | Out-tangent (only present for interpolation type 3) |

## Interpolation Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | None | No interpolation, value changes instantly at keyframe time |
| 1 | Linear | Linear interpolation between keyframes |
| 2 | Hermite | Hermite spline interpolation, includes in/out tangents |
| 3 | Bezier | Bezier spline interpolation, includes in/out tangents |

## Common Template Types

| Type | Usage | Description |
|------|-------|-------------|
| Vector3 | Translation, Scaling | 3D vector for position or scale |
| Quaternion | Rotation | 4D quaternion for rotation |
| float | Alpha, Intensity | Scalar value for transparency or intensity |
| Vector3 | Color | RGB color value |
| int | Visibility | Integer flag for visibility (0 = hidden, 1 = visible) |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Added support for compressed quaternions in rotation tracks |

## Usage Context
The MDLKEYTRACK structure:
- Provides a consistent way to animate different property types
- Supports multiple interpolation methods for different animation needs
- Integrates with the sequence system for timeline-based animations
- Supports global sequences for continuous, timeline-independent animations

## Animation System
The MDX animation system features:
- Time-based keyframe animations
- Multiple animation sequences (SEQS chunk)
- Global sequences for continuous animations (GLBS chunk)
- Different interpolation types for various animation needs
- Complex hierarchical animations through object parenting

## Global Sequences
Global sequences provide a way to create continuous animations:
- If globalSequenceId is not 0xFFFFFFFF, the track uses a global sequence
- Global sequences are defined in the GLBS chunk
- Global sequence time is based on real time rather than sequence time
- Multiple objects can share the same global sequence

## Implementation Notes
- Keyframe times are in milliseconds
- For non-global sequence tracks, time is relative to sequence start
- For global sequence tracks, time is modulo the global sequence duration
- Tangent vectors for Hermite/Bezier interpolation follow the same memory layout as the main value
- Different property types may have specific optimizations or formats
- Rotation tracks in WoW versions often use compressed quaternions

## Implementation Example

```csharp
public class MdxKeyTrack<T>
{
    public uint NumKeys { get; private set; }
    public InterpolationType InterpolationType { get; private set; }
    public uint GlobalSequenceId { get; private set; }
    public List<MdxKeyFrame<T>> KeyFrames { get; private set; }
    
    public bool UsesGlobalSequence => GlobalSequenceId != 0xFFFFFFFF;
    
    public enum InterpolationType : uint
    {
        None = 0,
        Linear = 1,
        Hermite = 2,
        Bezier = 3
    }
    
    public void Parse(BinaryReader reader, Func<BinaryReader, T> valueReader)
    {
        NumKeys = reader.ReadUInt32();
        InterpolationType = (InterpolationType)reader.ReadUInt32();
        GlobalSequenceId = reader.ReadUInt32();
        
        KeyFrames = new List<MdxKeyFrame<T>>((int)NumKeys);
        
        for (int i = 0; i < NumKeys; i++)
        {
            var keyFrame = new MdxKeyFrame<T>();
            keyFrame.Time = reader.ReadUInt32();
            keyFrame.Value = valueReader(reader);
            
            // Read tangent vectors if needed
            if (InterpolationType >= InterpolationType.Hermite)
            {
                keyFrame.InTangent = valueReader(reader);
                
                if (InterpolationType == InterpolationType.Bezier)
                {
                    keyFrame.OutTangent = valueReader(reader);
                }
            }
            
            KeyFrames.Add(keyFrame);
        }
    }
    
    public void Write(BinaryWriter writer, Action<BinaryWriter, T> valueWriter)
    {
        writer.Write(NumKeys);
        writer.Write((uint)InterpolationType);
        writer.Write(GlobalSequenceId);
        
        foreach (var keyFrame in KeyFrames)
        {
            writer.Write(keyFrame.Time);
            valueWriter(writer, keyFrame.Value);
            
            // Write tangent vectors if needed
            if (InterpolationType >= InterpolationType.Hermite)
            {
                valueWriter(writer, keyFrame.InTangent);
                
                if (InterpolationType == InterpolationType.Bezier)
                {
                    valueWriter(writer, keyFrame.OutTangent);
                }
            }
        }
    }
    
    public T Evaluate(uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (KeyFrames.Count == 0)
        {
            return default(T);
        }
        
        // Handle global sequences
        if (UsesGlobalSequence && globalSequences.ContainsKey(GlobalSequenceId))
        {
            uint globalDuration = globalSequences[GlobalSequenceId];
            if (globalDuration > 0)
            {
                time = time % globalDuration;
            }
        }
        else if (sequenceDuration > 0)
        {
            // Loop animation within sequence
            time = time % sequenceDuration;
        }
        
        // Find keyframes that surround the requested time
        if (time <= KeyFrames[0].Time || KeyFrames.Count == 1)
        {
            return KeyFrames[0].Value;
        }
        
        if (time >= KeyFrames[KeyFrames.Count - 1].Time)
        {
            return KeyFrames[KeyFrames.Count - 1].Value;
        }
        
        // Find the surrounding keyframes
        int index = 0;
        while (index < KeyFrames.Count - 1 && KeyFrames[index + 1].Time <= time)
        {
            index++;
        }
        
        var frame1 = KeyFrames[index];
        var frame2 = KeyFrames[index + 1];
        
        // Calculate the interpolation factor
        float t = (float)(time - frame1.Time) / (frame2.Time - frame1.Time);
        
        // Interpolate based on the track type
        return InterpolateValue(frame1, frame2, t);
    }
    
    private T InterpolateValue(MdxKeyFrame<T> frame1, MdxKeyFrame<T> frame2, float t)
    {
        // Implementation depends on type T and interpolation type
        // This is a placeholder - actual implementation would depend on the type
        
        switch (InterpolationType)
        {
            case InterpolationType.None:
                return frame1.Value;
                
            case InterpolationType.Linear:
                // Linear interpolation implementation
                // Would need specific implementations for Vector3, Quaternion, float, etc.
                return default(T);
                
            case InterpolationType.Hermite:
            case InterpolationType.Bezier:
                // Spline interpolation implementation
                // Would need specific implementations for each type
                return default(T);
                
            default:
                return frame1.Value;
        }
    }
}

public class MdxKeyFrame<T>
{
    public uint Time { get; set; }
    public T Value { get; set; }
    public T InTangent { get; set; }
    public T OutTangent { get; set; }
}
``` 