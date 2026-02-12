# GLBS - MDX Global Sequences Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The GLBS (Global Sequences) chunk defines timeline-independent animation cycles. Unlike normal sequences that operate within specific time ranges, global sequences run continuously based on the real-time clock. They are used for animations that should continue regardless of the currently playing sequence, such as continuous particle effects, ambient animations, or cyclical movements.

## Structure

```csharp
public struct GLBS
{
    /// <summary>
    /// Array of global sequence durations
    /// </summary>
    // uint durations[numGlobalSequences] follows
}
```

## Properties

### GLBS Chunk
The GLBS chunk contains an array of unsigned integers, each representing the duration of a global sequence in milliseconds. The number of global sequences is determined by the chunk size divided by 4 (the size of a uint).

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | durations | uint[] | Array of global sequence durations in milliseconds |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure and behavior |

## Dependencies
- Animation tracks in other chunks may reference global sequences via globalSequenceId field

## Implementation Notes
- Global sequences are referenced by index (0-based)
- A duration of 0 indicates an animation that doesn't advance (static)
- Global sequence time is typically calculated as: (currentTime % duration)
- The chunk may not exist if the model doesn't use any global sequences
- Multiple animation tracks can reference the same global sequence for synchronized animation

## Usage Context
Global sequences are used for:
- Continuous animations that should play regardless of the character's current action
- Ambient effects like glowing, pulsing, or rotation
- Clock-based animations (e.g., rotating wheels, swinging pendulums)
- Effects that should maintain their state across sequence changes

## Animation System Integration
- Animation tracks reference global sequences via the globalSequenceId field in MDLKEYTRACK
- When a track uses a global sequence, its time is derived from the global timer rather than the sequence time
- Normal sequence-based tracks and global sequence tracks can coexist in the same model
- When transitioning between sequences, global sequence animations continue uninterrupted

## Implementation Example

```csharp
public class GLBSChunk : IMdxChunk
{
    public string ChunkId => "GLBS";
    
    public List<uint> GlobalSequenceDurations { get; private set; } = new List<uint>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate number of global sequences
        int numGlobalSequences = (int)(size / 4); // Each duration is 4 bytes (uint)
        
        // Clear any existing data
        GlobalSequenceDurations.Clear();
        
        // Read all global sequence durations
        for (int i = 0; i < numGlobalSequences; i++)
        {
            uint duration = reader.ReadUInt32();
            GlobalSequenceDurations.Add(duration);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write all global sequence durations
        foreach (uint duration in GlobalSequenceDurations)
        {
            writer.Write(duration);
        }
    }
    
    /// <summary>
    /// Gets the current time for a global sequence based on the system time
    /// </summary>
    /// <param name="globalSequenceIndex">Index of the global sequence</param>
    /// <param name="currentTime">Current system time in milliseconds</param>
    /// <returns>The current time position within the global sequence</returns>
    public uint GetGlobalSequenceTime(int globalSequenceIndex, uint currentTime)
    {
        if (globalSequenceIndex < 0 || globalSequenceIndex >= GlobalSequenceDurations.Count)
        {
            return 0;
        }
        
        uint duration = GlobalSequenceDurations[globalSequenceIndex];
        if (duration == 0)
        {
            return 0; // Static animation
        }
        
        return currentTime % duration;
    }
    
    /// <summary>
    /// Gets a dictionary mapping global sequence indices to their durations
    /// </summary>
    /// <returns>Dictionary with global sequence index as key and duration as value</returns>
    public Dictionary<uint, uint> GetGlobalSequenceMap()
    {
        var result = new Dictionary<uint, uint>();
        for (uint i = 0; i < GlobalSequenceDurations.Count; i++)
        {
            result[i] = GlobalSequenceDurations[(int)i];
        }
        return result;
    }
}
``` 