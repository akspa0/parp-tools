# SEQS - MDX Sequences Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The SEQS (Sequences) chunk defines all the animation sequences in an MDX model. Each sequence represents a specific animation (like walking, attacking, or casting) with a defined time range and playback flags. Sequences provide the timing context for keyframe animations throughout the model.

## Structure

```csharp
public struct SEQS
{
    /// <summary>
    /// Array of animation sequences
    /// </summary>
    // MDLSEQUENCE sequences[numSequences] follows
}

public struct MDLSEQUENCE
{
    /// <summary>
    /// Sequence name (null-terminated string, max length 80)
    /// </summary>
    public fixed byte name[80];
    
    /// <summary>
    /// Start time of sequence in milliseconds
    /// </summary>
    public uint startTime;
    
    /// <summary>
    /// End time of sequence in milliseconds
    /// </summary>
    public uint endTime;
    
    /// <summary>
    /// Animation movement speed
    /// </summary>
    public float moveSpeed;
    
    /// <summary>
    /// Animation flags
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Rarity/probability of this sequence playing (0-10000)
    /// </summary>
    public uint rarity;
    
    /// <summary>
    /// Index of the sync point
    /// </summary>
    public uint syncPoint;
    
    /// <summary>
    /// Forward/backward extents of sequence
    /// </summary>
    public Vector3 boundingBox1;
    
    /// <summary>
    /// Forward/backward extents of sequence
    /// </summary>
    public Vector3 boundingBox2;
    
    /// <summary>
    /// Radius of bounding sphere
    /// </summary>
    public float boundingRadius;
}
```

## Properties

### SEQS Chunk
The SEQS chunk itself has no properties beyond the sequence array. Its size is determined by the number of MDLSEQUENCE structures it contains.

### MDLSEQUENCE Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | name | char[80] | Null-terminated sequence name |
| 0x50 | startTime | uint | Start time in milliseconds |
| 0x54 | endTime | uint | End time in milliseconds |
| 0x58 | moveSpeed | float | Movement speed during this animation |
| 0x5C | flags | uint | Sequence flags (see Flags section) |
| 0x60 | rarity | uint | Probability of this sequence (0-10000) |
| 0x64 | syncPoint | uint | Synchronization timing point |
| 0x68 | boundingBox1 | Vector3 | Minimum corner of sequence bounding box |
| 0x74 | boundingBox2 | Vector3 | Maximum corner of sequence bounding box |
| 0x80 | boundingRadius | float | Radius of bounding sphere for the sequence |

## Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | NonLooping | Sequence does not loop (plays once and stops) |
| 1 | Looping | Sequence loops continuously |
| 2 | NoMoveLoop | Do not repeat movement data when looping |
| 3 | StartAtFirst | Start at timestamp of first keyframe |
| 4 | EndAtLast | End at timestamp of last keyframe |
| 5-31 | Reserved | Reserved for future use, typically set to 0 |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure but with additional usage flags |

## Dependencies
- GLBS - If a sequence uses global sequences, it references data from the GLBS chunk
- All animation chunks that use timeline keyframes depend on SEQS for timing information

## Implementation Notes
- Sequence durations are in milliseconds
- The duration of a sequence is (endTime - startTime)
- Sequence names should be meaningful and match the animation's purpose (e.g., "Walk", "Attack1", "Death")
- The rarity field is primarily used for random animations, with higher values being more likely to play
- The syncPoint is used to align animations when transitioning between sequences
- Bounding boxes and radius are used for culling and collision during specific animations
- Models without animations may have empty SEQS chunks or a single static sequence
- The moveSpeed field is used for character movement during the animation (e.g., walk/run speeds)

## Usage Context
The SEQS chunk provides:
- The animation timeline context for keyframes in animation tracks
- Information about each available animation and its properties
- Movement speed for character animations
- Spatial bounds for each animation for culling and collision
- Flags that control animation playback behavior

## Animation System Integration
- The startTime and endTime define the sequence's time range
- Keyframe times in animation tracks are relative to these sequence times
- Global sequences (defined in GLBS) operate independently of normal sequences
- When a sequence loops, all animation tracks for that sequence also loop
- The NonLooping and Looping flags determine default playback behavior

## Implementation Example

```csharp
public class SEQSChunk : IMdxChunk
{
    public string ChunkId => "SEQS";
    
    public List<MdxSequence> Sequences { get; private set; } = new List<MdxSequence>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate number of sequences from chunk size
        int numSequences = (int)(size / 132); // Each MDLSEQUENCE is 132 bytes (80 + 4*13)
        
        // Clear any existing sequences
        Sequences.Clear();
        
        // Read all sequences
        for (int i = 0; i < numSequences; i++)
        {
            var sequence = new MdxSequence();
            
            // Read name (null-terminated string, 80 bytes)
            byte[] nameBytes = reader.ReadBytes(80);
            int nameLength = 0;
            while (nameLength < nameBytes.Length && nameBytes[nameLength] != 0)
            {
                nameLength++;
            }
            sequence.Name = System.Text.Encoding.ASCII.GetString(nameBytes, 0, nameLength);
            
            // Read sequence properties
            sequence.StartTime = reader.ReadUInt32();
            sequence.EndTime = reader.ReadUInt32();
            sequence.MoveSpeed = reader.ReadSingle();
            sequence.Flags = reader.ReadUInt32();
            sequence.Rarity = reader.ReadUInt32();
            sequence.SyncPoint = reader.ReadUInt32();
            
            // Read bounding box
            sequence.BoundingBoxMin = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            sequence.BoundingBoxMax = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            sequence.BoundingRadius = reader.ReadSingle();
            
            Sequences.Add(sequence);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var sequence in Sequences)
        {
            // Write name (pad with nulls to 80 bytes)
            byte[] nameBytes = new byte[80];
            if (!string.IsNullOrEmpty(sequence.Name))
            {
                byte[] temp = System.Text.Encoding.ASCII.GetBytes(sequence.Name);
                int copyLength = Math.Min(temp.Length, 79); // Leave at least one byte for null terminator
                Array.Copy(temp, nameBytes, copyLength);
            }
            writer.Write(nameBytes);
            
            // Write sequence properties
            writer.Write(sequence.StartTime);
            writer.Write(sequence.EndTime);
            writer.Write(sequence.MoveSpeed);
            writer.Write(sequence.Flags);
            writer.Write(sequence.Rarity);
            writer.Write(sequence.SyncPoint);
            
            // Write bounding box
            writer.Write(sequence.BoundingBoxMin.X);
            writer.Write(sequence.BoundingBoxMin.Y);
            writer.Write(sequence.BoundingBoxMin.Z);
            
            writer.Write(sequence.BoundingBoxMax.X);
            writer.Write(sequence.BoundingBoxMax.Y);
            writer.Write(sequence.BoundingBoxMax.Z);
            
            writer.Write(sequence.BoundingRadius);
        }
    }
}

public class MdxSequence
{
    public string Name { get; set; }
    public uint StartTime { get; set; }
    public uint EndTime { get; set; }
    public float MoveSpeed { get; set; }
    public uint Flags { get; set; }
    public uint Rarity { get; set; }
    public uint SyncPoint { get; set; }
    public Vector3 BoundingBoxMin { get; set; }
    public Vector3 BoundingBoxMax { get; set; }
    public float BoundingRadius { get; set; }
    
    // Flag accessors
    public bool NonLooping => (Flags & 0x1) != 0;
    public bool Looping => (Flags & 0x2) != 0;
    public bool NoMoveLoop => (Flags & 0x4) != 0;
    public bool StartAtFirst => (Flags & 0x8) != 0;
    public bool EndAtLast => (Flags & 0x10) != 0;
    
    // Helper properties
    public uint Duration => EndTime - StartTime;
    public bool IsValid => EndTime > StartTime;
}
``` 