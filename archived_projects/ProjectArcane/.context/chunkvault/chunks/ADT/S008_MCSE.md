# S008: MCSE

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCSE (Map Chunk Sound Emitters) subchunk contains information about sound emitters placed in an MCNK chunk. These sound emitters add ambient sounds to the environment, such as bird calls, water sounds, or wind.

## Structure
```csharp
struct SMSoundEmitter
{
    /*0x00*/ C3Vector position;         // Sound emitter position
    /*0x0C*/ uint32_t sound_id;         // Sound ID from SoundEntries.dbc
    /*0x10*/ uint32_t sound_name;       // Sound name from SoundEntries.dbc
    /*0x14*/ C3Vector size;             // Size of the emitter
    /*0x20*/ float frequency;           // Frequency of the sound
    /*0x24*/ uint32_t flags;            // Emitter flags
};

struct MCSE
{
    /*0x00*/ SMSoundEmitter sound_emitters[n];  // Array of sound emitters, where n is num_sound_emitters from MCNK header
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| sound_emitters | SMSoundEmitter[] | Array of sound emitter structures |

## SMSoundEmitter Properties
| Name | Type | Description |
|------|------|-------------|
| position | C3Vector | 3D position of the sound emitter |
| sound_id | uint32 | Sound ID from SoundEntries.dbc |
| sound_name | uint32 | Sound name from SoundEntries.dbc |
| size | C3Vector | Size/range of the sound emitter |
| frequency | float | How frequently the sound plays |
| flags | uint32 | Flags that control the emitter behavior |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | SOUND_LOOPING | Sound loops continuously |
| 0x2 | SOUND_RANDOM | Sound plays randomly |
| 0x4 | SOUND_DISTANCE_BASED | Volume varies with distance |
| 0x8 | SOUND_DAY_ONLY | Only plays during daytime |
| 0x10 | SOUND_NIGHT_ONLY | Only plays during nighttime |
| 0x20 | SOUND_INDOOR | For indoor environments |
| 0x40 | SOUND_OUTDOOR | For outdoor environments |

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCNK.mcse - Offset to this subchunk
- MCNK.num_sound_emitters - Number of sound emitters

## Presence Determination
This subchunk is only present when:
- MCNK.mcse offset is non-zero
- MCNK.num_sound_emitters is greater than 0

## Implementation Notes
- The position of the sound emitter is in world coordinates
- The size affects the audible range of the sound
- Frequency controls how often the sound plays (for non-looping sounds)
- Sound IDs and names reference entries in SoundEntries.dbc
- Multiple emitters can exist in a single MCNK
- Flags control when and how the sound plays

## Implementation Example
```csharp
public class SoundEmitter
{
    [Flags]
    public enum SoundFlags : uint
    {
        None = 0,
        Looping = 0x1,
        Random = 0x2,
        DistanceBased = 0x4,
        DayOnly = 0x8,
        NightOnly = 0x10,
        Indoor = 0x20,
        Outdoor = 0x40
    }

    public C3Vector Position { get; set; }
    public uint SoundId { get; set; }
    public uint SoundName { get; set; }
    public C3Vector Size { get; set; }
    public float Frequency { get; set; }
    public SoundFlags Flags { get; set; }
    
    public bool IsLooping => (Flags & SoundFlags.Looping) != 0;
    public bool IsRandom => (Flags & SoundFlags.Random) != 0;
    public bool IsDistanceBased => (Flags & SoundFlags.DistanceBased) != 0;
    public bool IsDayOnly => (Flags & SoundFlags.DayOnly) != 0;
    public bool IsNightOnly => (Flags & SoundFlags.NightOnly) != 0;
    public bool IsIndoor => (Flags & SoundFlags.Indoor) != 0;
    public bool IsOutdoor => (Flags & SoundFlags.Outdoor) != 0;
}

public class MCSE : IChunk
{
    public List<SoundEmitter> SoundEmitters { get; set; } = new List<SoundEmitter>();
    
    public void Parse(BinaryReader reader, uint numSoundEmitters)
    {
        for (int i = 0; i < numSoundEmitters; i++)
        {
            var emitter = new SoundEmitter
            {
                Position = new C3Vector
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                SoundId = reader.ReadUInt32(),
                SoundName = reader.ReadUInt32(),
                Size = new C3Vector
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Frequency = reader.ReadSingle(),
                Flags = (SoundEmitter.SoundFlags)reader.ReadUInt32()
            };
            
            SoundEmitters.Add(emitter);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var emitter in SoundEmitters)
        {
            writer.Write(emitter.Position.X);
            writer.Write(emitter.Position.Y);
            writer.Write(emitter.Position.Z);
            writer.Write(emitter.SoundId);
            writer.Write(emitter.SoundName);
            writer.Write(emitter.Size.X);
            writer.Write(emitter.Size.Y);
            writer.Write(emitter.Size.Z);
            writer.Write(emitter.Frequency);
            writer.Write((uint)emitter.Flags);
        }
    }
}
```

## Sound System Integration
Sound emitters work with the game's ambient sound system:
- The client checks for sound emitters in the player's vicinity
- Sound emitters activate based on distance from the player
- Flags determine when sounds can play (day/night, indoor/outdoor)
- For random sounds, the frequency determines likelihood of playing
- The size vector affects the audible range and falloff

## Usage Context
The MCSE subchunk enhances the immersion of the game world by adding ambient sounds to specific locations. For example, a waterfall might have a water sound emitter, a forest might have bird and insect sounds, and a cave might have dripping water or wind sounds. These sound emitters create a more dynamic and immersive audio experience as players move through the world, with sounds naturally fading in and out based on position and other environmental factors. 