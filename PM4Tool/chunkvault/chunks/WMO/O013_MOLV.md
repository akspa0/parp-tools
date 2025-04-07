# O013: MOLV

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOLV (Map Object Light Volumes) chunk extends the MOLT chunk by providing additional light volume information. This chunk was introduced in expansion level 9.1 (first seen in the Broker Dungeon, file 3623016) and should not be confused with the old 0.5.3 MOLV chunk. It allows for more complex lighting behaviors by defining directional properties and values for lights.

## Structure
```cpp
struct {
 /*0x00*/  struct {
   /*0x00*/  C3Vector direction;      // usually either xy or z and the remainder 0.
   /*0x0C*/  float value;
   /*0x10*/ } _0x00[6];
 /*0x60*/  byte _0x60[3];
 /*0x63*/  uint8_t molt_index;         // multiple MOLV may reference/extend the same MOLT.
 /*0x64*/
} mapObjectLightV[];
```

### C# Structure
```csharp
struct MOLV
{
    SMOLightVolume[] lightVolumes;  // Array of light volume definitions
};

struct SMOLightVolume
{
    /*0x00*/ SMOLightVolumeDirection[6] directions;  // Array of 6 direction/value pairs
    /*0x60*/ byte[3] unknown;                        // Unknown values
    /*0x63*/ uint8_t moldIndex;                      // Index into MOLT array
};

struct SMOLightVolumeDirection
{
    /*0x00*/ C3Vector direction;  // Usually either xy or z components with remainder 0
    /*0x0C*/ float value;        // Associated value for this direction
};
```

## Properties

### SMOLightVolume Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | directions | SMOLightVolumeDirection[6] | Array of 6 direction/value pairs |
| 0x60 | unknown | byte[3] | Unknown values |
| 0x63 | moldIndex | uint8_t | Index into MOLT array |

### SMOLightVolumeDirection Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | direction | C3Vector | Direction vector (usually has either xy or z components with remainder 0) |
| 0x0C | value | float | Associated value for this direction |

## Dependencies
- MOLT: Referenced by moldIndex, multiple MOLV entries may extend the same MOLT entry

## Implementation Notes
- Each light volume is 100 bytes (0x64)
- Each direction/value pair is 16 bytes (0x10)
- The direction vectors typically have either:
  - X and Y components with Z = 0
  - Z component with X and Y = 0
- Multiple MOLV entries can reference and extend the same MOLT entry
- The unknown bytes at 0x60 may contain flags or additional parameters
- Total size of the chunk should be a multiple of 100 bytes

## Implementation Example
```csharp
public class MOLV : IChunk
{
    public List<LightVolume> LightVolumes { get; private set; }
    
    public MOLV()
    {
        LightVolumes = new List<LightVolume>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many light volumes we expect
        int volumeCount = (int)(size / 100); // Each volume is 100 bytes
        
        LightVolumes.Clear();
        
        for (int i = 0; i < volumeCount; i++)
        {
            var volume = new LightVolume();
            
            // Read 6 direction/value pairs
            volume.Directions = new LightVolumeDirection[6];
            for (int j = 0; j < 6; j++)
            {
                var direction = new LightVolumeDirection
                {
                    Direction = new Vector3(
                        reader.ReadSingle(),
                        reader.ReadSingle(),
                        reader.ReadSingle()
                    ),
                    Value = reader.ReadSingle()
                };
                volume.Directions[j] = direction;
            }
            
            // Read unknown bytes and MOLT index
            volume.Unknown = new byte[3];
            reader.Read(volume.Unknown, 0, 3);
            volume.MoltIndex = reader.ReadByte();
            
            LightVolumes.Add(volume);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var volume in LightVolumes)
        {
            // Write 6 direction/value pairs
            foreach (var direction in volume.Directions)
            {
                writer.Write(direction.Direction.X);
                writer.Write(direction.Direction.Y);
                writer.Write(direction.Direction.Z);
                writer.Write(direction.Value);
            }
            
            // Write unknown bytes and MOLT index
            writer.Write(volume.Unknown);
            writer.Write(volume.MoltIndex);
        }
    }
}

public class LightVolume
{
    public LightVolumeDirection[] Directions { get; set; }
    public byte[] Unknown { get; set; }
    public byte MoltIndex { get; set; }
}

public class LightVolumeDirection
{
    public Vector3 Direction { get; set; }
    public float Value { get; set; }
}
```

## Validation Requirements
- The chunk size must be a multiple of 100 bytes
- Each light volume must have exactly 6 direction/value pairs
- Direction vectors should be normalized
- MOLT indices must be valid (within bounds of MOLT array)
- Unknown bytes should be preserved but not modified
- Each direction vector should typically follow the xy-or-z pattern

## Usage Context
Light volumes in WMOs provide more sophisticated lighting control:

1. **Directional Control**: 
   - Allows light to behave differently in different directions
   - Can create asymmetric lighting effects
   - Useful for simulating complex light sources

2. **MOLT Extension**:
   - Extends basic light definitions from MOLT
   - Multiple volumes can affect the same base light
   - Provides more granular control over light behavior

3. **Common Applications**:
   - Complex architectural lighting
   - Environmental effects
   - Dynamic lighting scenarios
   - Special visual effects

The lighting system uses these volumes to:
1. Modify base light behavior defined in MOLT
2. Create directionally-varying light effects
3. Provide more realistic or stylized lighting control
4. Support complex lighting scenarios in modern WMO files 