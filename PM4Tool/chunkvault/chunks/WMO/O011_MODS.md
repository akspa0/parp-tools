# O011: MODS

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MODS (Map Object DoodadSets) chunk defines collections of doodads that can be shown or hidden together as a group. Each doodad set represents a particular configuration of doodads for the WMO, allowing for variations such as seasonal decorations, damage states, or different usage scenarios. By controlling which sets are active, the game can display different variations of the same WMO without needing multiple complete model files.

## Structure
```csharp
struct MODS
{
    SMODoodadSet[] doodadSets;  // Array of doodad set definitions
};

struct SMODoodadSet
{
    /*0x00*/ char[20] name;       // Name of the doodad set (null-terminated)
    /*0x14*/ uint32_t startIndex; // First doodad in the set (index into MODD array)
    /*0x18*/ uint32_t count;      // Number of doodads in this set
    /*0x1C*/ uint32_t unused;     // Unused field, typically set to 0
};
```

## Properties
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | name | char[20] | Name of the doodad set (null-terminated string) |
| 0x14 | startIndex | uint32_t | Index of the first doodad in this set (in MODD array) |
| 0x18 | count | uint32_t | Number of consecutive doodads in this set |
| 0x1C | unused | uint32_t | Unused field (padding or reserved for future use) |

## Dependencies
- MOHD: The nDoodadSets field indicates how many doodad set definitions should be present
- MODD: Contains the doodad definitions referenced by startIndex and count

## Implementation Notes
- Each doodad set definition is 32 bytes (0x20)
- The name field is a fixed-size array of 20 characters, null-terminated
- If the name is shorter than 20 characters, the remaining bytes should be filled with zeros
- The startIndex is an index into the MODD array, not a byte offset
- Doodads belonging to a set are stored contiguously in the MODD array
- The count field indicates how many consecutive doodad entries belong to this set
- The unused field is typically set to zero and should be ignored
- Doodad sets are used for showing/hiding groups of doodads based on game events, seasons, or player actions
- The number of doodad sets must match the nDoodadSets field in the MOHD chunk

## Implementation Example
```csharp
public class MODS : IChunk
{
    public const int NAME_LENGTH = 20;
    public List<DoodadSet> Sets { get; private set; }
    
    public MODS()
    {
        Sets = new List<DoodadSet>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many doodad sets we expect
        int setCount = (int)(size / 0x20); // Each set is 32 bytes
        
        Sets.Clear();
        
        for (int i = 0; i < setCount; i++)
        {
            DoodadSet set = new DoodadSet();
            
            // Read name (fixed 20 bytes)
            byte[] nameBytes = reader.ReadBytes(NAME_LENGTH);
            
            // Find null terminator
            int nameLength = 0;
            while (nameLength < nameBytes.Length && nameBytes[nameLength] != 0)
            {
                nameLength++;
            }
            
            // Convert to string
            set.Name = Encoding.ASCII.GetString(nameBytes, 0, nameLength);
            
            set.StartIndex = reader.ReadUInt32();
            set.Count = reader.ReadUInt32();
            set.Unused = reader.ReadUInt32();
            
            Sets.Add(set);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (DoodadSet set in Sets)
        {
            // Write name (fixed 20 bytes, null-terminated)
            byte[] nameBytes = new byte[NAME_LENGTH];
            
            if (!string.IsNullOrEmpty(set.Name))
            {
                byte[] actualNameBytes = Encoding.ASCII.GetBytes(set.Name);
                int copyLength = Math.Min(actualNameBytes.Length, NAME_LENGTH - 1); // -1 to ensure space for null terminator
                Buffer.BlockCopy(actualNameBytes, 0, nameBytes, 0, copyLength);
            }
            
            // Ensure null termination
            if (set.Name.Length < NAME_LENGTH)
            {
                nameBytes[set.Name.Length] = 0;
            }
            else
            {
                nameBytes[NAME_LENGTH - 1] = 0;
            }
            
            writer.Write(nameBytes);
            writer.Write(set.StartIndex);
            writer.Write(set.Count);
            writer.Write(set.Unused);
        }
    }
    
    public DoodadSet GetDoodadSet(int index)
    {
        if (index >= 0 && index < Sets.Count)
        {
            return Sets[index];
        }
        
        throw new IndexOutOfRangeException($"Doodad set index {index} is out of range. Valid range: 0-{Sets.Count - 1}");
    }
    
    public DoodadSet GetDoodadSetByName(string name)
    {
        return Sets.FirstOrDefault(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
    }
    
    public void AddDoodadSet(string name, uint startIndex, uint count)
    {
        DoodadSet set = new DoodadSet
        {
            Name = name,
            StartIndex = startIndex,
            Count = count,
            Unused = 0
        };
        
        Sets.Add(set);
    }
}

public class DoodadSet
{
    public string Name { get; set; }
    public uint StartIndex { get; set; }
    public uint Count { get; set; }
    public uint Unused { get; set; }
    
    public DoodadSet()
    {
        Name = string.Empty;
        StartIndex = 0;
        Count = 0;
        Unused = 0;
    }
}
```

## Validation Requirements
- The number of doodad sets should match the nDoodadSets field in the MOHD chunk
- Each name should be null-terminated and within the 20-byte limit
- The startIndex should be within the bounds of the MODD array
- The startIndex + count should not exceed the total number of doodads in MODD
- Each set should have a unique name for identification
- The unused field should be set to zero

## Usage Context
The MODS chunk enables flexible control over doodad visibility and WMO variations:

1. **Visual Variations**: Different doodad sets can represent various states of the same building
2. **Seasonal Changes**: Holiday or seasonal decorations can be shown or hidden as appropriate
3. **Progressive Changes**: Buildings can change appearance (like damage or construction) as game events progress
4. **Instance Variations**: Different instances of a dungeon can show different doodad configurations
5. **Optimization**: Allows selectively showing only relevant doodads based on player position or game state

Common naming patterns for doodad sets include:
- "Set_Default" or "Main" for the standard configuration
- "Set_Winter", "Set_Summer" for seasonal variations
- "Set_Damaged", "Set_Pristine" for different damage states
- "Set_Interior", "Set_Exterior" for position-based visibility
- "Set_Alliance", "Set_Horde" for faction-specific decorations

When rendering a WMO, the client:
1. Determines which doodad set(s) should be active based on game state
2. For each active set, identifies which doodads to display using startIndex and count
3. Loads and renders only the doodads from active sets

This system provides an efficient way to create visual variety without duplicating entire models, saving both file size and memory usage while allowing for contextual changes to the game world. 