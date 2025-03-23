# MOPY - Material Info for Triangles

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOPY chunk contains material information for triangles in a WMO group. It includes flags that control visibility, collision, culling, and other rendering properties, as well as a material ID that references the appropriate material from the MOMT chunk in the root file. Each triangle in the WMO group has a corresponding entry in this chunk.

## Structure

```csharp
public struct MOPYEntry
{
    public byte Flags;      // Bit flags controlling rendering and collision properties
    public byte MaterialId; // Index into the MOMT chunk in the root file (0xFF for collision-only faces)
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Flags | byte | Bit flags controlling various properties of the triangle |
| 0x01 | MaterialId | byte | Index into the MOMT chunk in the root file, or 0xFF for collision-only triangles |

## Flag Values (Offset 0x00)

| Value | Flag | Description |
|-------|------|-------------|
| 0x01 | F_UNK_0x01 | Unknown flag, used with F_DETAIL or F_RENDER to mark transition triangles |
| 0x02 | F_NOCAMCOLLIDE | Triangle does not collide with camera |
| 0x04 | F_DETAIL | Triangle is a detail face |
| 0x08 | F_COLLISION | Turns off rendering of water ripple effects, used for ghost material triangles |
| 0x10 | F_HINT | Triangle is a hint |
| 0x20 | F_RENDER | Triangle should be rendered |
| 0x40 | F_CULL_OBJECTS | Enables/disables game object culling |
| 0x80 | F_COLLIDE_HIT | Triangle participates in collision detection |

## Flag Combinations

| Combination | Result | Description |
|-------------|--------|-------------|
| F_UNK_0x01 && (F_DETAIL \|\| F_RENDER) | Transition Face | These triangles blend lighting from exterior to interior |
| !F_COLLISION | Color Face | Triangle should receive color |
| F_RENDER && !F_DETAIL | Render Face | Normal rendered triangle |
| F_COLLISION \|\| (F_RENDER && !F_DETAIL) | Collidable Face | Triangle participates in collision detection |

## MaterialId Values

| Value | Description |
|-------|-------------|
| 0-254 | Index into the MOMT chunk in the root file |
| 0xFF | Collision-only triangle that is not rendered but still has collision |

## Dependencies
- MOMT chunk in the root file (referenced by MaterialId)
- MOVI chunk (defines the triangle indices that correspond to these material entries)

## Implementation Notes
- Each entry in this chunk corresponds to one triangle in the WMO group.
- Triangles are generally pre-sorted by texture for more efficient rendering.
- Collision-only triangles (MaterialId = 0xFF) are not rendered but still participate in collision detection.
- The size of this chunk in bytes will be twice the number of triangles in the WMO group.
- In newer versions of the format (expansion level 10+), this chunk may be replaced by MPY2, which allows for more material information.
- Some combinations of flags have special meaning, particularly for handling transitions between interior and exterior lighting.

## Implementation Example

```csharp
public class MOPYChunk : IWmoGroupChunk
{
    public string ChunkId => "MOPY";
    
    public List<MOPYEntry> Entries { get; } = new List<MOPYEntry>();
    
    public void Read(BinaryReader reader, uint size)
    {
        // Each entry is 2 bytes, calculate the number of entries
        int entryCount = (int)(size / 2);
        Entries.Clear();
        
        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MOPYEntry
            {
                Flags = reader.ReadByte(),
                MaterialId = reader.ReadByte()
            };
            Entries.Add(entry);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        writer.Write((uint)(Entries.Count * 2)); // Size = entryCount * 2 bytes per entry
        
        // Write entries
        foreach (var entry in Entries)
        {
            writer.Write(entry.Flags);
            writer.Write(entry.MaterialId);
        }
    }
    
    public bool IsTransitionFace(int index)
    {
        if (index < 0 || index >= Entries.Count)
            return false;
            
        byte flags = Entries[index].Flags;
        return (flags & 0x01) != 0 && ((flags & 0x04) != 0 || (flags & 0x20) != 0);
    }
    
    public bool IsColorFace(int index)
    {
        if (index < 0 || index >= Entries.Count)
            return false;
            
        return (Entries[index].Flags & 0x08) == 0;
    }
    
    public bool IsRenderFace(int index)
    {
        if (index < 0 || index >= Entries.Count)
            return false;
            
        byte flags = Entries[index].Flags;
        return (flags & 0x20) != 0 && (flags & 0x04) == 0;
    }
    
    public bool IsCollidableFace(int index)
    {
        if (index < 0 || index >= Entries.Count)
            return false;
            
        byte flags = Entries[index].Flags;
        return (flags & 0x08) != 0 || ((flags & 0x20) != 0 && (flags & 0x04) == 0);
    }
}

public struct MOPYEntry
{
    public byte Flags;
    public byte MaterialId;
    
    // Helper properties for common flag combinations
    public bool IsTransitionFace => (Flags & 0x01) != 0 && ((Flags & 0x04) != 0 || (Flags & 0x20) != 0);
    public bool IsColorFace => (Flags & 0x08) == 0;
    public bool IsRenderFace => (Flags & 0x20) != 0 && (Flags & 0x04) == 0;
    public bool IsCollidableFace => (Flags & 0x08) != 0 || IsRenderFace;
}
```

## Validation Requirements
- The size of the chunk should be exactly twice the number of triangles.
- Each entry should correspond to one triangle in the MOVI chunk.
- Material IDs should be valid indices into the MOMT chunk in the root file, or 0xFF for collision-only faces.
- Flag combinations should be checked for consistency (e.g., a face cannot have both F_COLLISION and F_RENDER with !F_DETAIL).

## Usage Context
- **Material Assignment**: Associates triangles with materials from the root file.
- **Rendering Control**: Determines which triangles are visible and how they're rendered.
- **Collision Detection**: Controls which triangles participate in collision detection.
- **Special Rendering Effects**: Flags control specific rendering behaviors like transition faces.
- **Optimization**: Triangles are pre-sorted by texture for efficient rendering. 