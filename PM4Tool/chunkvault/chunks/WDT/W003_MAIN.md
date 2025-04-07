# W003: MAIN

## Type
WDT Chunk

## Source
WDT.md

## Description
The MAIN (Main Array) chunk contains a 64×64 grid of map tile entries, with each entry corresponding to a potential ADT file. Each entry indicates whether an ADT file exists for that grid position and contains the necessary flags or file ID to load it.

## Structure
```csharp
struct SMMapTileEntry
{
    /*0x00*/ uint32_t flags;       // Flags for this map tile
    /*0x04*/ uint32_t async_id;    // FileID for asynchronous loading (version 22+) or unused
};

struct MAIN
{
    /*0x00*/ SMMapTileEntry entries[64][64];  // 64×64 grid of map tile entries
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| entries | SMMapTileEntry[64][64] | Grid of map tile entries, each potentially corresponding to an ADT file |

## SMMapTileEntry Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32_t | Flags for this map tile, primarily indicating existence |
| async_id | uint32_t | FileID for async loading (in version 22+) or unused |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | FLAG_EXISTS | ADT file exists for this grid position |
| 0x2 | FLAG_LOADED | Internal flag used by the client to track loaded tiles |
| 0x4 | FLAG_LOADED_HIGH_RES | Internal flag for high-resolution texture loading |
| 0x8 | FLAG_IN_MEMORY | Internal flag indicating tile is in memory |
| 0x10 | FLAG_HIRES_TEXTURES | Tile has high-resolution textures |
| 0x20 | FLAG_MAPPED | For memory mapping (internal client use) |
| 0x40 | FLAG_HAS_TERRAIN | Tile has terrain data (vs. only WMO) |
| 0x80 | FLAG_UNUSED | Unused flag |

## Dependencies
- MVER (W001) - Version information determines exact format of MAIN entries
- MPHD (W002) - Header flags affect how MAIN entries are used

## Implementation Notes
- The MAIN chunk is required in all WDT files
- The grid is 64×64, matching the world grid system
- Only cells with the FLAG_EXISTS flag set should have corresponding ADT files
- In version 18 (Classic through WotLK), the async_id is unused
- In version 22+ (Cataclysm and later), the async_id may contain a FileDataID
- The entire grid must be present (4096 entries) even if most are empty
- The MAIN chunk is typically the largest chunk in a WDT file

## Implementation Example
```csharp
public class MapTileEntry
{
    [Flags]
    public enum TileFlags : uint
    {
        None = 0,
        Exists = 0x1,
        Loaded = 0x2,
        LoadedHighRes = 0x4,
        InMemory = 0x8,
        HasHighResTextures = 0x10,
        Mapped = 0x20,
        HasTerrain = 0x40,
        Unused = 0x80
    }
    
    public TileFlags Flags { get; set; }
    public uint AsyncId { get; set; }
    
    public bool Exists => (Flags & TileFlags.Exists) != 0;
    public bool HasHighResTextures => (Flags & TileFlags.HasHighResTextures) != 0;
    public bool HasTerrain => (Flags & TileFlags.HasTerrain) != 0;
}

public class MAIN : IChunk
{
    public const int GRID_SIZE = 64;
    public MapTileEntry[,] Entries { get; set; } = new MapTileEntry[GRID_SIZE, GRID_SIZE];
    
    public MAIN()
    {
        // Initialize all entries
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                Entries[y, x] = new MapTileEntry();
            }
        }
    }
    
    public void Parse(BinaryReader reader)
    {
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                MapTileEntry entry = new MapTileEntry
                {
                    Flags = (MapTileEntry.TileFlags)reader.ReadUInt32(),
                    AsyncId = reader.ReadUInt32()
                };
                Entries[y, x] = entry;
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                writer.Write((uint)Entries[y, x].Flags);
                writer.Write(Entries[y, x].AsyncId);
            }
        }
    }
    
    // Helper method to get ADT filename for a tile
    public string GetAdtFilename(int x, int y, string mapName)
    {
        if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE)
            throw new ArgumentOutOfRangeException();
            
        if (!Entries[y, x].Exists)
            return null; // No ADT file exists
            
        return $"{mapName}_{x}_{y}.adt";
    }
    
    // Helper method to count existing ADTs
    public int CountExistingTiles()
    {
        int count = 0;
        for (int y = 0; y < GRID_SIZE; y++)
        {
            for (int x = 0; x < GRID_SIZE; x++)
            {
                if (Entries[y, x].Exists)
                    count++;
            }
        }
        return count;
    }
}
```

## Version Differences
- **Version 18 (Classic - WotLK)**: async_id field is unused, set to 0
- **Version 22+ (Cataclysm and later)**: async_id field may contain a FileDataID for the corresponding ADT file
  - When FileDataID is used, the file is loaded from the game's database rather than by filename
  - This change supports the patch-based file system introduced in later expansions

## ADT File References
Each entry in the MAIN grid corresponds to a potential ADT file with coordinates in the filename:
- Format: `{mapName}_{x}_{y}.adt`
- Example: `Kalimdor_31_42.adt` (for x=31, y=42 on the Kalimdor map)
- Only entries with FLAG_EXISTS set should have corresponding ADT files
- In version 22+, the async_id may be used instead of constructing a filename

## Usage Context
The MAIN chunk is the core of the WDT file, defining which ADT files make up the map. The client uses this grid to:
- Determine which ADT files to load based on player position
- Manage the loading/unloading of map tiles as the player moves
- Track which areas of the map have terrain data
- Establish the overall dimensions and layout of the map

This chunk effectively serves as a map of maps, defining the presence and properties of each map tile in the world. 