# WA06: MODR

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MODR (Map Object Directory) chunk contains structural information about map objects and their hierarchical relationships in the Alpha WDT format. This chunk defines the organization and grouping of objects within the map, creating a directory-like structure that allows for logical arrangement and efficient access to map elements.

## Structure
```csharp
struct MODR
{
    struct DirectoryEntry
    {
        /*0x00*/ uint32_t flags;           // Entry flags
        /*0x04*/ uint32_t name_offset;     // Offset to name string in MOTX chunk
        /*0x08*/ uint32_t object_id;       // Unique identifier
        /*0x0C*/ uint32_t parent_id;       // Parent directory ID (0 for root)
        /*0x10*/ uint32_t first_child_id;  // ID of first child (0 if none)
        /*0x14*/ uint32_t next_sibling_id; // ID of next sibling (0 if none)
        /*0x18*/ uint32_t data_offset;     // Offset to data in MAOI chunk
        /*0x1C*/ uint32_t data_size;       // Size of data in MAOI chunk
    };

    DirectoryEntry[] entries;              // Array of directory entries
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| entries | DirectoryEntry[] | Array of directory entries defining object structure |

### DirectoryEntry Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32_t | Flags indicating entry properties |
| name_offset | uint32_t | Offset to entry name in MOTX chunk |
| object_id | uint32_t | Unique identifier for this entry |
| parent_id | uint32_t | ID of parent directory (0 for root level) |
| first_child_id | uint32_t | ID of first child entry (0 if none) |
| next_sibling_id | uint32_t | ID of next sibling entry (0 if none) |
| data_offset | uint32_t | Offset to entry data in MAOI chunk |
| data_size | uint32_t | Size of entry data in MAOI chunk |

## Flag Values
The flags field may contain bits indicating various entry properties:

| Value | Name | Description |
|-------|------|-------------|
| 0x01 | MODR_DIRECTORY | Entry is a directory/group (not a leaf object) |
| 0x02 | MODR_OBJECT | Entry is an actual map object |
| 0x04 | MODR_HIDDEN | Entry is hidden by default |
| 0x08 | MODR_LOCKED | Entry is locked (cannot be modified) |
| 0x10 | MODR_PLACEHOLDER | Entry is a placeholder/stub |
| 0x20 | MODR_SPECIAL_RENDER | Entry has special rendering requirements |
| 0x40 | MODR_SYSTEM | Entry is a system object (internal use) |
| 0x80 | MODR_INSTANCE | Entry is an instance of another object |

These flag values are speculative and would require further research to confirm.

## Dependencies
- MOTX (WA04) - Contains name strings referenced by name_offset
- MAOI (WA02) - Contains object data referenced by data_offset
- MAOT (WA01) - May indirectly reference objects through MAOI
- MAOH (WA03) - May contain global settings affecting directory structure

## Implementation Notes
- The MODR chunk creates a hierarchical organization of map objects
- Entries form a tree structure with parent-child relationships
- Sibling entries at the same level are linked via next_sibling_id
- Directory entries with the MODR_DIRECTORY flag set contain other entries
- Leaf entries with the MODR_OBJECT flag set represent actual map objects
- The name strings in MOTX typically follow a path-like structure (e.g., "Buildings/Stormwind/Castle")
- Object data in MAOI contains the actual geometry, textures, and properties
- The directory structure allows for logical grouping and efficient traversal
- This chunk is essential for map editing tools and runtime object management

## Implementation Example
```csharp
public class MODR : IChunk
{
    public class DirectoryEntry
    {
        public uint Flags { get; set; }
        public uint NameOffset { get; set; }
        public uint ObjectId { get; set; }
        public uint ParentId { get; set; }
        public uint FirstChildId { get; set; }
        public uint NextSiblingId { get; set; }
        public uint DataOffset { get; set; }
        public uint DataSize { get; set; }
        
        // Helper properties for flag checking
        public bool IsDirectory => (Flags & 0x01) != 0;
        public bool IsObject => (Flags & 0x02) != 0;
        public bool IsHidden => (Flags & 0x04) != 0;
        public bool IsLocked => (Flags & 0x08) != 0;
        public bool IsPlaceholder => (Flags & 0x10) != 0;
        public bool HasSpecialRender => (Flags & 0x20) != 0;
        public bool IsSystem => (Flags & 0x40) != 0;
        public bool IsInstance => (Flags & 0x80) != 0;
        
        // Cached name (to be populated from MOTX)
        public string Name { get; set; }
        
        // Cached children and parent (to be populated during tree building)
        public DirectoryEntry Parent { get; set; }
        public List<DirectoryEntry> Children { get; set; } = new List<DirectoryEntry>();
    }
    
    public List<DirectoryEntry> Entries { get; private set; } = new List<DirectoryEntry>();
    public Dictionary<uint, DirectoryEntry> EntriesById { get; private set; } = new Dictionary<uint, DirectoryEntry>();
    public List<DirectoryEntry> RootEntries { get; private set; } = new List<DirectoryEntry>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many entries are in the chunk
        int entryCount = (int)(size / 32); // Each entry is 32 bytes
        
        // Read all entries
        for (int i = 0; i < entryCount; i++)
        {
            DirectoryEntry entry = new DirectoryEntry
            {
                Flags = reader.ReadUInt32(),
                NameOffset = reader.ReadUInt32(),
                ObjectId = reader.ReadUInt32(),
                ParentId = reader.ReadUInt32(),
                FirstChildId = reader.ReadUInt32(),
                NextSiblingId = reader.ReadUInt32(),
                DataOffset = reader.ReadUInt32(),
                DataSize = reader.ReadUInt32()
            };
            
            Entries.Add(entry);
            EntriesById[entry.ObjectId] = entry;
        }
        
        // Build the hierarchical structure
        BuildDirectoryTree();
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            writer.Write(entry.Flags);
            writer.Write(entry.NameOffset);
            writer.Write(entry.ObjectId);
            writer.Write(entry.ParentId);
            writer.Write(entry.FirstChildId);
            writer.Write(entry.NextSiblingId);
            writer.Write(entry.DataOffset);
            writer.Write(entry.DataSize);
        }
    }
    
    // Build the directory tree structure from the flat list of entries
    private void BuildDirectoryTree()
    {
        RootEntries.Clear();
        
        // First pass: establish parent-child relationships
        foreach (var entry in Entries)
        {
            // Root entries have ParentId == 0
            if (entry.ParentId == 0)
            {
                RootEntries.Add(entry);
            }
            else
            {
                // Link to parent
                if (EntriesById.TryGetValue(entry.ParentId, out DirectoryEntry parent))
                {
                    entry.Parent = parent;
                }
            }
        }
        
        // Second pass: establish child and sibling relationships
        foreach (var entry in Entries)
        {
            // Process FirstChildId
            if (entry.FirstChildId != 0 && EntriesById.TryGetValue(entry.FirstChildId, out DirectoryEntry firstChild))
            {
                entry.Children.Add(firstChild);
                
                // Process all siblings
                uint siblingId = firstChild.NextSiblingId;
                while (siblingId != 0)
                {
                    if (EntriesById.TryGetValue(siblingId, out DirectoryEntry sibling))
                    {
                        entry.Children.Add(sibling);
                        siblingId = sibling.NextSiblingId;
                    }
                    else
                    {
                        break; // Invalid sibling ID
                    }
                }
            }
        }
    }
    
    // Populate entry names from MOTX chunk
    public void PopulateNames(MOTX motxChunk)
    {
        foreach (var entry in Entries)
        {
            entry.Name = motxChunk.GetTextureNameByOffset((int)entry.NameOffset);
        }
    }
    
    // Helper method to traverse the directory tree using a visitor pattern
    public void TraverseTree(DirectoryEntry startEntry, IDirectoryVisitor visitor)
    {
        if (startEntry == null)
        {
            // Start with root entries if no specific entry is provided
            foreach (var rootEntry in RootEntries)
            {
                TraverseTreeRecursive(rootEntry, visitor, 0);
            }
        }
        else
        {
            TraverseTreeRecursive(startEntry, visitor, 0);
        }
    }
    
    private void TraverseTreeRecursive(DirectoryEntry entry, IDirectoryVisitor visitor, int depth)
    {
        if (!visitor.VisitEntry(entry, depth))
            return;
            
        foreach (var child in entry.Children)
        {
            TraverseTreeRecursive(child, visitor, depth + 1);
        }
    }
}

// Interface for directory tree traversal using visitor pattern
public interface IDirectoryVisitor
{
    bool VisitEntry(MODR.DirectoryEntry entry, int depth);
}
```

## Directory Hierarchies
The MODR chunk typically organizes objects in logical hierarchies such as:

```
World
├── Terrain
│   ├── Mountains
│   ├── Plains
│   └── Rivers
├── Buildings
│   ├── Stormwind
│   │   ├── Castle
│   │   ├── Cathedral
│   │   └── Houses
│   └── Goldshire
│       └── Inn
└── Environment
    ├── Trees
    ├── Rocks
    └── Foliage
```

This structure allows for efficient object management and selective loading/rendering.

## Version Information
- Present only in the Alpha version of the WDT format
- In later versions, hierarchical structure data was moved into individual WMO files
- The approach to object organization evolved in later versions to include more specialized file formats

## Architectural Significance
The MODR chunk exemplifies the Alpha WDT format's approach to object organization:

1. **Centralized Object Management**: All map objects organized in a single hierarchical structure
2. **Logical Grouping**: Objects grouped by type, location, or function
3. **Efficient Traversal**: Tree structure enables quick navigation and selective processing
4. **Integrated References**: Direct references to object data in other chunks

This contrasts with the modern approach where:
- Object hierarchy is distributed across multiple files
- WMO files contain their own internal object organization
- The world is divided into discrete ADT tiles with separate object references
- Object groups are more specialized by type (doodads, WMOs, etc.) 