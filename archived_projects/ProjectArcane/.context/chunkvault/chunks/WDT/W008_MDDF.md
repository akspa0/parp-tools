# W008: MDDF

## Type
WDT Chunk

## Source
WDT.md

## Description
The MDDF (Map Doodad Definition) chunk contains placement information for global M2 doodad models in the world map. Similar to how MODF places global WMO models, MDDF defines where M2 models (doodads) are positioned, rotated, and scaled. This chunk works together with MDNM and MDID chunks to define global doodad placements.

## Structure
```csharp
struct SMDoodadDef // sizeof(0x24)
{
    /*0x00*/ uint32_t name_id;        // Index into MDID list for the filename
    /*0x04*/ uint32_t unique_id;      // Unique identifier for this instance
    /*0x08*/ Vector3 position;        // Position (X, Y, Z) coordinates
    /*0x14*/ Vector3 rotation;        // Rotation (X, Y, Z) in radians
    /*0x20*/ float scale;             // Uniform scale factor 
};

struct MDDF
{
    /*0x00*/ SMDoodadDef doodad_instances[];  // Array of doodad instance definitions
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| doodad_instances | SMDoodadDef[] | Array of doodad instance placement data |

### SMDoodadDef Properties
| Name | Type | Description |
|------|------|-------------|
| name_id | uint32_t | Index into the MDID list for the filename |
| unique_id | uint32_t | Unique identifier for this instance |
| position | Vector3 | Position coordinates (X, Y, Z) in world space |
| rotation | Vector3 | Rotation angles (X, Y, Z) in radians |
| scale | float | Uniform scale factor for the model |

## Dependencies
- MDNM (W007) - Contains the M2 filenames
- MDID (W009) - Contains offsets into the MDNM chunk for filenames

## Implementation Notes
- The MDDF chunk is only present if the map contains global doodad objects
- Each entry in the `doodad_instances` array represents one M2 model placement in the world
- The `name_id` field references an index in the MDID array, which in turn provides an offset into the MDNM string table
- Rotation values are in radians (not degrees)
- The chunk size should be a multiple of 0x24 (36 bytes, the size of SMDoodadDef)
- The number of entries can be calculated by dividing the chunk size by the structure size
- The `unique_id` field should be globally unique for each doodad instance
- Unlike the MODF structure, the scale is a single float value (uniform scaling) rather than a Vector3
- In versions 22+, the filename may be referenced by FileDataID rather than through MDID/MDNM

## Implementation Example
```csharp
public class SMDoodadDef
{
    public uint NameId { get; set; }        // Index into MDID
    public uint UniqueId { get; set; }       // Unique identifier
    public Vector3 Position { get; set; }    // X, Y, Z position
    public Vector3 Rotation { get; set; }    // X, Y, Z rotation in radians
    public float Scale { get; set; }         // Uniform scale factor
    
    // Helper method to get the M2 filename using MDID and MDNM
    public string GetFilename(MDID mdid, MDNM mdnm)
    {
        if (NameId >= mdid.Offsets.Count)
            throw new IndexOutOfRangeException($"NameId {NameId} is out of range");
            
        uint offset = mdid.GetOffset((int)NameId);
        return mdnm.GetFilenameByOffset((int)offset);
    }
}

public class MDDF : IChunk
{
    public List<SMDoodadDef> DoodadInstances { get; private set; } = new List<SMDoodadDef>();
    
    public void Parse(BinaryReader reader, long size)
    {
        const int ENTRY_SIZE = 0x24; // 36 bytes per entry
        int count = (int)(size / ENTRY_SIZE);
        DoodadInstances.Clear();
        
        for (int i = 0; i < count; i++)
        {
            var instance = new SMDoodadDef
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Rotation = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Scale = reader.ReadSingle()
            };
            
            DoodadInstances.Add(instance);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var instance in DoodadInstances)
        {
            writer.Write(instance.NameId);
            writer.Write(instance.UniqueId);
            
            writer.Write(instance.Position.X);
            writer.Write(instance.Position.Y);
            writer.Write(instance.Position.Z);
            
            writer.Write(instance.Rotation.X);
            writer.Write(instance.Rotation.Y);
            writer.Write(instance.Rotation.Z);
            
            writer.Write(instance.Scale);
        }
    }
    
    // Helper to add a new doodad instance
    public void AddDoodadInstance(SMDoodadDef instance)
    {
        DoodadInstances.Add(instance);
    }
    
    // Helper to get all instances that reference a specific nameId
    public List<SMDoodadDef> GetInstancesByNameId(uint nameId)
    {
        return DoodadInstances.Where(i => i.NameId == nameId).ToList();
    }
}
```

## M2 Positioning System
The coordinate system for M2 model placement works as follows:

1. Position (X, Y, Z): World space coordinates for the origin of the M2 model
   - X: East-West axis (positive eastward)
   - Y: North-South axis (positive northward)
   - Z: Elevation (positive upward)

2. Rotation (X, Y, Z): Rotation angles in radians
   - X: Pitch (rotation around X-axis)
   - Y: Yaw (rotation around Y-axis)
   - Z: Roll (rotation around Z-axis)
   - Rotation is applied in the order of Y, Z, X (yaw, roll, pitch)

3. Scale: Uniform scale factor
   - Unlike WMO objects (in MODF) which can have non-uniform scaling
   - A value of 1.0 represents the default size of the model
   - M2 models are typically scaled uniformly to maintain their proportions

## Version Information
- Present in later versions of WDT files (version 18+) if global doodad objects are included
- The structure remains consistent across all WDT versions that include it
- In version 22+, the references may be replaced with FileDataIDs in a similar manner to MODF

## Usage Context
The MDDF chunk provides placement data for global M2 doodad objects in the world:

- Defines how doodad models are positioned in the world
- Controls scale, rotation, and positioning of M2 models
- Global doodads are loaded independently of ADT tiles (unlike ADT-specific doodads)

Global doodads differ from ADT-specific doodads in several important ways:
1. They are loaded regardless of which ADT tile the player is in
2. They can span multiple ADT tiles if needed
3. They are defined at the world level rather than the tile level
4. They are often used for important objects that need to be visible from a distance

The MDDF chunk works together with the MDNM and MDID chunks to define the complete doodad placement system:
1. MDNM stores the unique M2 filenames
2. MDID provides efficient access to these filenames
3. MDDF contains the actual placement data for each doodad instance

This pattern mirrors the WMO placement system (MWMO/MWID/MODF), but for M2 models instead of WMO objects. 