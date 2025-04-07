# L004: MODF

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MODF (Map Object Definition) chunk contains placement information for WMO (World Map Object) models in the low-resolution map. It defines where and how each WMO model is positioned and oriented in the world space. This chunk works in conjunction with the MWMO and MWID chunks to place models that are significant enough to be visible at a distance, such as major structures and landmarks.

## Structure
```csharp
struct MODF
{
    /*0x00*/ uint32_t nameId;        // Index into MWID array
    /*0x04*/ uint32_t uniqueId;      // Unique identifier for this instance
    /*0x08*/ C3Vector position;      // Position in world space (X, Y, Z)
    /*0x14*/ C3Vector rotation;      // Rotation angles (A, B, C) in radians
    /*0x20*/ CAaBox boundingBox;     // Axis-aligned bounding box (min XYZ, max XYZ)
    /*0x38*/ uint16_t flags;         // Various control flags
    /*0x3A*/ uint16_t doodadSet;     // Doodad set index
    /*0x3C*/ uint16_t nameSet;       // Name set index
    /*0x3E*/ uint16_t scale;         // Scale factor (usually 1024 = 100%)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nameId | uint32_t | Index into the MWID array, identifying which WMO file to use |
| uniqueId | uint32_t | Unique identifier for this WMO instance |
| position | C3Vector | XYZ coordinates in world space where the WMO is placed |
| rotation | C3Vector | Rotation angles around the XYZ axes in radians |
| boundingBox | CAaBox | Axis-aligned bounding box defining the WMO's extent |
| flags | uint16_t | Flags controlling the WMO's behavior and rendering |
| doodadSet | uint16_t | Index of the doodad set to use (references a set in the WMO file) |
| nameSet | uint16_t | Index of the name set to use (references a set in the WMO file) |
| scale | uint16_t | Scaling factor, where 1024 = 100% |

## Referenced Structures
```csharp
struct C3Vector
{
    /*0x00*/ float x;
    /*0x04*/ float y;
    /*0x08*/ float z;
};

struct CAaBox
{
    /*0x00*/ C3Vector min;  // Minimum XYZ coordinates
    /*0x0C*/ C3Vector max;  // Maximum XYZ coordinates
};
```

## Flag Values
| Value | Flag Name | Description |
|-------|-----------|-------------|
| 0x01 | WMO_DESTRUCTION | 1: WMO has destruction states |
| 0x02 | WMO_UNKNOWN | Unknown purpose |
| 0x04 | WMO_UNKNOWN2 | Unknown purpose |
| 0x08 | WMO_HAS_LIGHTS | 1: WMO contains light information |
| 0x10 | WMO_UNKNOWN3 | Unknown purpose |
| 0x20 | WMO_UNKNOWN4 | Unknown purpose |
| 0x40 | WMO_UNKNOWN5 | Unknown purpose |
| 0x80 | WMO_UNKNOWN6 | Unknown purpose |
| 0x100 | WMO_UNKNOWN7 | Unknown purpose |
| 0x200 | WMO_UNKNOWN8 | Unknown purpose |
| 0x400 | WMO_UNKNOWN9 | Unknown purpose |
| 0x800 | WMO_UNKNOWN10 | Unknown purpose |

## Dependencies
- MWMO (L002) - Contains the filename strings for WMO models
- MWID (L003) - Contains indices to the WMO filenames

## Implementation Notes
- Each MODF entry is 64 bytes (0x40) in size
- The number of entries can be determined by dividing the chunk size by 64
- The nameId field is an index into the MWID array, which contains offsets into the MWMO chunk
- The bounding box coordinates are in world space
- Rotation values are in radians and represent rotations around the X, Y, and Z axes
- The scale field is a fixed-point value where 1024 represents 100% scale
- In the WDL format, only major, significant WMOs are included (subset of those in WDT)
- WMOs in WDL are typically those visible from a great distance

## Implementation Example
```csharp
public class MODF : IChunk
{
    public class MODFEntry
    {
        public uint NameId { get; set; }          // Index into MWID array
        public uint UniqueId { get; set; }         // Unique identifier
        public Vector3 Position { get; set; }      // XYZ position
        public Vector3 Rotation { get; set; }      // XYZ rotation in radians
        public BoundingBox BoundingBox { get; set; } // Min/Max XYZ
        public ushort Flags { get; set; }          // Various flags
        public ushort DoodadSet { get; set; }      // Doodad set index
        public ushort NameSet { get; set; }        // Name set index
        public ushort Scale { get; set; }          // Scale factor (1024 = 100%)
        
        // Get the actual scale factor
        public float GetScaleFactor()
        {
            return Scale / 1024.0f;
        }
        
        // Get the WMO filename from MWMO using the nameId
        public string GetFilename(MWID mwid, MWMO mwmo)
        {
            if (NameId >= mwid.GetCount())
                return string.Empty;
                
            return mwmo.GetFilename((int)NameId);
        }
        
        // Helper to check if a specific flag is set
        public bool HasFlag(ushort flag)
        {
            return (Flags & flag) != 0;
        }
    }
    
    public List<MODFEntry> Entries { get; private set; } = new List<MODFEntry>();
    
    public void Parse(BinaryReader reader, long size)
    {
        int numEntries = (int)(size / 64); // Each entry is 64 bytes
        Entries.Clear();
        
        for (int i = 0; i < numEntries; i++)
        {
            MODFEntry entry = new MODFEntry();
            
            // Read nameId and uniqueId
            entry.NameId = reader.ReadUInt32();
            entry.UniqueId = reader.ReadUInt32();
            
            // Read position vector
            entry.Position = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            // Read rotation vector
            entry.Rotation = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            // Read bounding box
            Vector3 bbMin = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            Vector3 bbMax = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            entry.BoundingBox = new BoundingBox(bbMin, bbMax);
            
            // Read flags and indices
            entry.Flags = reader.ReadUInt16();
            entry.DoodadSet = reader.ReadUInt16();
            entry.NameSet = reader.ReadUInt16();
            entry.Scale = reader.ReadUInt16();
            
            Entries.Add(entry);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (MODFEntry entry in Entries)
        {
            // Write nameId and uniqueId
            writer.Write(entry.NameId);
            writer.Write(entry.UniqueId);
            
            // Write position vector
            writer.Write(entry.Position.X);
            writer.Write(entry.Position.Y);
            writer.Write(entry.Position.Z);
            
            // Write rotation vector
            writer.Write(entry.Rotation.X);
            writer.Write(entry.Rotation.Y);
            writer.Write(entry.Rotation.Z);
            
            // Write bounding box
            writer.Write(entry.BoundingBox.Min.X);
            writer.Write(entry.BoundingBox.Min.Y);
            writer.Write(entry.BoundingBox.Min.Z);
            writer.Write(entry.BoundingBox.Max.X);
            writer.Write(entry.BoundingBox.Max.Y);
            writer.Write(entry.BoundingBox.Max.Z);
            
            // Write flags and indices
            writer.Write(entry.Flags);
            writer.Write(entry.DoodadSet);
            writer.Write(entry.NameSet);
            writer.Write(entry.Scale);
        }
    }
    
    // Helper method to add a new WMO placement
    public void AddEntry(MODFEntry entry)
    {
        Entries.Add(entry);
    }
    
    // Helper method to find WMO entries by uniqueId
    public MODFEntry FindByUniqueId(uint uniqueId)
    {
        return Entries.FirstOrDefault(e => e.UniqueId == uniqueId);
    }
    
    // Helper method to get all WMO entries for a specific nameId
    public List<MODFEntry> GetEntriesByNameId(uint nameId)
    {
        return Entries.Where(e => e.NameId == nameId).ToList();
    }
}
```

## Relationship to WDT
The MODF chunk in WDL serves a similar purpose to its counterpart in WDT files, but with some key differences:

- WDL MODF contains a subset of the WMO placements in WDT MODF
- Only WMOs visible from a distance are included
- The structure and format remain the same across both formats
- The uniqueId values may be different between WDT and WDL formats

## Coordinate System
- X-axis: East (positive) to West (negative)
- Y-axis: North (positive) to South (negative)
- Z-axis: Up (positive) to Down (negative)
- Origin (0,0,0) is typically at the map's center

## Transformation Application Order
When transforming a WMO for placement in the world:
1. Apply scale transformation
2. Apply rotation transformations (in order: Z, Y, X)
3. Apply position translation

## Version Information
- The MODF chunk format remains consistent across different versions
- The chunk may be absent in maps with no distant WMO objects 