# O010: MODD

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MODD (Map Object DoodaD definitions) chunk contains placement information for all doodads (M2 models) used within the WMO. Each entry defines the position, orientation, scale, and other properties of a specific doodad instance. These doodads add visual detail and variety to the WMO environment without requiring these details to be built into the main model structure.

## Structure
```csharp
struct MODD
{
    SMODoodadDef[] doodadDefs;  // Array of doodad placement definitions
};

struct SMODoodadDef
{
    /*0x00*/ uint32_t nameOffset;      // Offset into MODN chunk
    /*0x04*/ uint32_t flags;           // Placement flags
    /*0x08*/ C3Vector position;        // Position coordinates
    /*0x14*/ C3Vector rotation;        // Rotation coordinates (radians)
    /*0x20*/ float scale;              // Uniform scaling factor
    /*0x24*/ uint32_t color;           // Color multiplier (BGRA)
    /*0x28*/ uint32_t state;           // State information (often for opening/closing)
    /*0x2C*/ uint32_t set;             // Which MODS doodad set this belongs to
    /*0x30*/ uint16_t groupId;         // Original Group ID (MOP+ only) 
};
```

## Properties

### SMODoodadDef Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | nameOffset | uint32_t | Offset to the doodad's filename in the MODN chunk |
| 0x04 | flags | uint32_t | Placement flags (see table below) |
| 0x08 | position | C3Vector | Position coordinates in the WMO's coordinate space |
| 0x14 | rotation | C3Vector | Rotation angles in radians (X, Y, Z) |
| 0x20 | scale | float | Uniform scale factor applied to the doodad |
| 0x24 | color | uint32_t | Color multiplier in BGRA format |
| 0x28 | state | uint32_t | State information for animating doodads |
| 0x2C | set | uint32_t | Doodad set index this doodad belongs to |
| 0x30 | groupId | uint16_t | Group ID that this doodad is primarily associated with (MOP+ only) |

### Doodad Flags
| Flag Value | Name | Description |
|------------|------|-------------|
| 0x0001 | NO_COLLISION | Disables collision with this doodad |
| 0x0002 | LOCKED_DOOR | Doodad represents a locked door |
| 0x0004 | TILT_X | Tilt doodad along local X axis |
| 0x0008 | TILT_Y | Tilt doodad along local Y axis |
| 0x0010 | TILT_Z | Tilt doodad along local Z axis |
| 0x0020 | ALT_STATE | Use alternate state (enables different animation) |
| 0x0040 | SCALE_BY_DISTANCE | Scale visibility by distance (fade out when far) |
| 0x0080 | REDUCED_LOD | Use reduced level of detail model |
| 0x0100 | IGNORE_LIGHTING | Ignore external lighting effects |
| 0x0200 | DRAW_FIRST | Draw before other doodads |
| 0x0400 | DISABLE_SPECIAL_LIGHTING | Disables special lighting calculations |
| 0x0800 | ONLY_IN_BOUNDARY | Only display in WMO boundary |
| 0x1000 | SKYBOX_DOODAD | Part of skybox (only visible when skybox is visible) |
| 0x2000 | WATER_VISIBLE_ONLY | Only visible from under water |
| 0x4000 | WATER_INVISIBLE_ONLY | Invisible from under water |
| 0x8000 | SPECIAL_MO_ONLY | Special object that must be rendered separately |
| 0x10000 | NOT_IN_BOUNDARY | Object must be rendered outside of WMO boundary |
| 0x20000 | CULLED_WHEN_NOT_VISIBLE | Aggressive culling when not in view |
| 0x40000 | SHOW_IF_VIEWER_IN_SET | Only visible if the viewer is in the same set |
| 0x80000 | UNUSED_2 | Unused flag |
| 0x100000 | CAN_BE_STAGGERED | Can use staggered animation start times |
| 0x200000 | UNUSED_3 | Unused flag |
| 0x400000 | UNUSED_4 | Unused flag |
| 0x800000 | ANIM_DRIVEN_1 | Use animation-driven motion |
| 0x1000000 | ANIM_DRIVEN_2 | Animation-driven flag - alternate |
| 0x2000000 | CULL_IF_CLIENT_SEES | Cull if client sees through portal (portal optimization) |
| 0x4000000 | SCALE_IN_CUSTOM_MAP | Scale when in custom (instanced) map |
| 0x8000000 | DISABLE_CHARACTER_STAGGER | Disable character stagger when colliding with this |
| 0x10000000 | SHOW_ANIM_DEPENDENT | Only show when certain animation frames are active |
| 0x20000000 | UNUSED_5 | Unused flag |
| 0x40000000 | IGNORE_FOG | Ignore fog effects |
| 0x80000000 | INVERSE_SHADOWMASK | Invert shadow masking (shadow casting) |

## Dependencies
- MOHD: The nDoodadDefs field indicates how many doodad definitions should be present
- MODN: Contains the filenames referenced by nameOffset
- MODS: Contains definitions of doodad sets referenced by the set field

## Implementation Notes
- Each doodad definition is 48 bytes (0x30) or 52 bytes (0x34) in later versions
- The nameOffset field points to the start of a null-terminated filename in the MODN chunk
- The position coordinates are in the WMO's local coordinate system
- Rotation values are in radians, not degrees
- The color multiplier is applied to the doodad's texture colors (tints the model)
- The scale value is a uniform scale applied to the entire model
- The state field is used for doodads with multiple states (e.g., open/closed doors)
- The set field determines which doodad set this doodad belongs to (defined in MODS)
- The groupId field was added in Mists of Pandaria (MoP) and may not be present in older files
- The effective size of this structure can vary between WoW versions

## Implementation Example
```csharp
public class MODD : IChunk
{
    public List<DoodadDefinition> Doodads { get; private set; }
    
    public MODD()
    {
        Doodads = new List<DoodadDefinition>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many doodad entries we expect
        int structSize = 0x30; // Default size is 48 bytes
        
        // Check if we have the extended doodad format (added in MoP)
        bool extendedFormat = (size % structSize != 0) && (size % 0x34 == 0);
        if (extendedFormat)
        {
            structSize = 0x34; // Extended size is 52 bytes
        }
        
        int doodadCount = (int)(size / structSize);
        
        Doodads.Clear();
        
        for (int i = 0; i < doodadCount; i++)
        {
            DoodadDefinition doodad = new DoodadDefinition();
            
            doodad.NameOffset = reader.ReadUInt32();
            doodad.Flags = reader.ReadUInt32();
            
            // Read position vector
            doodad.Position = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            // Read rotation vector
            doodad.Rotation = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            doodad.Scale = reader.ReadSingle();
            doodad.Color = reader.ReadUInt32();
            doodad.State = reader.ReadUInt32();
            doodad.Set = reader.ReadUInt32();
            
            // Read the group ID if we have the extended format
            if (extendedFormat)
            {
                doodad.GroupId = reader.ReadUInt16();
                
                // Skip the padding (2 bytes)
                reader.ReadUInt16();
            }
            else
            {
                doodad.GroupId = 0; // Default value for older formats
            }
            
            Doodads.Add(doodad);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Check if we need to use the extended format
        bool extendedFormat = false;
        foreach (DoodadDefinition doodad in Doodads)
        {
            if (doodad.GroupId > 0)
            {
                extendedFormat = true;
                break;
            }
        }
        
        foreach (DoodadDefinition doodad in Doodads)
        {
            writer.Write(doodad.NameOffset);
            writer.Write(doodad.Flags);
            
            // Write position vector
            writer.Write(doodad.Position.X);
            writer.Write(doodad.Position.Y);
            writer.Write(doodad.Position.Z);
            
            // Write rotation vector
            writer.Write(doodad.Rotation.X);
            writer.Write(doodad.Rotation.Y);
            writer.Write(doodad.Rotation.Z);
            
            writer.Write(doodad.Scale);
            writer.Write(doodad.Color);
            writer.Write(doodad.State);
            writer.Write(doodad.Set);
            
            // Write the group ID and padding if using extended format
            if (extendedFormat)
            {
                writer.Write(doodad.GroupId);
                
                // Add padding
                writer.Write((ushort)0);
            }
        }
    }
    
    public DoodadDefinition GetDoodad(int index)
    {
        if (index >= 0 && index < Doodads.Count)
        {
            return Doodads[index];
        }
        
        throw new IndexOutOfRangeException($"Doodad index {index} is out of range. Valid range: 0-{Doodads.Count - 1}");
    }
    
    public List<DoodadDefinition> GetDoodadsBySet(uint setIndex)
    {
        return Doodads.Where(d => d.Set == setIndex).ToList();
    }
}

public class DoodadDefinition
{
    public uint NameOffset { get; set; }
    public uint Flags { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; }
    public uint Color { get; set; }
    public uint State { get; set; }
    public uint Set { get; set; }
    public ushort GroupId { get; set; }
    
    // Helper properties for flag checks
    public bool HasNoCollision => (Flags & 0x0001) != 0;
    public bool IsLockedDoor => (Flags & 0x0002) != 0;
    // ... additional flag helpers ...
    
    // Helper for color conversion
    public Color GetColor()
    {
        return new Color(
            (byte)((Color >> 16) & 0xFF),  // R
            (byte)((Color >> 8) & 0xFF),   // G
            (byte)(Color & 0xFF),          // B
            (byte)((Color >> 24) & 0xFF)   // A
        );
    }
    
    public void SetColor(Color color)
    {
        Color = (uint)(
            (color.B) |
            (color.G << 8) |
            (color.R << 16) |
            (color.A << 24)
        );
    }
    
    public DoodadDefinition()
    {
        // Initialize with defaults
        NameOffset = 0;
        Flags = 0;
        Position = new Vector3(0, 0, 0);
        Rotation = new Vector3(0, 0, 0);
        Scale = 1.0f;
        Color = 0xFFFFFFFF; // White, fully opaque
        State = 0;
        Set = 0;
        GroupId = 0;
    }
}
```

## Validation Requirements
- The number of doodad definitions should match the nDoodadDefs field in the MOHD chunk
- The nameOffset values should be valid offsets within the MODN chunk
- The set values should be valid indices into the doodad set definitions in MODS
- The scale value should be positive and reasonable (typically between 0.1 and 10)
- Color values should be valid BGRA format
- Position coordinates should be within the overall bounding box of the WMO
- If using the extended format, all entries should have the same structure size

## Usage Context
The MODD chunk is essential for populating WMOs with detailed objects:

1. **Environmental Enhancement**: Doodads add visual richness and detail to WMO environments
2. **Gameplay Interaction**: Doodad flags control how players interact with objects (collision, doors)
3. **Optimization**: Allows reuse of the same models in different locations and orientations
4. **Visual Variety**: Different doodad sets can be shown or hidden to create variation
5. **Dynamic Elements**: State values allow for animated or interactive doodads

When rendering a WMO with doodads:
1. The client determines which doodad set(s) should be active
2. For each active doodad in the set, it loads the M2 model referenced by nameOffset
3. The model is positioned, rotated, and scaled according to the transform values
4. The color tint is applied to the model's textures
5. The model is rendered with the appropriate state and flag settings

Doodads can represent a wide variety of objects:
- Furniture and decorative items
- Light sources (torches, lamps)
- Doors and windows (potentially interactive)
- Signs and banners
- Flora (potted plants, hanging vines)
- Small architectural elements (columns, statues)

The set system allows for variations of the WMO (for example, showing different seasonal decorations or damage states) without requiring multiple complete WMO versions. 