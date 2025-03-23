# C011: MODF

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
WMO Placement Information - contains data about positioning of WMO (World Map Object) models in the ADT tile. Each entry references a filename in the MWMO/MWID chunks.

## Structure
```csharp
struct SMMapObjDef
{
    /*0x00*/ uint32_t nameId;        // References an entry in the MWID chunk specifying which WMO model to use
    /*0x04*/ uint32_t uniqueId;      // This ID should be unique for all ADTs currently loaded
    /*0x08*/ C3Vector position;      // Position in world coordinates
    /*0x14*/ C3Vector rotation;      // Rotation in degrees
    /*0x20*/ CAaBox extents;         // Bounding box dimensions (min/max corners)
    /*0x38*/ uint16_t flags;         // Placement flags
    /*0x3A*/ uint16_t doodadSet;     // Doodad set index
    /*0x3C*/ uint16_t nameSet;       // Name set index
    /*0x3E*/ uint16_t padding;       // Padding to ensure 4-byte alignment
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nameId | uint32 | References an entry in the MWID chunk, specifying which WMO to use |
| uniqueId | uint32 | Unique identifier for this WMO instance (should be unique across all loaded ADTs) |
| position | C3Vector | Position of the WMO in world coordinates |
| rotation | C3Vector | Rotation angles in degrees (pitch, yaw, roll) |
| extents | CAaBox | Axis-aligned bounding box (includes 6 floats: 3 for min corner, 3 for max corner) |
| flags | uint16 | Various flags controlling the WMO's behavior |
| doodadSet | uint16 | Index of the doodad set to use for this WMO |
| nameSet | uint16 | Index of the name set to use for this WMO |
| padding | uint16 | Padding bytes for 4-byte alignment |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | WMO_Flag_Destroyed | Destroyed version of the WMO should be shown |
| 0x2 | WMO_Flag_UseServerLighting | Use server-controlled lighting instead of WMO internal lighting |
| 0x8 | WMO_Flag_PlaceHolderOnly | Only a placeholder (not a real WMO) |
| 0x40 | WMO_Flag_LiquidQuad | Indicates if this is a water-type WMO |

## Dependencies
- MWMO (C007) - Contains the WMO filenames
- MWID (C008) - Contains indices into the MWMO chunk

## Implementation Notes
- Position coordinates are global world coordinates
- Rotation values are in degrees, not radians
- The bounding box is pre-calculated and used for culling operations
- WMO instances may have multiple doodad sets, only one of which is active at a time
- The uniqueId must be maintained when copying WMOs between ADT tiles

## Coordinate System Translation
To correctly position a WMO in the world, its local coordinate system must be transformed to the world coordinate system:

```csharp
// Example transformation code in C#
private Matrix4x4 CreatePlacementMatrix(MODF modf)
{
    // Create rotation matrices for each axis
    Matrix4x4 rotX = Matrix4x4.CreateRotationX(MathHelper.ToRadians(modf.Rotation.X));
    Matrix4x4 rotY = Matrix4x4.CreateRotationY(MathHelper.ToRadians(modf.Rotation.Y - 270.0f));
    Matrix4x4 rotZ = Matrix4x4.CreateRotationZ(MathHelper.ToRadians(-modf.Rotation.Z));
    
    // Create translation matrix
    Matrix4x4 trans = Matrix4x4.CreateTranslation(
        modf.Position.X,
        modf.Position.Y,
        modf.Position.Z);
    
    // Combine transformations (order matters!)
    Matrix4x4 result = rotX * rotZ * rotY * trans;
    
    return result;
}
```

## Implementation Example
```csharp
public class MODF : IChunk
{
    public List<WMOPlacement> WMOPlacements { get; private set; } = new List<WMOPlacement>();
    
    public MODF(BinaryReader reader, uint size)
    {
        int count = (int)(size / 64); // Each entry is 64 bytes
        
        for (int i = 0; i < count; i++)
        {
            WMOPlacements.Add(new WMOPlacement(reader));
        }
    }
    
    public class WMOPlacement
    {
        public uint NameId { get; private set; }
        public uint UniqueId { get; private set; }
        public Vector3 Position { get; private set; }
        public Vector3 Rotation { get; private set; }
        public BoundingBox Extents { get; private set; }
        public ushort Flags { get; private set; }
        public ushort DoodadSet { get; private set; }
        public ushort NameSet { get; private set; }
        
        public WMOPlacement(BinaryReader reader)
        {
            NameId = reader.ReadUInt32();
            UniqueId = reader.ReadUInt32();
            
            // Read position
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            Position = new Vector3(x, y, z);
            
            // Read rotation (in degrees)
            float rotX = reader.ReadSingle();
            float rotY = reader.ReadSingle();
            float rotZ = reader.ReadSingle();
            Rotation = new Vector3(rotX, rotY, rotZ);
            
            // Read bounding box
            Vector3 min = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            Vector3 max = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            Extents = new BoundingBox(min, max);
            
            // Read flags and set indices
            Flags = reader.ReadUInt16();
            DoodadSet = reader.ReadUInt16();
            NameSet = reader.ReadUInt16();
            
            // Skip padding
            reader.ReadUInt16();
        }
        
        public bool IsDestroyed => (Flags & 0x1) != 0;
        public bool UsesServerLighting => (Flags & 0x2) != 0;
        public bool IsPlaceholder => (Flags & 0x8) != 0;
        public bool IsLiquidQuad => (Flags & 0x40) != 0;
    }
}
```

## Usage Context
The MODF chunk is used to place WMO models in the game world. These models represent complex structures like buildings, caves, bridges, and other large environmental objects. Each MODF entry:

1. References a WMO model from the MWMO chunk
2. Positions and orients it in the world
3. Specifies which doodad set to use (controlling which decorative objects appear)
4. Provides a bounding box for culling operations

WMOs are more complex than M2 models (placed via MDDF chunk) and can include their own internal lighting, doodad sets, and portals. 