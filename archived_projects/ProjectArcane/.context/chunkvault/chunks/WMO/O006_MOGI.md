# O006: MOGI

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOGI (Map Object Group Information) chunk contains information about each group within the WMO, including flags, bounding boxes, and name references. This chunk serves as a directory of all the groups that make up the complete WMO, providing essential metadata for rendering and gameplay systems. Each MOGI entry corresponds to a separate WMO group file.

## Structure
```csharp
struct MOGI
{
    GroupInfo[] groupInfos;  // Array of group information entries
};

struct GroupInfo
{
    /*0x00*/ uint32_t flags;            // Group flags
    /*0x04*/ uint32_t nameOffset;       // Offset into MOGN chunk
    /*0x08*/ CAaBox boundingBox;        // Bounding box (24 bytes: min and max XYZ)
    /*0x20*/ int16_t portalStart;       // Starting portal index (or -1)
    /*0x22*/ int16_t portalCount;       // Number of portals for this group
    /*0x24*/ uint16_t batchCount[4];    // Number of batches per render pass (A, B, C, D)
    /*0x2C*/ uint8_t fogIndices[4];     // Fog indices for each render pass
    /*0x30*/ uint32_t liquidType;       // Liquid type (if present)
    /*0x34*/ uint32_t groupID;          // Unique identifier for the group (added in later versions)
    /*0x38*/ uint32_t unused1;          // Unused field
    /*0x3C*/ uint32_t unused2;          // Unused field
};
```

## Properties

### GroupInfo Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | flags | uint32_t | Group flags (see table below) |
| 0x04 | nameOffset | uint32_t | Offset to this group's name in the MOGN chunk |
| 0x08 | boundingBox | CAaBox | Bounding box for culling (24 bytes: min and max XYZ) |
| 0x20 | portalStart | int16_t | Index of first portal for this group, or -1 if none |
| 0x22 | portalCount | int16_t | Number of portals used by this group |
| 0x24 | batchCount | uint16_t[4] | Number of batches per render pass (A, B, C, D) |
| 0x2C | fogIndices | uint8_t[4] | Fog indices for each render pass |
| 0x30 | liquidType | uint32_t | Liquid type (if present, used for water/lava effects) |
| 0x34 | groupID | uint32_t | Unique identifier for the group (added in later versions) |
| 0x38 | unused1 | uint32_t | Unused field |
| 0x3C | unused2 | uint32_t | Unused field |

### Group Flags
| Flag Value | Name | Description |
|------------|------|-------------|
| 0x0001 | HAS_BM_GEOMETRY | Has BSP tree (collision) geometry |
| 0x0002 | USING_MOCV | Uses vertex colors (MOCV chunk) |
| 0x0004 | HAS_EXTERNAL_LIGHTING | Has custom lighting data |
| 0x0008 | HAS_LIGHT_MAP | Uses a light map |
| 0x0010 | HAS_WATER | Has water in this group |
| 0x0020 | INDOOR_GROUP | Interior group (for weather effects) |
| 0x0040 | UNUSED | Unused flag |
| 0x0080 | SHOW_SKYBOX | Always show skybox for this group |
| 0x0100 | MOUNT_ALLOWED | Mounting is allowed in this group |
| 0x0200 | EXTERIOR_LIT | Use exterior lighting model |
| 0x1000 | HAS_THREE_VERSIONS | Has low/medium/high quality versions (LODs) |
| 0x2000 | FROZEN | Has exterior lighting even when below ground |
| 0x4000 | HAS_DAYNIGHT_LIGHTING | Has day/night transition lighting |
| 0x8000 | HAS_SKYBOX | Has skybox defined in this group |
| 0x10000 | USES_MSL | Uses model-space lighting (MSL) instead of world-space |
| 0x20000 | LAVA_EFFECTS | Has lava effects |
| 0x40000 | UNDERWATER | Is underwater interior |
| 0x80000 | USE_FOG | Use fog when rendering |
| 0x100000 | GLOW_EFFECT | Has glow effect |
| 0x200000 | LIQUID_OCEAN | Liquid is ocean water |
| 0x400000 | ANTIPORTAL | Group is an antiportal |
| 0x800000 | AMBIENT_SKY_LIGHT_OVERRIDE | Override ambient sky light amount |
| 0x1000000 | EXTERIOR_VERTS_ONLY | Only contains exterior vertices |

### Render Passes
| Index | Name | Description |
|-------|------|-------------|
| 0 | Render Pass A | Opaque geometry (no transparency) |
| 1 | Render Pass B | Alpha-tested geometry (cutouts) |
| 2 | Render Pass C | Blended transparency (sorted) |
| 3 | Render Pass D | Additional transparent geometry (water, etc.) |

## Dependencies
- MOHD: The nGroups field indicates how many group info entries should be present
- MOGN: Contains the group names referenced by nameOffset
- MOPT/MOPV: Portal data referenced by portalStart/portalCount

## Implementation Notes
- Each entry corresponds to a separate WMO group file (filename.wmo_###.wmo where ### is the group index)
- The number of entries should match the nGroups field in the MOHD chunk
- The boundingBox is used for culling - if the box is outside the view frustum, the entire group can be skipped
- The portalStart and portalCount fields are used for visibility determination between connected groups
- A portalStart value of -1 indicates no portals for this group
- The batchCount array indicates how many batches of each render pass type are in the group
- Fog indices reference fog settings for different render passes (values 0-4)
- The liquidType field is only relevant if the HAS_WATER flag is set
- Group IDs were added in later versions and may not be present in older files
- The effective size of this structure can vary between versions

## Implementation Example
```csharp
public class MOGI : IChunk
{
    public List<GroupInfo> Groups { get; private set; }
    
    public MOGI()
    {
        Groups = new List<GroupInfo>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many group entries we expect
        int groupInfoSize = 0x40; // Default size is 64 bytes
        int groupCount = (int)(size / groupInfoSize);
        
        Groups.Clear();
        
        for (int i = 0; i < groupCount; i++)
        {
            GroupInfo groupInfo = new GroupInfo();
            
            groupInfo.Flags = reader.ReadUInt32();
            groupInfo.NameOffset = reader.ReadUInt32();
            
            // Read bounding box
            groupInfo.BoundingBox = new BoundingBox();
            groupInfo.BoundingBox.Min = new Vector3(
                reader.ReadSingle(),    // X min
                reader.ReadSingle(),    // Y min
                reader.ReadSingle()     // Z min
            );
            groupInfo.BoundingBox.Max = new Vector3(
                reader.ReadSingle(),    // X max
                reader.ReadSingle(),    // Y max
                reader.ReadSingle()     // Z max
            );
            
            groupInfo.PortalStart = reader.ReadInt16();
            groupInfo.PortalCount = reader.ReadInt16();
            
            // Read batch counts for each render pass
            groupInfo.BatchCounts = new ushort[4];
            for (int j = 0; j < 4; j++)
            {
                groupInfo.BatchCounts[j] = reader.ReadUInt16();
            }
            
            // Read fog indices
            groupInfo.FogIndices = new byte[4];
            for (int j = 0; j < 4; j++)
            {
                groupInfo.FogIndices[j] = reader.ReadByte();
            }
            
            groupInfo.LiquidType = reader.ReadUInt32();
            
            // Check if we have enough data for the additional fields
            long currentPosition = reader.BaseStream.Position;
            long bytesRemaining = size - (currentPosition - (reader.BaseStream.Position - groupInfoSize * i));
            
            if (bytesRemaining >= 4)
            {
                groupInfo.GroupID = reader.ReadUInt32();
                
                if (bytesRemaining >= 12)
                {
                    groupInfo.Unused1 = reader.ReadUInt32();
                    groupInfo.Unused2 = reader.ReadUInt32();
                }
            }
            
            Groups.Add(groupInfo);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Determine if we need to write extended format
        bool extendedFormat = false;
        foreach (GroupInfo group in Groups)
        {
            if (group.GroupID != 0 || group.Unused1 != 0 || group.Unused2 != 0)
            {
                extendedFormat = true;
                break;
            }
        }
        
        foreach (GroupInfo group in Groups)
        {
            writer.Write(group.Flags);
            writer.Write(group.NameOffset);
            
            // Write bounding box
            writer.Write(group.BoundingBox.Min.X);
            writer.Write(group.BoundingBox.Min.Y);
            writer.Write(group.BoundingBox.Min.Z);
            writer.Write(group.BoundingBox.Max.X);
            writer.Write(group.BoundingBox.Max.Y);
            writer.Write(group.BoundingBox.Max.Z);
            
            writer.Write(group.PortalStart);
            writer.Write(group.PortalCount);
            
            // Write batch counts
            for (int i = 0; i < 4; i++)
            {
                writer.Write(group.BatchCounts[i]);
            }
            
            // Write fog indices
            for (int i = 0; i < 4; i++)
            {
                writer.Write(group.FogIndices[i]);
            }
            
            writer.Write(group.LiquidType);
            
            // Write extended format fields if needed
            if (extendedFormat)
            {
                writer.Write(group.GroupID);
                writer.Write(group.Unused1);
                writer.Write(group.Unused2);
            }
        }
    }
}

public class GroupInfo
{
    public uint Flags { get; set; }
    public uint NameOffset { get; set; }
    public BoundingBox BoundingBox { get; set; }
    public short PortalStart { get; set; }
    public short PortalCount { get; set; }
    public ushort[] BatchCounts { get; set; }
    public byte[] FogIndices { get; set; }
    public uint LiquidType { get; set; }
    public uint GroupID { get; set; }
    public uint Unused1 { get; set; }
    public uint Unused2 { get; set; }
    
    // Flag helper properties
    public bool HasBSPGeometry => (Flags & 0x0001) != 0;
    public bool HasVertexColors => (Flags & 0x0002) != 0;
    public bool HasExternalLighting => (Flags & 0x0004) != 0;
    public bool HasLightMap => (Flags & 0x0008) != 0;
    public bool HasWater => (Flags & 0x0010) != 0;
    public bool IsIndoor => (Flags & 0x0020) != 0;
    // ... additional flag helpers ...
    
    public GroupInfo()
    {
        // Initialize with defaults
        Flags = 0;
        NameOffset = 0;
        BoundingBox = new BoundingBox();
        PortalStart = -1;
        PortalCount = 0;
        BatchCounts = new ushort[4] { 0, 0, 0, 0 };
        FogIndices = new byte[4] { 0, 0, 0, 0 };
        LiquidType = 0;
        GroupID = 0;
        Unused1 = 0;
        Unused2 = 0;
    }
}
```

## Validation Requirements
- The number of group info entries should match the nGroups field in the MOHD chunk
- The nameOffset values should be valid offsets within the MOGN chunk
- The boundingBox min values should be less than the corresponding max values
- If portalStart is not -1, it should be a valid index into the portal array
- The combined portalStart and portalCount should not exceed the total number of portals
- Batch counts should be consistent with the actual number of batches in the group files

## Usage Context
The MOGI chunk is central to WMO processing and serves several critical functions:

1. **Culling and Visibility**: The bounding boxes enable efficient culling of groups outside the view frustum
2. **Portal Connectivity**: The portal references establish relationships between connected groups
3. **Rendering Configuration**: Flags and batch counts determine how each group is rendered
4. **Gameplay Attributes**: Flags indicate properties like indoor/outdoor status and whether mounting is allowed
5. **Group Identification**: Name offsets allow groups to be identified by human-readable names
6. **Environmental Effects**: Liquid types and fog indices control environmental effects within groups

The client uses this information to:
- Determine which groups to load and render based on player position
- Apply appropriate environmental effects (weather, lighting, water)
- Configure the rendering pipeline for different material types (transparent, opaque)
- Establish portal-based visibility relationships between connected groups
- Apply gameplay rules based on group properties (mounting, flying, etc.)

Each entry in this chunk corresponds to a separate group file, and together they define the complete structure of the WMO. 