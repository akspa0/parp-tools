# MOGP - WMO Group Header

## Type
WMO Group Chunk (Container)

## Source
WMO.md

## Description
The MOGP chunk is the main container chunk for WMO group files. It contains a header with important information about the group, including flags, bounding box, and references to other chunks, followed by all other chunks in the group file. This chunk effectively functions like a file header for the group file, with the following chunks being contained within it.

## Structure

```csharp
public struct MOGP
{
    // Header Structure
    public uint GroupNameOffset;        // Offset into MOGN in the root file
    public uint DescriptiveGroupNameOffset; // Offset into MOGN in the root file
    public uint Flags;                  // Group flags (see below)
    public CAaBox BoundingBox;          // Bounding box for the group
    public ushort PortalStart;          // Starting index into the MOPR chunk in the root file
    public ushort PortalCount;          // Number of portals used by this group
    public ushort TransBatchCount;      // Number of transparent batches
    public ushort IntBatchCount;        // Number of interior batches
    public ushort ExtBatchCount;        // Number of exterior batches
    public ushort BatchTypeD;           // Likely padding
    public byte[] FogIndices;           // 4 bytes of fog indices referencing MFOG
    public uint GroupLiquid;            // Liquid type information
    public uint UniqueId;               // Group ID for WMOAreaTable
    public uint Flags2;                 // Secondary flags
    public short ParentOrFirstChildSplitGroupIndex; // For split groups
    public short NextSplitChildGroupIndex;  // For split groups

    // Remaining chunks follow this header
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | GroupNameOffset | uint | Offset into the MOGN chunk in the root file |
| 0x04 | DescriptiveGroupNameOffset | uint | Offset into the MOGN chunk in the root file |
| 0x08 | Flags | uint | Group flags (see below) |
| 0x0C | BoundingBox | CAaBox | Axis-aligned bounding box for the group (24 bytes) |
| 0x24 | PortalStart | ushort | Starting index into the MOPR chunk in the root file |
| 0x26 | PortalCount | ushort | Number of portals used by this group |
| 0x28 | TransBatchCount | ushort | Number of transparent render batches |
| 0x2A | IntBatchCount | ushort | Number of interior render batches |
| 0x2C | ExtBatchCount | ushort | Number of exterior render batches |
| 0x2E | BatchTypeD | ushort | Likely padding or unused data |
| 0x30 | FogIndices | byte[4] | Four indices into the MFOG chunk |
| 0x34 | GroupLiquid | uint | Liquid type information for the group |
| 0x38 | UniqueId | uint | Unique ID used in WMOAreaTable |
| 0x3C | Flags2 | uint | Secondary flags (see below) |
| 0x40 | ParentOrFirstChildSplitGroupIndex | short | Index of parent or first child for split groups (-1 if not used) |
| 0x42 | NextSplitChildGroupIndex | short | Index of next child for split groups (-1 if not used) |

## Group Flags (Offset 0x08)

| Value | Flag | Description |
|-------|------|-------------|
| 0x00000001 | HasBSP | Has BSP tree (MOBN and MOBR chunks) |
| 0x00000004 | HasVertexColors | Has vertex colors (MOCV chunk) |
| 0x00000008 | Exterior | Outdoor group. Influences doodad culling |
| 0x00000040 | ExteriorLit | Use exterior lighting for this group |
| 0x00000080 | Unreachable | Group is unreachable/unused |
| 0x00000100 | ShowSky | Show exterior sky in interior WMO group |
| 0x00000200 | HasLights | Has lights (MOLR chunk) |
| 0x00000800 | HasDoodads | Has doodads (MODR chunk) |
| 0x00001000 | LiquidSurface | Has water (MLIQ chunk) |
| 0x00002000 | Interior | Indoor group |
| 0x00010000 | AlwaysDraw | Always render this group |
| 0x00040000 | ShowSkybox | Show skybox for this group |
| 0x00080000 | IsOcean | Is ocean rather than water (liquid-related) |
| 0x00200000 | MountAllowed | Mounting is allowed in this group |
| 0x01000000 | HasSecondVertexColors | Has the second MOCV chunk |
| 0x02000000 | HasTwoTextureCoords | Has two MOTV chunks |
| 0x04000000 | AntiPortal | Group functions as an occlusion portal |
| 0x40000000 | HasThreeTextureCoords | Has three MOTV chunks |

## Group Flags2 (Offset 0x3C)

| Value | Flag | Description |
|-------|------|-------------|
| 0x00000001 | CanCutTerrain | Has portal planes that can cut terrain |
| 0x00000040 | IsSplitGroupParent | Group is a parent of split groups |
| 0x00000080 | IsSplitGroupChild | Group is a child of a split group |
| 0x00000100 | IsAttachmentMesh | Group is an attachment mesh |

## Dependencies
- MOGI chunk in root file (flags and bounding box should match)
- MOGN chunk in root file (for group name)
- MOPR chunk in root file (for portal references)
- MFOG chunk in root file (for fog indices)

## Implementation Notes
- The MOGP chunk is a container for all other chunks in a WMO group file.
- The sum of TransBatchCount, IntBatchCount, and ExtBatchCount equals the total number of batches in the MOBA chunk.
- If the group is named "antiportal", it functions as an occlusion portal, and certain flags are automatically set.
- For liquid handling, if GroupLiquid is set but no MLIQ chunk is present (or xtiles/ytiles are 0), the entire group is filled with liquid at the maximum Z height of the bounding box.
- Split groups (introduced in version 9.2.0) use the ParentOrFirstChildSplitGroupIndex and NextSplitChildGroupIndex fields to create a linked list of related groups.

## Implementation Example

```csharp
public class MOGPChunk : IWmoGroupChunk
{
    public string ChunkId => "MOGP";
    
    // Header Properties
    public uint GroupNameOffset { get; set; }
    public uint DescriptiveGroupNameOffset { get; set; }
    public uint Flags { get; set; }
    public CAaBox BoundingBox { get; set; } = new CAaBox();
    public ushort PortalStart { get; set; }
    public ushort PortalCount { get; set; }
    public ushort TransBatchCount { get; set; }
    public ushort IntBatchCount { get; set; }
    public ushort ExtBatchCount { get; set; }
    public ushort BatchTypeD { get; set; }
    public byte[] FogIndices { get; set; } = new byte[4];
    public uint GroupLiquid { get; set; }
    public uint UniqueId { get; set; }
    public uint Flags2 { get; set; }
    public short ParentOrFirstChildSplitGroupIndex { get; set; } = -1;
    public short NextSplitChildGroupIndex { get; set; } = -1;
    
    // Collection of sub-chunks
    public Dictionary<string, IWmoGroupChunk> SubChunks { get; } = new Dictionary<string, IWmoGroupChunk>();
    
    public void Read(BinaryReader reader, uint size)
    {
        long startPosition = reader.BaseStream.Position;
        
        // Read header
        GroupNameOffset = reader.ReadUInt32();
        DescriptiveGroupNameOffset = reader.ReadUInt32();
        Flags = reader.ReadUInt32();
        BoundingBox.Read(reader);
        PortalStart = reader.ReadUInt16();
        PortalCount = reader.ReadUInt16();
        TransBatchCount = reader.ReadUInt16();
        IntBatchCount = reader.ReadUInt16();
        ExtBatchCount = reader.ReadUInt16();
        BatchTypeD = reader.ReadUInt16();
        FogIndices = reader.ReadBytes(4);
        GroupLiquid = reader.ReadUInt32();
        UniqueId = reader.ReadUInt32();
        Flags2 = reader.ReadUInt32();
        ParentOrFirstChildSplitGroupIndex = reader.ReadInt16();
        NextSplitChildGroupIndex = reader.ReadInt16();
        
        // Read remaining chunks until we reach the end of this chunk
        long headerSize = reader.BaseStream.Position - startPosition;
        long remainingSize = size - headerSize;
        
        while (remainingSize > 0 && reader.BaseStream.Position < startPosition + size)
        {
            string chunkId = new string(reader.ReadChars(4));
            uint chunkSize = reader.ReadUInt32();
            
            // Create and read appropriate chunk type
            IWmoGroupChunk chunk = ChunkFactory.CreateGroupChunk(chunkId);
            if (chunk != null)
            {
                chunk.Read(reader, chunkSize);
                SubChunks[chunkId] = chunk;
            }
            else
            {
                // Skip unknown chunk
                reader.BaseStream.Seek(chunkSize, SeekOrigin.Current);
            }
            
            remainingSize -= (8 + chunkSize); // 8 bytes for chunk header
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write chunk header (will need to come back and update size)
        long sizePosition = writer.BaseStream.Position + 4; // Position to write size
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        writer.Write((uint)0); // Placeholder for size
        
        long startPosition = writer.BaseStream.Position;
        
        // Write header data
        writer.Write(GroupNameOffset);
        writer.Write(DescriptiveGroupNameOffset);
        writer.Write(Flags);
        BoundingBox.Write(writer);
        writer.Write(PortalStart);
        writer.Write(PortalCount);
        writer.Write(TransBatchCount);
        writer.Write(IntBatchCount);
        writer.Write(ExtBatchCount);
        writer.Write(BatchTypeD);
        writer.Write(FogIndices);
        writer.Write(GroupLiquid);
        writer.Write(UniqueId);
        writer.Write(Flags2);
        writer.Write(ParentOrFirstChildSplitGroupIndex);
        writer.Write(NextSplitChildGroupIndex);
        
        // Write all subchunks
        foreach (var chunk in SubChunks.Values)
        {
            chunk.Write(writer);
        }
        
        // Go back and update the size
        long endPosition = writer.BaseStream.Position;
        writer.BaseStream.Position = sizePosition;
        writer.Write((uint)(endPosition - startPosition));
        writer.BaseStream.Position = endPosition;
    }
}
```

## Validation Requirements
- The sum of TransBatchCount, IntBatchCount, and ExtBatchCount should equal the total number of batches in the MOBA chunk.
- Flags in the MOGP chunk should be consistent with the corresponding MOGI entry in the root file.
- If AntiPortal flag is set, the IntBatchCount and ExtBatchCount should be 0.
- If IsSplitGroupParent flag is set, ParentOrFirstChildSplitGroupIndex should be valid and NextSplitChildGroupIndex should be -1.
- If IsSplitGroupChild flag is set, both ParentOrFirstChildSplitGroupIndex and NextSplitChildGroupIndex should be valid indices or -1.

## Usage Context
- **Group Structure Definition**: Defines the basic properties and structure of a WMO group.
- **Rendering Control**: Flags determine how the group is rendered (interior/exterior, lighting model, etc.).
- **Occlusion Culling**: Information used for occlusion and visibility determination.
- **Spatial Organization**: Bounding box and portal references are used for spatial queries and rendering optimization.
- **Split Group Management**: Parent/child indices control relationships between split groups for large or complex models. 