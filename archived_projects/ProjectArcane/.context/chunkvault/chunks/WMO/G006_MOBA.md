# MOBA - WMO Group Render Batches

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOBA chunk contains render batch information for efficient rendering of the WMO group. Render batches are groups of triangles that share the same material and can be rendered together, which optimizes the rendering process by minimizing state changes and draw calls. Each batch provides information about the range of vertices and indices that it uses, creating a subset of the geometry that can be rendered in a single call.

## Structure

```csharp
public struct MOBA
{
    public SMOBatch[] batchList; // Array of render batches
}

public struct SMOBatch
{
    // For versions prior to Shadowlands (pre-expansion level 7)
    public short bx, by, bz;          // Min point of bounding box for culling
    public short tx, ty, tz;          // Max point of bounding box for culling
    
    // For Shadowlands+ (expansion level 7+)
    // First 10 bytes contain different data, with material_id_large at offset 0x0A
    public ushort material_id_large;  // Used if flag_use_uint16_t_material is set
    
    public uint startIndex;           // Index of the first face index used in MOVI
    public ushort count;              // Number of MOVI indices used
    public ushort minIndex;           // Index of the first vertex used in MOVT
    public ushort maxIndex;           // Index of the last vertex used (inclusive)
    public byte flags;                // Flags for this batch
    public byte material_id;          // Index into MOMT chunk in the root file
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | bx, by, bz | short[3] | Low-resolution bounding box minimum point used for batch-level culling (pre-Shadowlands) |
| 0x06 | tx, ty, tz | short[3] | Low-resolution bounding box maximum point used for batch-level culling (pre-Shadowlands) |
| 0x0A | material_id_large | ushort | Extended material ID used in Shadowlands+ when more than 255 materials are needed |
| 0x0C | startIndex | uint | Starting index into the MOVI chunk's array of indices |
| 0x10 | count | ushort | Number of indices to use from MOVI, starting at startIndex |
| 0x12 | minIndex | ushort | Index of the first vertex used from the MOVT array |
| 0x14 | maxIndex | ushort | Index of the last vertex used from the MOVT array (inclusive) |
| 0x16 | flags | byte | Bit flags for the batch (see below) |
| 0x17 | material_id | byte | Index into the MOMT chunk in the root file, specifying the material for this batch |

## Batch Flags (Offset 0x16)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | flag_unknown_1 | Unknown usage |
| 1 | flag_use_material_id_large | When set, use material_id_large instead of material_id (Shadowlands+) |

## Dependencies
- **MOVI**: Contains the triangle indices referenced by startIndex and count
- **MOVT**: Contains the vertices referenced by minIndex and maxIndex
- **MOMT**: In the root file, contains the material definitions referenced by material_id or material_id_large

## Implementation Notes
- Each batch is 24 bytes in size.
- The bounding box (bx, by, bz, tx, ty, tz) in pre-Shadowlands versions is used for batch-level culling. If set incorrectly, batches may disappear when they should be visible.
- In Shadowlands and later, the bounding box fields are repurposed, with material_id_large at offset 0x0A.
- The vertex buffer range from minIndex to maxIndex may contain vertices that aren't used by the batch, but all needed vertices must be within this range.
- When "retroporting" files from Shadowlands to earlier versions, the bounding box values need to be calculated from the vertices in the range [minIndex, maxIndex].
- In Shadowlands, having collision-only invisible triangles between minIndex and maxIndex can cause rendering glitches on machines that support bindless texturing.
- The rendering approach differs between Direct3D and OpenGL in the game client, with Direct3D making fuller use of the batch information.
- Batches are processed in the order they appear in the file, which can affect rendering for transparent materials.

## Implementation Example

```csharp
public class MOBAChunk : IWmoGroupChunk
{
    public string ChunkId => "MOBA";
    public List<RenderBatch> Batches { get; private set; } = new List<RenderBatch>();
    
    // Track the WoW version we're parsing for
    private bool isPreShadowlands = true;

    public void Parse(BinaryReader reader, long size)
    {
        // Each batch is 24 bytes
        int batchCount = (int)(size / 24);
        
        for (int i = 0; i < batchCount; i++)
        {
            RenderBatch batch = new RenderBatch();
            
            if (isPreShadowlands)
            {
                // Read the bounding box for pre-Shadowlands versions
                batch.BoundingBoxMin = new Vector3Short
                {
                    X = reader.ReadInt16(),
                    Y = reader.ReadInt16(),
                    Z = reader.ReadInt16()
                };
                
                batch.BoundingBoxMax = new Vector3Short
                {
                    X = reader.ReadInt16(),
                    Y = reader.ReadInt16(),
                    Z = reader.ReadInt16()
                };
            }
            else
            {
                // Skip the repurposed data in Shadowlands+
                reader.BaseStream.Position += 10;
                
                // Read the extended material ID
                batch.MaterialIdLarge = reader.ReadUInt16();
            }
            
            batch.StartIndex = reader.ReadUInt32();
            batch.Count = reader.ReadUInt16();
            batch.MinIndex = reader.ReadUInt16();
            batch.MaxIndex = reader.ReadUInt16();
            batch.Flags = reader.ReadByte();
            batch.MaterialId = reader.ReadByte();
            
            Batches.Add(batch);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 24 != 0)
        {
            throw new InvalidDataException("MOBA chunk size is not a multiple of 24 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var batch in Batches)
        {
            if (isPreShadowlands)
            {
                // Write the bounding box for pre-Shadowlands versions
                writer.Write(batch.BoundingBoxMin.X);
                writer.Write(batch.BoundingBoxMin.Y);
                writer.Write(batch.BoundingBoxMin.Z);
                
                writer.Write(batch.BoundingBoxMax.X);
                writer.Write(batch.BoundingBoxMax.Y);
                writer.Write(batch.BoundingBoxMax.Z);
            }
            else
            {
                // Write zeroes for the repurposed data in Shadowlands+
                for (int i = 0; i < 10; i++)
                {
                    writer.Write((byte)0);
                }
                
                // Write the extended material ID
                writer.Write(batch.MaterialIdLarge);
            }
            
            writer.Write(batch.StartIndex);
            writer.Write(batch.Count);
            writer.Write(batch.MinIndex);
            writer.Write(batch.MaxIndex);
            writer.Write(batch.Flags);
            writer.Write(batch.MaterialId);
        }
    }
    
    public class RenderBatch
    {
        public Vector3Short BoundingBoxMin { get; set; } = new Vector3Short();
        public Vector3Short BoundingBoxMax { get; set; } = new Vector3Short();
        public ushort MaterialIdLarge { get; set; }
        public uint StartIndex { get; set; }
        public ushort Count { get; set; }
        public ushort MinIndex { get; set; }
        public ushort MaxIndex { get; set; }
        public byte Flags { get; set; }
        public byte MaterialId { get; set; }
        
        public bool UseExtendedMaterialId => (Flags & 0x02) != 0;
    }
    
    public struct Vector3Short
    {
        public short X { get; set; }
        public short Y { get; set; }
        public short Z { get; set; }
    }
}
```

## Usage Context
- Render batches are used to optimize the rendering process by grouping triangles with the same material.
- The batches allow for efficient rendering by minimizing state changes (such as texture or shader switches).
- The bounding box in pre-Shadowlands versions enables quick culling of batches that are outside the view frustum.
- Multiple batches can use the same material but might be separated for better occlusion culling or LOD management.
- The order of batches in the file can be important for transparent materials, which should typically be rendered back-to-front. 