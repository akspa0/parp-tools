namespace WoWMapConverter.Core.Formats.Liquids;

/// <summary>
/// MH2O liquid chunk structure (WotLK+ format).
/// Based on wowdev.wiki ADT_v18 documentation.
/// </summary>
public class Mh2oChunk
{
    public const int ChunkCount = 256; // 16x16 MCNK chunks

    /// <summary>Per-chunk headers (256 entries).</summary>
    public Mh2oChunkHeader[] Headers { get; set; } = new Mh2oChunkHeader[ChunkCount];

    /// <summary>Liquid instances for all chunks.</summary>
    public List<Mh2oInstance> Instances { get; set; } = new();

    /// <summary>Per-chunk attributes.</summary>
    public Mh2oAttributes[] Attributes { get; set; } = new Mh2oAttributes[ChunkCount];

    /// <summary>
    /// Parse MH2O chunk from raw bytes.
    /// </summary>
    public static Mh2oChunk Parse(byte[] data)
    {
        var chunk = new Mh2oChunk();
        int offset = 0;

        // Read 256 chunk headers (12 bytes each)
        for (int i = 0; i < ChunkCount; i++)
        {
            chunk.Headers[i] = new Mh2oChunkHeader
            {
                OffsetInstances = BitConverter.ToUInt32(data, offset),
                LayerCount = BitConverter.ToUInt32(data, offset + 4),
                OffsetAttributes = BitConverter.ToUInt32(data, offset + 8)
            };
            offset += 12;
        }

        // Parse instances and attributes for each chunk
        for (int i = 0; i < ChunkCount; i++)
        {
            var header = chunk.Headers[i];

            // Parse attributes if present
            if (header.OffsetAttributes > 0 && header.OffsetAttributes < data.Length - 16)
            {
                int attrOffset = (int)header.OffsetAttributes;
                chunk.Attributes[i] = new Mh2oAttributes
                {
                    Fishable = BitConverter.ToUInt64(data, attrOffset),
                    Deep = BitConverter.ToUInt64(data, attrOffset + 8)
                };
            }

            // Parse instances
            if (header.LayerCount > 0 && header.OffsetInstances > 0 && header.OffsetInstances < data.Length)
            {
                int instOffset = (int)header.OffsetInstances;
                for (int layer = 0; layer < header.LayerCount && instOffset + 24 <= data.Length; layer++)
                {
                    var instance = Mh2oInstance.Parse(data, ref instOffset, i);
                    chunk.Instances.Add(instance);
                }
            }
        }

        return chunk;
    }

    /// <summary>
    /// Get instances for a specific MCNK chunk index.
    /// </summary>
    public IEnumerable<Mh2oInstance> GetInstancesForChunk(int chunkIndex)
    {
        return Instances.Where(i => i.ChunkIndex == chunkIndex);
    }
}

/// <summary>
/// MH2O per-chunk header (SMLiquidChunk).
/// </summary>
public struct Mh2oChunkHeader
{
    /// <summary>Offset to SMLiquidInstance array.</summary>
    public uint OffsetInstances;

    /// <summary>Number of liquid layers (0 = no liquid).</summary>
    public uint LayerCount;

    /// <summary>Offset to mh2o_chunk_attributes.</summary>
    public uint OffsetAttributes;
}

/// <summary>
/// MH2O chunk attributes (8x8 bit masks).
/// </summary>
public struct Mh2oAttributes
{
    /// <summary>Fishable mask (8x8 bits).</summary>
    public ulong Fishable;

    /// <summary>Deep/fatigue mask (8x8 bits).</summary>
    public ulong Deep;

    public bool IsFishable(int x, int y) => ((Fishable >> (y * 8 + x)) & 1) != 0;
    public bool IsDeep(int x, int y) => ((Deep >> (y * 8 + x)) & 1) != 0;

    public void SetFishable(int x, int y, bool value)
    {
        ulong bit = 1UL << (y * 8 + x);
        if (value) Fishable |= bit;
        else Fishable &= ~bit;
    }

    public void SetDeep(int x, int y, bool value)
    {
        ulong bit = 1UL << (y * 8 + x);
        if (value) Deep |= bit;
        else Deep &= ~bit;
    }
}

/// <summary>
/// MH2O liquid instance (SMLiquidInstance).
/// </summary>
public class Mh2oInstance
{
    /// <summary>Parent chunk index (0-255).</summary>
    public int ChunkIndex { get; set; }

    /// <summary>LiquidType.dbc entry ID.</summary>
    public ushort LiquidTypeId { get; set; }

    /// <summary>Liquid vertex format (0-3).</summary>
    public Mh2oVertexFormat VertexFormat { get; set; }

    /// <summary>Minimum height level.</summary>
    public float MinHeightLevel { get; set; }

    /// <summary>Maximum height level.</summary>
    public float MaxHeightLevel { get; set; }

    /// <summary>X offset in chunk (0-7).</summary>
    public byte XOffset { get; set; }

    /// <summary>Y offset in chunk (0-7).</summary>
    public byte YOffset { get; set; }

    /// <summary>Width in tiles (1-8).</summary>
    public byte Width { get; set; }

    /// <summary>Height in tiles (1-8).</summary>
    public byte Height { get; set; }

    /// <summary>Exists bitmap offset (0 = all exist).</summary>
    public uint OffsetExistsBitmap { get; set; }

    /// <summary>Vertex data offset.</summary>
    public uint OffsetVertexData { get; set; }

    /// <summary>Parsed exists bitmap (null = all exist).</summary>
    public byte[]? ExistsBitmap { get; set; }

    /// <summary>Height map data (if present).</summary>
    public float[]? HeightMap { get; set; }

    /// <summary>Depth map data (if present).</summary>
    public byte[]? DepthMap { get; set; }

    /// <summary>UV map data for magma (if present).</summary>
    public ushort[]? UvMap { get; set; }

    /// <summary>Number of vertices: (Width+1) * (Height+1).</summary>
    public int VertexCount => (Width + 1) * (Height + 1);

    public static Mh2oInstance Parse(byte[] data, ref int offset, int chunkIndex)
    {
        var inst = new Mh2oInstance { ChunkIndex = chunkIndex };

        inst.LiquidTypeId = BitConverter.ToUInt16(data, offset); offset += 2;
        inst.VertexFormat = (Mh2oVertexFormat)BitConverter.ToUInt16(data, offset); offset += 2;
        inst.MinHeightLevel = BitConverter.ToSingle(data, offset); offset += 4;
        inst.MaxHeightLevel = BitConverter.ToSingle(data, offset); offset += 4;
        inst.XOffset = data[offset++];
        inst.YOffset = data[offset++];
        inst.Width = data[offset++];
        inst.Height = data[offset++];
        inst.OffsetExistsBitmap = BitConverter.ToUInt32(data, offset); offset += 4;
        inst.OffsetVertexData = BitConverter.ToUInt32(data, offset); offset += 4;

        // Parse exists bitmap if present
        if (inst.OffsetExistsBitmap > 0 && inst.OffsetExistsBitmap < data.Length)
        {
            int bitmapSize = (inst.Width * inst.Height + 7) / 8;
            inst.ExistsBitmap = new byte[bitmapSize];
            Array.Copy(data, (int)inst.OffsetExistsBitmap, inst.ExistsBitmap, 0, 
                Math.Min(bitmapSize, data.Length - (int)inst.OffsetExistsBitmap));
        }

        // Parse vertex data based on format
        if (inst.OffsetVertexData > 0 && inst.OffsetVertexData < data.Length)
        {
            int vOffset = (int)inst.OffsetVertexData;
            int vCount = inst.VertexCount;

            switch (inst.VertexFormat)
            {
                case Mh2oVertexFormat.HeightDepth: // Case 0: float height + byte depth
                    inst.HeightMap = new float[vCount];
                    inst.DepthMap = new byte[vCount];
                    for (int i = 0; i < vCount && vOffset + 4 <= data.Length; i++)
                    {
                        inst.HeightMap[i] = BitConverter.ToSingle(data, vOffset);
                        vOffset += 4;
                    }
                    for (int i = 0; i < vCount && vOffset < data.Length; i++)
                    {
                        inst.DepthMap[i] = data[vOffset++];
                    }
                    break;

                case Mh2oVertexFormat.HeightUv: // Case 1: float height + ushort[2] uv
                    inst.HeightMap = new float[vCount];
                    inst.UvMap = new ushort[vCount * 2];
                    for (int i = 0; i < vCount && vOffset + 4 <= data.Length; i++)
                    {
                        inst.HeightMap[i] = BitConverter.ToSingle(data, vOffset);
                        vOffset += 4;
                    }
                    for (int i = 0; i < vCount * 2 && vOffset + 2 <= data.Length; i++)
                    {
                        inst.UvMap[i] = BitConverter.ToUInt16(data, vOffset);
                        vOffset += 2;
                    }
                    break;

                case Mh2oVertexFormat.DepthOnly: // Case 2: byte depth only
                    inst.DepthMap = new byte[vCount];
                    for (int i = 0; i < vCount && vOffset < data.Length; i++)
                    {
                        inst.DepthMap[i] = data[vOffset++];
                    }
                    break;

                case Mh2oVertexFormat.HeightUvDepth: // Case 3: float height + ushort[2] uv + byte depth
                    inst.HeightMap = new float[vCount];
                    inst.UvMap = new ushort[vCount * 2];
                    inst.DepthMap = new byte[vCount];
                    for (int i = 0; i < vCount && vOffset + 4 <= data.Length; i++)
                    {
                        inst.HeightMap[i] = BitConverter.ToSingle(data, vOffset);
                        vOffset += 4;
                    }
                    for (int i = 0; i < vCount * 2 && vOffset + 2 <= data.Length; i++)
                    {
                        inst.UvMap[i] = BitConverter.ToUInt16(data, vOffset);
                        vOffset += 2;
                    }
                    for (int i = 0; i < vCount && vOffset < data.Length; i++)
                    {
                        inst.DepthMap[i] = data[vOffset++];
                    }
                    break;
            }
        }

        return inst;
    }

    /// <summary>Check if a tile exists in this instance.</summary>
    public bool TileExists(int localX, int localY)
    {
        if (ExistsBitmap == null) return true;
        int bitIndex = localY * Width + localX;
        int byteIndex = bitIndex / 8;
        int bit = bitIndex % 8;
        return byteIndex < ExistsBitmap.Length && (ExistsBitmap[byteIndex] & (1 << bit)) != 0;
    }
}

/// <summary>
/// MH2O liquid vertex format (LVF).
/// </summary>
public enum Mh2oVertexFormat : ushort
{
    /// <summary>Case 0: float heightmap + byte depthmap.</summary>
    HeightDepth = 0,

    /// <summary>Case 1: float heightmap + ushort[2] uvmap.</summary>
    HeightUv = 1,

    /// <summary>Case 2: byte depthmap only (height = 0).</summary>
    DepthOnly = 2,

    /// <summary>Case 3: float heightmap + ushort[2] uvmap + byte depthmap.</summary>
    HeightUvDepth = 3
}
