using System.Runtime.InteropServices;

namespace WoWMapConverter.Core.Formats.Liquids;

/// <summary>
/// MCLQ liquid chunk structure (pre-WotLK format, still parsed for backwards compatibility).
/// Based on wowdev.wiki ADT_v18 documentation and Noggit3 implementation.
/// </summary>
public class MclqChunk
{
    public const int VertexGridSize = 9;  // 9x9 vertices
    public const int TileGridSize = 8;    // 8x8 tiles
    public const int VertexCount = 81;    // 9*9
    public const int TileCount = 64;      // 8*8

    /// <summary>Minimum height of liquid surface.</summary>
    public float MinHeight { get; set; }

    /// <summary>Maximum height of liquid surface.</summary>
    public float MaxHeight { get; set; }

    /// <summary>9x9 vertex data (81 vertices).</summary>
    public MclqVertex[] Vertices { get; set; } = new MclqVertex[VertexCount];

    /// <summary>8x8 tile flags.</summary>
    public MclqTile[] Tiles { get; set; } = new MclqTile[TileCount];

    /// <summary>Number of flow vectors (0-2).</summary>
    public uint FlowCount { get; set; }

    /// <summary>Flow vectors (always 2 in file, but FlowCount indicates how many are valid).</summary>
    public MclqFlowVector[] FlowVectors { get; set; } = new MclqFlowVector[2];

    /// <summary>
    /// Parse MCLQ chunk from raw bytes.
    /// </summary>
    public static MclqChunk? Parse(byte[] data, MclqLiquidType liquidType)
    {
        if (data == null || data.Length < 8) // At minimum need height range
            return null;

        var chunk = new MclqChunk();
        int offset = 0;

        // Read height range (CRange)
        chunk.MinHeight = BitConverter.ToSingle(data, offset); offset += 4;
        chunk.MaxHeight = BitConverter.ToSingle(data, offset); offset += 4;

        // Read 9x9 vertices (81 vertices)
        for (int i = 0; i < VertexCount && offset + 8 <= data.Length; i++)
        {
            chunk.Vertices[i] = MclqVertex.Parse(data, ref offset, liquidType);
        }

        // Read 8x8 tile flags (64 bytes)
        for (int i = 0; i < TileCount && offset < data.Length; i++)
        {
            chunk.Tiles[i] = new MclqTile(data[offset++]);
        }

        // Read flow vector count and vectors
        if (offset + 4 <= data.Length)
        {
            chunk.FlowCount = BitConverter.ToUInt32(data, offset); offset += 4;

            // Always 2 flow vectors in file
            for (int i = 0; i < 2 && offset + MclqFlowVector.Size <= data.Length; i++)
            {
                chunk.FlowVectors[i] = MclqFlowVector.Parse(data, ref offset);
            }
        }

        return chunk;
    }

    /// <summary>
    /// Serialize MCLQ chunk to bytes.
    /// </summary>
    public byte[] ToBytes(MclqLiquidType liquidType)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Write height range
        bw.Write(MinHeight);
        bw.Write(MaxHeight);

        // Write 9x9 vertices
        foreach (var vertex in Vertices)
        {
            vertex.Write(bw, liquidType);
        }

        // Write 8x8 tile flags
        foreach (var tile in Tiles)
        {
            bw.Write(tile.RawValue);
        }

        // Write flow vectors
        bw.Write(FlowCount);
        foreach (var flow in FlowVectors)
        {
            flow.Write(bw);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Calculate expected chunk size based on liquid type.
    /// </summary>
    public static int CalculateSize(MclqLiquidType liquidType)
    {
        // 8 bytes height range + 81 vertices + 64 tiles + 4 bytes flow count + 2 flow vectors
        int vertexSize = liquidType == MclqLiquidType.Magma ? 8 : 8; // Both are 8 bytes per vertex
        return 8 + (VertexCount * vertexSize) + TileCount + 4 + (2 * MclqFlowVector.Size);
    }
}

/// <summary>
/// MCLQ vertex data - union of water/ocean/magma vertex types.
/// </summary>
public struct MclqVertex
{
    /// <summary>Height of this vertex.</summary>
    public float Height;

    // Water/Ocean vertex data
    public byte Depth;
    public byte Flow0Pct;
    public byte Flow1Pct;
    public byte Filler;

    // Magma vertex data (UV coordinates)
    public ushort MagmaS;
    public ushort MagmaT;

    public static MclqVertex Parse(byte[] data, ref int offset, MclqLiquidType liquidType)
    {
        var v = new MclqVertex();

        if (liquidType == MclqLiquidType.Magma)
        {
            // Magma: s, t (uint16), then height (float)
            v.MagmaS = BitConverter.ToUInt16(data, offset); offset += 2;
            v.MagmaT = BitConverter.ToUInt16(data, offset); offset += 2;
            v.Height = BitConverter.ToSingle(data, offset); offset += 4;
        }
        else
        {
            // Water/Ocean/Slime: depth, flow0, flow1, filler, then height
            v.Depth = data[offset++];
            v.Flow0Pct = data[offset++];
            v.Flow1Pct = data[offset++];
            v.Filler = data[offset++];
            v.Height = BitConverter.ToSingle(data, offset); offset += 4;
        }

        return v;
    }

    public void Write(BinaryWriter bw, MclqLiquidType liquidType)
    {
        if (liquidType == MclqLiquidType.Magma)
        {
            bw.Write(MagmaS);
            bw.Write(MagmaT);
            bw.Write(Height);
        }
        else
        {
            bw.Write(Depth);
            bw.Write(Flow0Pct);
            bw.Write(Flow1Pct);
            bw.Write(Filler);
            bw.Write(Height);
        }
    }
}

/// <summary>
/// MCLQ tile flags (8x8 grid).
/// </summary>
public struct MclqTile
{
    public byte RawValue;

    public MclqTile(byte value) => RawValue = value;

    /// <summary>Liquid type from lower 4 bits.</summary>
    public MclqLiquidType LiquidType => (MclqLiquidType)(RawValue & 0x0F);

    /// <summary>Don't render this tile (0x0F or 0x08).</summary>
    public bool DontRender => (RawValue & 0x0F) == 0x0F || (RawValue & 0x08) != 0;

    /// <summary>Unknown flag 0x10.</summary>
    public bool Flag10 => (RawValue & 0x10) != 0;

    /// <summary>Unknown flag 0x20.</summary>
    public bool Flag20 => (RawValue & 0x20) != 0;

    /// <summary>Not low depth / forced swimming (0x40).</summary>
    public bool ForcedSwim => (RawValue & 0x40) != 0;

    /// <summary>Fatigue area (0x80).</summary>
    public bool Fatigue => (RawValue & 0x80) != 0;

    /// <summary>Create a tile with specified type and flags.</summary>
    public static MclqTile Create(MclqLiquidType type, bool dontRender = false, bool forcedSwim = false, bool fatigue = false)
    {
        byte value = (byte)type;
        if (dontRender) value = 0x0F;
        if (forcedSwim) value |= 0x40;
        if (fatigue) value |= 0x80;
        return new MclqTile(value);
    }
}

/// <summary>
/// MCLQ liquid type (lower 4 bits of tile flags).
/// </summary>
public enum MclqLiquidType : byte
{
    None = 0,
    Ocean = 1,
    Slime = 3,
    River = 4,
    Magma = 6,
    DontRender = 0x0F
}

/// <summary>
/// MCLQ flow vector for animated water.
/// </summary>
public struct MclqFlowVector
{
    public const int Size = 40; // CAaSphere (16) + C3Vector (12) + 3 floats (12)

    // CAaSphere (bounding sphere)
    public float SphereX, SphereY, SphereZ, SphereRadius;

    // Direction vector
    public float DirX, DirY, DirZ;

    // Animation parameters
    public float Velocity;
    public float Amplitude;
    public float Frequency;

    public static MclqFlowVector Parse(byte[] data, ref int offset)
    {
        var f = new MclqFlowVector();
        f.SphereX = BitConverter.ToSingle(data, offset); offset += 4;
        f.SphereY = BitConverter.ToSingle(data, offset); offset += 4;
        f.SphereZ = BitConverter.ToSingle(data, offset); offset += 4;
        f.SphereRadius = BitConverter.ToSingle(data, offset); offset += 4;
        f.DirX = BitConverter.ToSingle(data, offset); offset += 4;
        f.DirY = BitConverter.ToSingle(data, offset); offset += 4;
        f.DirZ = BitConverter.ToSingle(data, offset); offset += 4;
        f.Velocity = BitConverter.ToSingle(data, offset); offset += 4;
        f.Amplitude = BitConverter.ToSingle(data, offset); offset += 4;
        f.Frequency = BitConverter.ToSingle(data, offset); offset += 4;
        return f;
    }

    public void Write(BinaryWriter bw)
    {
        bw.Write(SphereX);
        bw.Write(SphereY);
        bw.Write(SphereZ);
        bw.Write(SphereRadius);
        bw.Write(DirX);
        bw.Write(DirY);
        bw.Write(DirZ);
        bw.Write(Velocity);
        bw.Write(Amplitude);
        bw.Write(Frequency);
    }
}
