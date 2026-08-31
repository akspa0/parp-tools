using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Model representing Material Diffuse (MDID) and Material Height (MHID) material tables
/// introduced in late Cataclysm (4.3.4) and Mists of Pandaria (5.0.1–5.1.0).
/// </summary>
public sealed class MopMaterialTables
{
    public List<uint> DiffuseFileDataIds { get; } = new();
    public List<uint> HeightFileDataIds { get; } = new();
    public List<string> TextureFilenames { get; } = new();
}

/// <summary>
/// Model representing height blend parameters per chunk layer (MCXH chunk).
/// </summary>
public sealed class MopHeightBlendLayer
{
    public float HeightScale { get; set; } = 1.0f;
    public float HeightOffset { get; set; } = 0.0f;
}

/// <summary>
/// Parsed MoP ADT chunk info containing multi-stream texture, height blend, and object placements.
/// </summary>
public sealed class MopParsedAdtTile
{
    public int TileX { get; set; }
    public int TileY { get; set; }
    public MopMaterialTables Materials { get; } = new();
    public List<MopParsedChunk> Chunks { get; } = new();
    public List<MopObjectPlacement> Placements { get; } = new();
}

/// <summary>
/// Parsed single MCNK chunk in a MoP split ADT.
/// </summary>
public sealed class MopParsedChunk
{
    public int ChunkIndex { get; set; }
    public int ChunkX { get; set; }
    public int ChunkY { get; set; }
    public uint AreaId { get; set; }
    public uint Flags { get; set; }
    public float PositionX { get; set; }
    public float PositionY { get; set; }
    public float PositionZ { get; set; }
    public float[] Heights { get; } = new float[145];
    public byte[]? Normals { get; set; }
    public List<MopChunkLayer> Layers { get; } = new();
    public List<MopHeightBlendLayer> HeightBlends { get; } = new();
}

public sealed class MopChunkLayer
{
    public uint TextureId { get; set; }
    public uint Flags { get; set; }
    public uint EffectId { get; set; }
    public byte[]? AlphaMap { get; set; }
}

public sealed class MopObjectPlacement
{
    public bool IsWmo { get; set; }
    public uint NameId { get; set; }
    public uint UniqueId { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public float Scale { get; set; } = 1.0f;
    public uint Flags { get; set; }
}

/// <summary>
/// Parser for late-Cataclysm (4.3.4) and Mists of Pandaria (5.0.1–5.1.0) split ADT files.
/// </summary>
public static class MopAdtChunkParser
{
    private static uint MakeFourCC(string s) =>
        (uint)(s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24));

    private static readonly uint FourCC_MVER = MakeFourCC("MVER");
    private static readonly uint FourCC_MHDR = MakeFourCC("MHDR");
    private static readonly uint FourCC_MCIN = MakeFourCC("MCIN");
    private static readonly uint FourCC_MTEX = MakeFourCC("MTEX");
    private static readonly uint FourCC_MDID = MakeFourCC("MDID");
    private static readonly uint FourCC_MHID = MakeFourCC("MHID");
    private static readonly uint FourCC_MCNK = MakeFourCC("MCNK");
    private static readonly uint FourCC_MCVT = MakeFourCC("MCVT");
    private static readonly uint FourCC_MCNR = MakeFourCC("MCNR");
    private static readonly uint FourCC_MCLY = MakeFourCC("MCLY");
    private static readonly uint FourCC_MCAL = MakeFourCC("MCAL");
    private static readonly uint FourCC_MCXH = MakeFourCC("MCXH");
    private static readonly uint FourCC_MMDX = MakeFourCC("MMDX");
    private static readonly uint FourCC_MMID = MakeFourCC("MMID");
    private static readonly uint FourCC_MWMO = MakeFourCC("MWMO");
    private static readonly uint FourCC_MWID = MakeFourCC("MWID");
    private static readonly uint FourCC_MDDF = MakeFourCC("MDDF");
    private static readonly uint FourCC_MODF = MakeFourCC("MODF");

    /// <summary>
    /// Parses MDID (Material Diffuse FileDataIDs) chunk.
    /// </summary>
    public static List<uint> ParseMdidChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<uint>(data.Length / 4);
        for (int i = 0; i <= data.Length - 4; i += 4)
        {
            result.Add(BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(i, 4)));
        }
        return result;
    }

    /// <summary>
    /// Parses MHID (Material Height FileDataIDs) chunk.
    /// </summary>
    public static List<uint> ParseMhidChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<uint>(data.Length / 4);
        for (int i = 0; i <= data.Length - 4; i += 4)
        {
            result.Add(BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(i, 4)));
        }
        return result;
    }

    /// <summary>
    /// Parses MCXH (Material Height blend scale &amp; offset) chunk per layer.
    /// Each layer contains 2 floats (scale, offset).
    /// </summary>
    public static List<MopHeightBlendLayer> ParseMcxhChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<MopHeightBlendLayer>();
        int layerCount = data.Length / 8;
        for (int i = 0; i < layerCount; i++)
        {
            int offset = i * 8;
            if (offset + 8 <= data.Length)
            {
                float scale = BinaryPrimitives.ReadSingleLittleEndian(data.Slice(offset, 4));
                float heightOffset = BinaryPrimitives.ReadSingleLittleEndian(data.Slice(offset + 4, 4));
                result.Add(new MopHeightBlendLayer
                {
                    HeightScale = scale,
                    HeightOffset = heightOffset,
                });
            }
        }
        return result;
    }

    /// <summary>
    /// Parses MTEX null-terminated string table.
    /// </summary>
    public static List<string> ParseMtexChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                {
                    string str = Encoding.ASCII.GetString(data.Slice(start, i - start));
                    result.Add(str);
                }
                start = i + 1;
            }
        }
        return result;
    }

    /// <summary>
    /// Reads MODF (WMO placements) in MoP format with scale and bounds support.
    /// </summary>
    public static List<MopObjectPlacement> ParseModfChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<MopObjectPlacement>();
        const int recordSize = 64; // Standard Blizzard MODF record size
        int count = data.Length / recordSize;

        for (int i = 0; i < count; i++)
        {
            var span = data.Slice(i * recordSize, recordSize);
            uint nameId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(0, 4));
            uint uniqueId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(4, 4));
            float posX = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(8, 4));
            float posY = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(12, 4));
            float posZ = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(16, 4));
            float rotX = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(20, 4));
            float rotY = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(24, 4));
            float rotZ = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(28, 4));
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(56, 4));

            result.Add(new MopObjectPlacement
            {
                IsWmo = true,
                NameId = nameId,
                UniqueId = uniqueId,
                Position = new Vector3(posX, posY, posZ),
                Rotation = new Vector3(rotX, rotY, rotZ),
                Scale = 1.0f,
                Flags = flags,
            });
        }
        return result;
    }

    /// <summary>
    /// Reads MDDF (M2 doodad placements) in MoP format.
    /// </summary>
    public static List<MopObjectPlacement> ParseMddfChunk(ReadOnlySpan<byte> data)
    {
        var result = new List<MopObjectPlacement>();
        const int recordSize = 36; // Standard Blizzard MDDF record size
        int count = data.Length / recordSize;

        for (int i = 0; i < count; i++)
        {
            var span = data.Slice(i * recordSize, recordSize);
            uint nameId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(0, 4));
            uint uniqueId = BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(4, 4));
            float posX = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(8, 4));
            float posY = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(12, 4));
            float posZ = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(16, 4));
            float rotX = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(20, 4));
            float rotY = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(24, 4));
            float rotZ = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(28, 4));
            ushort scaleInt = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(32, 2));
            ushort flags = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(34, 2));

            float scale = scaleInt / 1024.0f;
            if (scale <= 0.0001f)
                scale = 1.0f;

            result.Add(new MopObjectPlacement
            {
                IsWmo = false,
                NameId = nameId,
                UniqueId = uniqueId,
                Position = new Vector3(posX, posY, posZ),
                Rotation = new Vector3(rotX, rotY, rotZ),
                Scale = scale,
                Flags = flags,
            });
        }
        return result;
    }
}
