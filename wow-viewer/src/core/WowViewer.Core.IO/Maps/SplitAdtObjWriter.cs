using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Writes standard split-format <c>_obj0.adt</c> placement files.
/// Compatible with Cataclysm/MoP split ADT structures and authentic developer datasets.
/// </summary>
public static class SplitAdtObjWriter
{
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;

    /// <summary>
    /// Builds binary bytes for a complete split <c>_obj0.adt</c> file from an <see cref="AdtPlacementCatalog"/>.
    /// </summary>
    public static byte[] Build(AdtPlacementCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        using MemoryStream ms = new(4096);

        // 1. MVER
        byte[] mverPayload = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(mverPayload, 18u);
        WriteChunk(ms, MapChunkIds.Mver, mverPayload);

        // 2. MMDX & MMID (M2 models)
        byte[] mmdx = SerializeStringBlock(catalog.ModelNames);
        byte[] mmid = SerializeOffsets(catalog.ModelNames);
        WriteChunk(ms, MapChunkIds.Mmdx, mmdx);
        WriteChunk(ms, MapChunkIds.Mmid, mmid);

        // 3. MWMO & MWID (WMO models)
        byte[] mwmo = SerializeStringBlock(catalog.WorldModelNames);
        byte[] mwid = SerializeOffsets(catalog.WorldModelNames);
        WriteChunk(ms, MapChunkIds.Mwmo, mwmo);
        WriteChunk(ms, MapChunkIds.Mwid, mwid);

        // 4. MDDF (M2 placements)
        byte[] mddf = SerializeMddf(catalog.ModelPlacements);
        WriteChunk(ms, MapChunkIds.Mddf, mddf);

        // 5. MODF (WMO placements)
        byte[] modf = SerializeModf(catalog.WorldModelPlacements);
        WriteChunk(ms, MapChunkIds.Modf, modf);

        // 6. 256 empty MCNK chunks
        byte[] emptyMcnkHeader = MapChunkIds.Mcnk.ToFileBytes();
        byte[] zeroLength = new byte[4];
        for (int i = 0; i < 256; i++)
        {
            ms.Write(emptyMcnkHeader, 0, 4);
            ms.Write(zeroLength, 0, 4);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Writes the placement catalog to disk as a split <c>_obj0.adt</c> file.
    /// </summary>
    public static void Write(string filePath, AdtPlacementCatalog catalog)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        ArgumentNullException.ThrowIfNull(catalog);

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        File.WriteAllBytes(filePath, Build(catalog));
    }

    private static void WriteChunk(Stream output, FourCC id, byte[] payload)
    {
        output.Write(id.ToFileBytes(), 0, 4);
        Span<byte> size = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(size, (uint)payload.Length);
        output.Write(size);
        if (payload.Length > 0)
            output.Write(payload, 0, payload.Length);
    }

    private static byte[] SerializeStringBlock(IReadOnlyList<string> names)
    {
        if (names.Count == 0)
            return [];

        using MemoryStream ms = new();
        foreach (string name in names)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(name);
            ms.Write(bytes);
            ms.WriteByte(0);
        }
        return ms.ToArray();
    }

    private static byte[] SerializeOffsets(IReadOnlyList<string> names)
    {
        if (names.Count == 0)
            return [];

        byte[] bytes = new byte[names.Count * sizeof(uint)];
        uint offset = 0;
        for (int index = 0; index < names.Count; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint), sizeof(uint)), offset);
            offset += (uint)Encoding.ASCII.GetByteCount(names[index]) + 1u;
        }
        return bytes;
    }

    private static byte[] SerializeMddf(IReadOnlyList<AdtModelPlacement> placements)
    {
        if (placements.Count == 0)
            return [];

        byte[] bytes = new byte[placements.Count * MddfEntrySize];
        for (int index = 0; index < placements.Count; index++)
        {
            AdtModelPlacement row = placements[index];
            int offset = index * MddfEntrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset, 4), (uint)row.NameId);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 4, 4), unchecked((uint)row.UniqueId));
            WriteSingle(bytes, offset + 8, MapOrigin - row.Position.Y);
            WriteSingle(bytes, offset + 12, row.Position.Z);
            WriteSingle(bytes, offset + 16, MapOrigin - row.Position.X);
            WriteSingle(bytes, offset + 20, row.Rotation.X);
            WriteSingle(bytes, offset + 24, row.Rotation.Z);
            WriteSingle(bytes, offset + 28, row.Rotation.Y);
            ushort scale = (ushort)Math.Clamp((int)MathF.Round(row.Scale * 1024f), ushort.MinValue, ushort.MaxValue);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 32, 2), scale);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 34, 2), 0);
        }
        return bytes;
    }

    private static byte[] SerializeModf(IReadOnlyList<AdtWorldModelPlacement> placements)
    {
        if (placements.Count == 0)
            return [];

        byte[] bytes = new byte[placements.Count * ModfEntrySize];
        for (int index = 0; index < placements.Count; index++)
        {
            AdtWorldModelPlacement row = placements[index];
            int offset = index * ModfEntrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset, 4), (uint)row.NameId);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 4, 4), unchecked((uint)row.UniqueId));
            WriteSingle(bytes, offset + 8, MapOrigin - row.Position.Y);
            WriteSingle(bytes, offset + 12, row.Position.Z);
            WriteSingle(bytes, offset + 16, MapOrigin - row.Position.X);
            WriteSingle(bytes, offset + 20, row.Rotation.X);
            WriteSingle(bytes, offset + 24, row.Rotation.Z);
            WriteSingle(bytes, offset + 28, row.Rotation.Y);

            // Bounding box: in raw MODF order
            WriteSingle(bytes, offset + 32, MapOrigin - row.BoundsMax.Y);
            WriteSingle(bytes, offset + 36, row.BoundsMin.Z);
            WriteSingle(bytes, offset + 40, MapOrigin - row.BoundsMax.X);
            WriteSingle(bytes, offset + 44, MapOrigin - row.BoundsMin.Y);
            WriteSingle(bytes, offset + 48, row.BoundsMax.Z);
            WriteSingle(bytes, offset + 52, MapOrigin - row.BoundsMin.X);

            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 56, 2), unchecked((ushort)row.Flags));
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 58, 2), unchecked((ushort)row.DoodadSet));
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 60, 2), unchecked((ushort)row.NameSet));
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 62, 2), unchecked((ushort)row.Scale));
        }
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
