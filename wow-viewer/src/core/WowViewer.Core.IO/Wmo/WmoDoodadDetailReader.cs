using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadDetailReader
{
    private const int ModsEntrySize = 32;
    private const int ModdEntrySize = 40;
    private const int NameBytes = 20;

    public static IReadOnlyList<WmoDoodadSetDetail> ReadSets(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadSets(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoDoodadSetDetail> ReadSets(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (_, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mods);
        if (payload.Length % ModsEntrySize != 0)
            throw new InvalidDataException($"MODS payload size {payload.Length} is not divisible by {ModsEntrySize}.");

        int entryCount = payload.Length / ModsEntrySize;
        List<WmoDoodadSetDetail> sets = new(entryCount);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * ModsEntrySize;
            string name = ReadFixedString(payload.AsSpan(offset, NameBytes));
            int startIndex = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 20, 4)));
            int count = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 24, 4)));
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 28, 4));
            sets.Add(new WmoDoodadSetDetail(
                index,
                string.IsNullOrWhiteSpace(name) ? $"Set {index}" : name,
                startIndex,
                count,
                startIndex + count,
                flags));
        }

        return sets;
    }

    public static IReadOnlyList<WmoDoodadPlacementDetail> ReadPlacements(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadPlacements(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoDoodadPlacementDetail> ReadPlacements(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (_, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] modd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modd);
        byte[] modn = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modn);
        if (modd.Length % ModdEntrySize != 0)
            throw new InvalidDataException($"MODD payload size {modd.Length} is not divisible by {ModdEntrySize}.");

        int entryCount = modd.Length / ModdEntrySize;
        List<WmoDoodadPlacementDetail> placements = new(entryCount);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * ModdEntrySize;
            uint nameIndex = BinaryPrimitives.ReadUInt32LittleEndian(modd.AsSpan(offset, 4)) & 0x00FFFFFFu;
            Vector3 position = new(
                BitConverter.ToSingle(modd, offset + 4),
                BitConverter.ToSingle(modd, offset + 8),
                BitConverter.ToSingle(modd, offset + 12));
            Quaternion rotation = new(
                BitConverter.ToSingle(modd, offset + 16),
                BitConverter.ToSingle(modd, offset + 20),
                BitConverter.ToSingle(modd, offset + 24),
                BitConverter.ToSingle(modd, offset + 28));
            float scale = BitConverter.ToSingle(modd, offset + 32);
            uint colorBgra = BinaryPrimitives.ReadUInt32LittleEndian(modd.AsSpan(offset + 36, 4));
            placements.Add(new WmoDoodadPlacementDetail(
                index,
                nameIndex,
                WmoRootReaderCommon.ResolveStringAtOffset(modn, nameIndex),
                position,
                rotation,
                scale,
                colorBgra));
        }

        return placements;
    }

    private static string ReadFixedString(ReadOnlySpan<byte> bytes)
    {
        int terminator = bytes.IndexOf((byte)0);
        if (terminator < 0)
            terminator = bytes.Length;

        return Encoding.UTF8.GetString(bytes.Slice(0, terminator));
    }
}