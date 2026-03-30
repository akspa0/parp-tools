using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtPlacementReader
{
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;

    public static AdtPlacementCatalog Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtPlacementCatalog Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtObj))
            throw new InvalidDataException($"ADT placement reading requires an ADT or ADTOBJ file, but found {fileSummary.Kind}.");

        byte[]? mmdxData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mmdx);
        byte[]? mmidData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mmid);
        byte[]? mwmoData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mwmo);
        byte[]? mwidData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mwid);
        byte[]? mddfData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mddf);
        byte[]? modfData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Modf);

        IReadOnlyList<string> modelNames = MapSummaryReaderCommon.ReadStringEntries(mmdxData);
        IReadOnlyList<string> worldModelNames = MapSummaryReaderCommon.ReadStringEntries(mwmoData);
        List<uint> mmidEntries = ReadIndexEntries(mmidData);
        List<uint> mwidEntries = ReadIndexEntries(mwidData);

        List<AdtModelPlacement> modelPlacements = ReadModelPlacements(mddfData, mmdxData, mmidEntries);
        List<AdtWorldModelPlacement> worldModelPlacements = ReadWorldModelPlacements(modfData, mwmoData, mwidEntries);

        return new AdtPlacementCatalog(
            fileSummary.SourcePath,
            fileSummary.Kind,
            modelNames,
            worldModelNames,
            modelPlacements,
            worldModelPlacements);
    }

    private static List<AdtModelPlacement> ReadModelPlacements(byte[]? payload, byte[]? stringBlock, List<uint> xidEntries)
    {
        if (payload is not { Length: >= MddfEntrySize })
            return [];

        int count = payload.Length / MddfEntrySize;
        List<AdtModelPlacement> placements = new(count);

        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> entry = payload.AsSpan(index * MddfEntrySize, MddfEntrySize);
            uint nameId = BinaryPrimitives.ReadUInt32LittleEndian(entry[0..4]);
            int uniqueId = unchecked((int)BinaryPrimitives.ReadUInt32LittleEndian(entry[4..8]));
            float rawX = ReadSingle(entry[8..12]);
            float rawZ = ReadSingle(entry[12..16]);
            float rawY = ReadSingle(entry[16..20]);
            float rotX = ReadSingle(entry[20..24]);
            float rotZ = ReadSingle(entry[24..28]);
            float rotY = ReadSingle(entry[28..32]);
            ushort scale = BinaryPrimitives.ReadUInt16LittleEndian(entry[32..34]);

            placements.Add(new AdtModelPlacement(
                checked((int)nameId),
                ResolveNameViaXid(nameId, xidEntries, stringBlock),
                uniqueId,
                new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new Vector3(rotX, rotY, rotZ),
                scale / 1024f));
        }

        return placements;
    }

    private static List<AdtWorldModelPlacement> ReadWorldModelPlacements(byte[]? payload, byte[]? stringBlock, List<uint> xidEntries)
    {
        if (payload is not { Length: >= ModfEntrySize })
            return [];

        int count = payload.Length / ModfEntrySize;
        List<AdtWorldModelPlacement> placements = new(count);

        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> entry = payload.AsSpan(index * ModfEntrySize, ModfEntrySize);
            uint nameId = BinaryPrimitives.ReadUInt32LittleEndian(entry[0..4]);
            int uniqueId = unchecked((int)BinaryPrimitives.ReadUInt32LittleEndian(entry[4..8]));
            float rawX = ReadSingle(entry[8..12]);
            float rawZ = ReadSingle(entry[12..16]);
            float rawY = ReadSingle(entry[16..20]);
            float rotX = ReadSingle(entry[20..24]);
            float rotZ = ReadSingle(entry[24..28]);
            float rotY = ReadSingle(entry[28..32]);
            float bbMinX = ReadSingle(entry[32..36]);
            float bbMinZ = ReadSingle(entry[36..40]);
            float bbMinY = ReadSingle(entry[40..44]);
            float bbMaxX = ReadSingle(entry[44..48]);
            float bbMaxZ = ReadSingle(entry[48..52]);
            float bbMaxY = ReadSingle(entry[52..56]);
            ushort flags = BinaryPrimitives.ReadUInt16LittleEndian(entry[56..58]);

            placements.Add(new AdtWorldModelPlacement(
                checked((int)nameId),
                ResolveNameViaXid(nameId, xidEntries, stringBlock),
                uniqueId,
                new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new Vector3(rotX, rotY, rotZ),
                new Vector3(MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ),
                new Vector3(MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ),
                flags));
        }

        return placements;
    }

    private static List<uint> ReadIndexEntries(byte[]? payload)
    {
        if (payload is not { Length: >= sizeof(uint) })
            return [];

        int count = payload.Length / sizeof(uint);
        List<uint> entries = new(count);
        for (int index = 0; index < count; index++)
            entries.Add(BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(index * sizeof(uint), sizeof(uint))));

        return entries;
    }

    private static string ResolveNameViaXid(uint nameId, IReadOnlyList<uint> xidEntries, byte[]? stringBlock)
    {
        if (stringBlock is not { Length: > 0 } || nameId >= xidEntries.Count)
            return $"unknown_{nameId}";

        uint byteOffset = xidEntries[(int)nameId];
        if (byteOffset >= stringBlock.Length)
            return $"unknown_{nameId}";

        int start = checked((int)byteOffset);
        int end = start;
        while (end < stringBlock.Length && stringBlock[end] != 0)
            end++;

        return Encoding.ASCII.GetString(stringBlock, start, end - start);
    }

    private static float ReadSingle(ReadOnlySpan<byte> bytes)
    {
        return BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes));
    }
}