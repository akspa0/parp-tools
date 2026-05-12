using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoV17ToV14Converter
{
    private static readonly FourCC[] RootChunkOrder =
    [
        WmoChunkIds.Mohd,
        WmoChunkIds.Motx,
        WmoChunkIds.Momt,
        WmoChunkIds.Mogn,
        WmoChunkIds.Mogi,
        WmoChunkIds.Mosb,
        WmoChunkIds.Mopv,
        WmoChunkIds.Mopt,
        WmoChunkIds.Mopr,
        WmoChunkIds.Movv,
        WmoChunkIds.Movb,
        WmoChunkIds.Molt,
        WmoChunkIds.Mods,
        WmoChunkIds.Modn,
        WmoChunkIds.Modd,
        WmoChunkIds.Mfog,
        WmoChunkIds.Mcvp,
    ];

    public static void Convert(string v17RootPath, string outputPath, string? groupsDirectory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(v17RootPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        byte[] rootBytes = File.ReadAllBytes(v17RootPath);
        RootChunkPayloads rootPayloads = ParseRootPayloads(rootBytes, v17RootPath);

        string groupDirectory = groupsDirectory ?? Path.GetDirectoryName(v17RootPath) ?? ".";
        string baseName = Path.GetFileNameWithoutExtension(v17RootPath);
        List<byte[]> groupBytes = new(rootPayloads.ReportedGroupCount);
        for (int groupIndex = 0; groupIndex < rootPayloads.ReportedGroupCount; groupIndex++)
        {
            string groupPath = Path.Combine(groupDirectory, $"{baseName}_{groupIndex:D3}.wmo");
            if (!File.Exists(groupPath))
                throw new FileNotFoundException($"Expected WMO group file '{groupPath}' was not found.", groupPath);

            groupBytes.Add(File.ReadAllBytes(groupPath));
        }

        byte[] converted = Convert(rootBytes, groupBytes, v17RootPath);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        File.WriteAllBytes(outputPath, converted);
    }

    public static byte[] Convert(byte[] v17RootBytes, IReadOnlyList<byte[]> groupBytes, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(v17RootBytes);
        ArgumentNullException.ThrowIfNull(groupBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        RootChunkPayloads rootPayloads = ParseRootPayloads(v17RootBytes, sourcePath);
        if (rootPayloads.ReportedGroupCount != groupBytes.Count)
        {
            throw new InvalidDataException(
                $"WMO root reports {rootPayloads.ReportedGroupCount} groups, but {groupBytes.Count} group payloads were supplied.");
        }

        List<byte[]> convertedGroups = new(groupBytes.Count);
        for (int groupIndex = 0; groupIndex < groupBytes.Count; groupIndex++)
            convertedGroups.Add(ConvertGroupPayload(groupBytes[groupIndex], $"{sourcePath}#{groupIndex:D3}"));

        return BuildV14Root(rootPayloads.PayloadsById, convertedGroups);
    }

    private static RootChunkPayloads ParseRootPayloads(byte[] rootBytes, string sourcePath)
    {
        using MemoryStream stream = new(rootBytes, writable: false);
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        if (version != 17)
            throw new InvalidDataException($"WMO root version '{version?.ToString() ?? "unknown"}' is not supported. Expected 17.");

        byte[] mohdPayload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohdPayload.Length < 64)
            throw new InvalidDataException($"MOHD payload is too short ({mohdPayload.Length} bytes). Expected at least 64 bytes.");

        Dictionary<FourCC, byte[]> payloadsById = [];
        payloadsById[WmoChunkIds.Mohd] = mohdPayload.ToArray();

        foreach (FourCC chunkId in RootChunkOrder.Where(static id => id != WmoChunkIds.Mohd))
        {
            byte[]? payload = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, chunkId);
            if (payload is null || payload.Length == 0)
                continue;

            payloadsById[chunkId] = chunkId == WmoChunkIds.Momt
                ? DownconvertMomtPayload(payload)
                : payload;
        }

        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohdPayload.AsSpan(4, 4)));
        return new RootChunkPayloads(reportedGroupCount, payloadsById);
    }

    private static byte[] ConvertGroupPayload(byte[] groupBytes, string sourcePath)
    {
        using MemoryStream stream = new(groupBytes, writable: false);
        (uint? version, byte[] mogpPayload) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        if (version is null || version < 16 || version > 17)
            throw new InvalidDataException($"WMO group version '{version?.ToString() ?? "unknown"}' is not supported. Expected 16 or 17.");

        if (mogpPayload.Length < 0x3C)
            throw new InvalidDataException($"MOGP payload is too short ({mogpPayload.Length} bytes). Expected at least 60 bytes.");

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogpPayload);

        byte[]? mopyPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Mopy);
        byte[]? indexPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Movi)
            ?? WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Moin);
        byte[]? movtPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Movt);
        byte[]? monrPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Monr);
        byte[]? motvPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Motv);
        byte[]? mobaPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Moba);

        if (mopyPayload is null || indexPayload is null || movtPayload is null || monrPayload is null || mobaPayload is null)
            throw new InvalidDataException("WMO group is missing one or more required subchunks for v17->v14 conversion.");

        byte[] legacyHeader = new byte[0x44];
        mogpPayload.AsSpan(0, Math.Min(0x3C, mogpPayload.Length)).CopyTo(legacyHeader);

        using MemoryStream payloadStream = new();
        using BinaryWriter writer = new(payloadStream);
        writer.Write(legacyHeader);
        WriteChunk(writer, WmoChunkIds.Mopy, DownconvertMopyPayload(mopyPayload, version.Value));
        WriteChunk(writer, WmoChunkIds.Movi, indexPayload);
        WriteChunk(writer, WmoChunkIds.Movt, movtPayload);
        WriteChunk(writer, WmoChunkIds.Monr, monrPayload);
        if (motvPayload is not null)
            WriteChunk(writer, WmoChunkIds.Motv, motvPayload);
        WriteChunk(writer, WmoChunkIds.Moba, DownconvertMobaPayload(mobaPayload, version.Value));

        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Molr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Modr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mobn);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mobr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mocv);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mliq);

        return payloadStream.ToArray();
    }

    private static void WriteOptionalGroupChunk(BinaryWriter writer, byte[] mogpPayload, int headerSizeBytes, FourCC chunkId)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, chunkId);
        if (payload is { Length: > 0 })
            WriteChunk(writer, chunkId, payload);
    }

    private static byte[] DownconvertMomtPayload(byte[] payload)
    {
        const int sourceEntrySize = 64;
        const int targetEntrySize = 48;

        if (payload.Length == 0 || payload.Length % sourceEntrySize != 0)
            return payload;

        int entryCount = payload.Length / sourceEntrySize;
        byte[] converted = new byte[entryCount * targetEntrySize];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            payload.AsSpan(entryIndex * sourceEntrySize, targetEntrySize)
                .CopyTo(converted.AsSpan(entryIndex * targetEntrySize, targetEntrySize));
        }

        return converted;
    }

    private static byte[] DownconvertMopyPayload(byte[] payload, uint version)
    {
        if (payload.Length == 0 || version <= 16)
            return payload;

        if (payload.Length % 2 != 0)
            throw new InvalidDataException($"MOPY payload size {payload.Length} is not divisible by 2.");

        int faceCount = payload.Length / 2;
        byte[] converted = new byte[faceCount * 4];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int sourceOffset = faceIndex * 2;
            int targetOffset = faceIndex * 4;
            converted[targetOffset] = payload[sourceOffset];
            converted[targetOffset + 1] = payload[sourceOffset + 1];
        }

        return converted;
    }

    private static byte[] DownconvertMobaPayload(byte[] payload, uint version)
    {
        const int batchEntrySize = 24;

        if (payload.Length == 0 || version <= 16)
            return payload;

        if (payload.Length % batchEntrySize != 0)
            throw new InvalidDataException($"MOBA payload size {payload.Length} is not divisible by {batchEntrySize}.");

        int batchCount = payload.Length / batchEntrySize;
        byte[] converted = new byte[payload.Length];
        for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            int sourceOffset = batchIndex * batchEntrySize;
            int targetOffset = batchIndex * batchEntrySize;

            uint startIndex = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(sourceOffset + 12, 4));
            if (startIndex > ushort.MaxValue)
            {
                throw new InvalidDataException(
                    $"MOBA batch {batchIndex} firstIndex {startIndex} exceeds the legacy ushort range.");
            }

            ushort indexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 16, 2));
            ushort startVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 18, 2));
            ushort endVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 20, 2));
            byte flags = payload[sourceOffset + 22];
            byte materialId = payload[sourceOffset + 23];

            converted[targetOffset] = 0;
            converted[targetOffset + 1] = materialId;
            payload.AsSpan(sourceOffset, 12).CopyTo(converted.AsSpan(targetOffset + 2, 12));
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 14, 2), checked((ushort)startIndex));
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 16, 2), indexCount);
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 18, 2), startVertex);
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 20, 2), endVertex);
            converted[targetOffset + 22] = flags;
            converted[targetOffset + 23] = 0;
        }

        return converted;
    }

    private static byte[] BuildV14Root(IReadOnlyDictionary<FourCC, byte[]> rootChunkPayloads, IReadOnlyList<byte[]> groupPayloads)
    {
        using MemoryStream rootStream = new();
        using BinaryWriter writer = new(rootStream);

        WriteChunk(writer, WmoChunkIds.Mver, BitConverter.GetBytes((uint)14));

        using MemoryStream momoStream = new();
        using BinaryWriter momoWriter = new(momoStream);
        foreach (FourCC chunkId in RootChunkOrder)
        {
            if (rootChunkPayloads.TryGetValue(chunkId, out byte[]? payload) && payload.Length > 0)
                WriteChunk(momoWriter, chunkId, payload);
        }

        foreach (byte[] groupPayload in groupPayloads)
            WriteChunk(momoWriter, WmoChunkIds.Mogp, groupPayload);

        WriteChunk(writer, WmoChunkIds.Momo, momoStream.ToArray());
        writer.Flush();
        return rootStream.ToArray();
    }

    private static void WriteChunk(BinaryWriter writer, FourCC chunkId, byte[] payload)
    {
        writer.Write(chunkId.ToFileBytes());
        writer.Write(payload.Length);
        writer.Write(payload);
    }

    private sealed record RootChunkPayloads(int ReportedGroupCount, IReadOnlyDictionary<FourCC, byte[]> PayloadsById);
}