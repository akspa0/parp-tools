using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoV14ToV17Converter
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

    public static void Convert(string v14RootPath, string outputRootPath, string? outputGroupsDirectory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(v14RootPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputRootPath);

        byte[] rootBytes = File.ReadAllBytes(v14RootPath);
        SplitWmoResult converted = Convert(rootBytes, v14RootPath);

        string fullOutputRootPath = Path.GetFullPath(outputRootPath);
        string groupDirectory = Path.GetFullPath(outputGroupsDirectory ?? Path.GetDirectoryName(fullOutputRootPath) ?? ".");
        string baseName = Path.GetFileNameWithoutExtension(fullOutputRootPath);

        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputRootPath) ?? ".");
        Directory.CreateDirectory(groupDirectory);
        File.WriteAllBytes(fullOutputRootPath, converted.RootBytes);

        for (int groupIndex = 0; groupIndex < converted.GroupBytes.Count; groupIndex++)
        {
            string groupPath = Path.Combine(groupDirectory, $"{baseName}_{groupIndex:D3}.wmo");
            File.WriteAllBytes(groupPath, converted.GroupBytes[groupIndex]);
        }
    }

    public static SplitWmoResult Convert(byte[] v14RootBytes, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(v14RootBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        using MemoryStream stream = new(v14RootBytes, writable: false);
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        if (version != 14)
            throw new InvalidDataException($"WMO root version '{version?.ToString() ?? "unknown"}' is not supported. Expected 14.");

        byte[] mohdPayload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohdPayload.Length < 64)
            throw new InvalidDataException($"MOHD payload is too short ({mohdPayload.Length} bytes). Expected at least 64 bytes.");

        int reportedMaterialCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohdPayload.AsSpan(0, 4)));
        List<ChunkSpan> embeddedGroups = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        if (embeddedGroups.Count == 0)
            throw new InvalidDataException("WMO v14 root does not contain any embedded MOGP groups to split.");

        Dictionary<FourCC, byte[]> rootPayloads = [];
        rootPayloads[WmoChunkIds.Mohd] = UpdateMohdGroupCount(mohdPayload, embeddedGroups.Count);

        foreach (FourCC chunkId in RootChunkOrder.Where(static id => id != WmoChunkIds.Mohd))
        {
            byte[]? payload = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, chunkId);
            if (payload is null || payload.Length == 0)
                continue;

            rootPayloads[chunkId] = chunkId switch
            {
                _ when chunkId == WmoChunkIds.Momt => UpconvertMomtPayload(payload, reportedMaterialCount),
                _ when chunkId == WmoChunkIds.Mogi => UpconvertMogiPayload(payload, embeddedGroups.Count),
                _ => payload.ToArray(),
            };
        }

        List<byte[]> groupBytes = new(embeddedGroups.Count);
        for (int groupIndex = 0; groupIndex < embeddedGroups.Count; groupIndex++)
        {
            ChunkSpan groupChunk = embeddedGroups[groupIndex];
            byte[] legacyGroupPayload = WmoRootReaderCommon.ReadChunkPayload(stream, groupChunk);
            groupBytes.Add(BuildGroupFile(ConvertGroupPayload(legacyGroupPayload), 17));
        }

        return new SplitWmoResult(BuildRootFile(rootPayloads), groupBytes);
    }

    public sealed record SplitWmoResult(byte[] RootBytes, IReadOnlyList<byte[]> GroupBytes);

    private static byte[] UpdateMohdGroupCount(byte[] mohdPayload, int groupCount)
    {
        byte[] updated = mohdPayload.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(updated.AsSpan(4, 4), checked((uint)groupCount));
        return updated;
    }

    private static byte[] UpconvertMomtPayload(byte[] payload, int reportedMaterialCount)
    {
        int sourceEntrySize = InferMomtEntrySize(payload, reportedMaterialCount);
        if (sourceEntrySize <= 0 || sourceEntrySize == 64)
            return payload.ToArray();

        int entryCount = payload.Length / sourceEntrySize;
        byte[] converted = new byte[entryCount * 64];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            payload.AsSpan(entryIndex * sourceEntrySize, sourceEntrySize)
                .CopyTo(converted.AsSpan(entryIndex * 64, sourceEntrySize));
        }

        return converted;
    }

    private static int InferMomtEntrySize(byte[] payload, int reportedMaterialCount)
    {
        if (payload.Length == 0)
            return 0;

        if (reportedMaterialCount > 0)
        {
            if (payload.Length == reportedMaterialCount * 64)
                return 64;

            if (payload.Length == reportedMaterialCount * 48)
                return 48;

            if (payload.Length == reportedMaterialCount * 44)
                return 44;
        }

        if (payload.Length % 64 == 0)
            return 64;

        if (payload.Length % 48 == 0)
            return 48;

        if (payload.Length % 44 == 0)
            return 44;

        return 0;
    }

    private static byte[] UpconvertMogiPayload(byte[] payload, int reportedGroupCount)
    {
        int sourceEntrySize = WmoRootReaderCommon.InferMogiEntrySize(payload, reportedGroupCount);
        if (sourceEntrySize <= 0 || sourceEntrySize == 32)
            return payload.ToArray();

        if (sourceEntrySize != 40)
            throw new InvalidDataException($"Unsupported MOGI entry size {sourceEntrySize}.");

        int entryCount = payload.Length / sourceEntrySize;
        byte[] converted = new byte[entryCount * 32];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            int sourceOffset = entryIndex * sourceEntrySize;
            int targetOffset = entryIndex * 32;
            payload.AsSpan(sourceOffset + 8, 32).CopyTo(converted.AsSpan(targetOffset, 32));
        }

        return converted;
    }

    private static byte[] ConvertGroupPayload(byte[] legacyMogpPayload)
    {
        if (legacyMogpPayload.Length < WmoGroupReaderCommon.MinimumGroupHeaderSize)
        {
            throw new InvalidDataException(
                $"MOGP payload is too short ({legacyMogpPayload.Length} bytes). Expected at least {WmoGroupReaderCommon.MinimumGroupHeaderSize} bytes.");
        }

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(legacyMogpPayload);
        byte[] v17Header = new byte[WmoGroupReaderCommon.AlternateHeaderSize];
        legacyMogpPayload.AsSpan(0, Math.Min(v17Header.Length, legacyMogpPayload.Length)).CopyTo(v17Header);

        using MemoryStream payloadStream = new();
        using BinaryWriter writer = new(payloadStream);
        writer.Write(v17Header);

        foreach ((ChunkHeader header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(legacyMogpPayload, headerSizeBytes))
        {
            byte[] payload = legacyMogpPayload.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            FourCC chunkId = header.Id == WmoChunkIds.Moin ? WmoChunkIds.Movi : header.Id;
            if (header.Id == WmoChunkIds.Mopy)
                payload = UpconvertMopyPayload(payload);

            WriteChunk(writer, chunkId, payload);
        }

        return payloadStream.ToArray();
    }

    private static byte[] UpconvertMopyPayload(byte[] payload)
    {
        if (payload.Length == 0)
            return [];

        if (payload.Length % 4 != 0)
            return payload.ToArray();

        int faceCount = payload.Length / 4;
        byte[] converted = new byte[faceCount * 2];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int sourceOffset = faceIndex * 4;
            int targetOffset = faceIndex * 2;
            converted[targetOffset] = payload[sourceOffset];
            converted[targetOffset + 1] = payload[sourceOffset + 1];
        }

        return converted;
    }

    private static byte[] BuildRootFile(IReadOnlyDictionary<FourCC, byte[]> rootPayloads)
    {
        using MemoryStream rootStream = new();
        using BinaryWriter writer = new(rootStream);

        WriteChunk(writer, WmoChunkIds.Mver, BitConverter.GetBytes((uint)17));
        foreach (FourCC chunkId in RootChunkOrder)
        {
            if (rootPayloads.TryGetValue(chunkId, out byte[]? payload) && payload.Length > 0)
                WriteChunk(writer, chunkId, payload);
        }

        writer.Flush();
        return rootStream.ToArray();
    }

    private static byte[] BuildGroupFile(byte[] mogpPayload, uint version)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream);
        WriteChunk(writer, WmoChunkIds.Mver, BitConverter.GetBytes(version));
        WriteChunk(writer, WmoChunkIds.Mogp, mogpPayload);
        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteChunk(BinaryWriter writer, FourCC chunkId, byte[] payload)
    {
        writer.Write(chunkId.ToFileBytes());
        writer.Write(payload.Length);
        writer.Write(payload);
    }
}