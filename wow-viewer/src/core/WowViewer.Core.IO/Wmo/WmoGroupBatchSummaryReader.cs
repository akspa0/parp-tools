using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupBatchSummaryReader
{
    private const int BatchEntrySize = 24;

    public static WmoGroupBatchSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupBatchSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupBatchSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        ArgumentNullException.ThrowIfNull(mogp);

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? mobaPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Moba);
        if (mobaPayload is null)
            throw new InvalidDataException("WMO group batch summary requires a MOBA subchunk.");

        if (mobaPayload.Length % BatchEntrySize != 0)
            throw new InvalidDataException($"MOBA payload size {mobaPayload.Length} is not divisible by {BatchEntrySize}.");

        int entryCount = mobaPayload.Length / BatchEntrySize;
        HashSet<byte> materialIds = [];
        bool sawMaterialId = false;
        bool allMaterialBytesAreSentinel = true;
        int totalIndexCount = 0;
        int minFirstIndex = int.MaxValue;
        int maxFirstIndex = 0;
        int maxIndexEnd = 0;
        int flaggedBatchCount = 0;

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * BatchEntrySize;
            byte materialId = mobaPayload[offset + 1];
            ushort firstIndex = BinaryPrimitives.ReadUInt16LittleEndian(mobaPayload.AsSpan(offset + 14, 2));
            ushort indexCount = BinaryPrimitives.ReadUInt16LittleEndian(mobaPayload.AsSpan(offset + 16, 2));
            byte flags = mobaPayload[offset + 22];

            if (materialId != byte.MaxValue)
            {
                sawMaterialId = true;
                allMaterialBytesAreSentinel = false;
                materialIds.Add(materialId);
            }
            else if (version != 16)
            {
                allMaterialBytesAreSentinel = false;
            }

            totalIndexCount += indexCount;
            minFirstIndex = Math.Min(minFirstIndex, firstIndex);
            maxFirstIndex = Math.Max(maxFirstIndex, firstIndex);
            maxIndexEnd = Math.Max(maxIndexEnd, firstIndex + indexCount);
            if (flags != 0)
                flaggedBatchCount++;
        }

        bool hasMaterialIds = entryCount > 0 && (sawMaterialId || !allMaterialBytesAreSentinel);
        if (entryCount == 0)
            minFirstIndex = 0;

        return new WmoGroupBatchSummary(
            sourcePath,
            version,
            mobaPayload.Length,
            entryCount,
            hasMaterialIds,
            distinctMaterialIdCount: materialIds.Count,
            highestMaterialId: materialIds.Count > 0 ? materialIds.Max() : 0,
            totalIndexCount,
            minFirstIndex,
            maxFirstIndex,
            maxIndexEnd,
            flaggedBatchCount);
    }
}
