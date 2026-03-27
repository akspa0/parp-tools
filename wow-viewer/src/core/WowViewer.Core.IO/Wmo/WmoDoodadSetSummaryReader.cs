using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadSetSummaryReader
{
    private const int ModsEntrySize = 32;
    private const int NameBytes = 20;

    public static WmoDoodadSetSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadSetSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mods);
        if (payload.Length % ModsEntrySize != 0)
            throw new InvalidDataException($"MODS payload size {payload.Length} is not divisible by {ModsEntrySize}.");

        int entryCount = payload.Length / ModsEntrySize;
        int nonEmptySetCount = 0;
        int longestNameLength = 0;
        int totalDoodadRefs = 0;
        int maxStartIndex = 0;
        int maxRangeEnd = 0;

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * ModsEntrySize;
            int length = 0;
            while (length < NameBytes && payload[offset + length] != 0)
                length++;

            uint startIndex = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 20, 4));
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 24, 4));

            longestNameLength = Math.Max(longestNameLength, length);
            if (count > 0)
                nonEmptySetCount++;

            totalDoodadRefs += checked((int)count);
            maxStartIndex = Math.Max(maxStartIndex, checked((int)startIndex));
            maxRangeEnd = Math.Max(maxRangeEnd, checked((int)(startIndex + count)));
        }

        return new WmoDoodadSetSummary(
            sourcePath,
            version,
            payload.Length,
            entryCount,
            nonEmptySetCount,
            longestNameLength,
            totalDoodadRefs,
            maxStartIndex,
            maxRangeEnd);
    }

}
