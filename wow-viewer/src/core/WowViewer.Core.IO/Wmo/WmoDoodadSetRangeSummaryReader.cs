using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadSetRangeSummaryReader
{
    private const int ModsEntrySize = 32;
    private const int ModdEntrySize = 40;

    public static WmoDoodadSetRangeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadSetRangeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mods = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mods);
        byte[] modd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modd);
        if (mods.Length % ModsEntrySize != 0)
            throw new InvalidDataException($"MODS payload size {mods.Length} is not divisible by {ModsEntrySize}.");
        if (modd.Length % ModdEntrySize != 0)
            throw new InvalidDataException($"MODD payload size {modd.Length} is not divisible by {ModdEntrySize}.");

        int entryCount = mods.Length / ModsEntrySize;
        int placementCount = modd.Length / ModdEntrySize;
        int empty = 0;
        int covered = 0;
        int outOfRange = 0;
        int maxRangeEnd = 0;
        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * ModsEntrySize;
            int start = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mods.AsSpan(offset + 20, 4)));
            int count = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mods.AsSpan(offset + 24, 4)));
            int end = start + count;
            maxRangeEnd = Math.Max(maxRangeEnd, end);
            if (count == 0)
            {
                empty++;
                continue;
            }

            if (end <= placementCount)
                covered++;
            else
                outOfRange++;
        }

        return new WmoDoodadSetRangeSummary(sourcePath, version, entryCount, placementCount, empty, covered, outOfRange, maxRangeEnd);
    }
}
