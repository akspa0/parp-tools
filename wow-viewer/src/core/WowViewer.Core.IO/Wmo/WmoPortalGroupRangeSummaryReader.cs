using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalGroupRangeSummaryReader
{
    private const int PortalRefEntrySize = 8;

    public static WmoPortalGroupRangeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalGroupRangeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        byte[] mogi = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogi);
        byte[] mopr = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mopr);
        int entrySize = WmoRootReaderCommon.InferMogiEntrySize(mogi, WmoRootReaderCommon.ReadReportedGroupCount(mohd));
        if (entrySize <= 0 || mogi.Length % entrySize != 0)
            throw new InvalidDataException($"MOGI payload size {mogi.Length} is not compatible with inferred entry size {entrySize}.");
        if (mopr.Length % PortalRefEntrySize != 0)
            throw new InvalidDataException($"MOPR payload size {mopr.Length} is not divisible by {PortalRefEntrySize}.");

        int groupCount = mogi.Length / entrySize;
        int refCount = mopr.Length / PortalRefEntrySize;
        int coveredRefCount = 0;
        int outOfRangeRefCount = 0;
        int maxGroupIndex = 0;
        HashSet<int> distinctGroupRefs = [];

        for (int i = 0; i < refCount; i++)
        {
            int offset = i * PortalRefEntrySize;
            int groupIndex = BinaryPrimitives.ReadUInt16LittleEndian(mopr.AsSpan(offset + 2, 2));
            maxGroupIndex = Math.Max(maxGroupIndex, groupIndex);
            distinctGroupRefs.Add(groupIndex);

            if (groupIndex < groupCount)
                coveredRefCount++;
            else
                outOfRangeRefCount++;
        }

        return new WmoPortalGroupRangeSummary(sourcePath, version, refCount, groupCount, coveredRefCount, outOfRangeRefCount, distinctGroupRefs.Count, maxGroupIndex);
    }
}
