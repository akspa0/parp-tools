using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalRefRangeSummaryReader
{
    private const int PortalInfoEntrySize = 20;
    private const int PortalRefEntrySize = 8;

    public static WmoPortalRefRangeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalRefRangeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mopt = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mopt);
        byte[] mopr = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mopr);
        if (mopt.Length % PortalInfoEntrySize != 0)
            throw new InvalidDataException($"MOPT payload size {mopt.Length} is not divisible by {PortalInfoEntrySize}.");
        if (mopr.Length % PortalRefEntrySize != 0)
            throw new InvalidDataException($"MOPR payload size {mopr.Length} is not divisible by {PortalRefEntrySize}.");

        int portalCount = mopt.Length / PortalInfoEntrySize;
        int refCount = mopr.Length / PortalRefEntrySize;
        int coveredRefCount = 0;
        int outOfRangeRefCount = 0;
        int maxPortalIndex = 0;
        HashSet<int> distinctPortalRefs = [];

        for (int i = 0; i < refCount; i++)
        {
            int offset = i * PortalRefEntrySize;
            int portalIndex = BinaryPrimitives.ReadUInt16LittleEndian(mopr.AsSpan(offset, 2));
            maxPortalIndex = Math.Max(maxPortalIndex, portalIndex);
            distinctPortalRefs.Add(portalIndex);

            if (portalIndex < portalCount)
                coveredRefCount++;
            else
                outOfRangeRefCount++;
        }

        return new WmoPortalRefRangeSummary(sourcePath, version, refCount, portalCount, coveredRefCount, outOfRangeRefCount, distinctPortalRefs.Count, maxPortalIndex);
    }
}
