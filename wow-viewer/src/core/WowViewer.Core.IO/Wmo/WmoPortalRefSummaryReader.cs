using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalRefSummaryReader
{
    private const int EntrySize = 8;

    public static WmoPortalRefSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalRefSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopr, out uint? version);
        if (payload.Length % EntrySize != 0)
            throw new InvalidDataException($"MOPR payload size {payload.Length} is not divisible by {EntrySize}.");

        int entryCount = payload.Length / EntrySize;
        HashSet<int> portalIndices = [];
        int maxGroupIndex = 0;
        int positive = 0;
        int negative = 0;
        int neutral = 0;

        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * EntrySize;
            int portalIndex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2));
            int groupIndex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2));
            short side = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset + 4, 2));
            portalIndices.Add(portalIndex);
            maxGroupIndex = Math.Max(maxGroupIndex, groupIndex);
            if (side > 0)
                positive++;
            else if (side < 0)
                negative++;
            else
                neutral++;
        }

        return new WmoPortalRefSummary(sourcePath, version, payload.Length, entryCount, portalIndices.Count, maxGroupIndex, positive, negative, neutral);
    }
}
