using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

internal static class MapSummaryReaderCommon
{
    public static byte[]? ReadChunkPayload(Stream stream, MapFileSummary fileSummary, FourCC id)
    {
        MapChunkLocation chunk = default;
        bool found = false;
        foreach (MapChunkLocation location in fileSummary.Chunks)
        {
            if (location.Id != id)
                continue;

            chunk = location;
            found = true;
            break;
        }

        if (!found)
            return null;

        if (!stream.CanSeek)
            throw new ArgumentException("Chunk payload reading requires a seekable stream.", nameof(stream));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    public static byte[]? ReadFirstAvailableChunkPayload(Stream stream, MapFileSummary fileSummary, IReadOnlyList<FourCC> ids)
    {
        foreach (FourCC id in ids)
        {
            byte[]? payload = ReadChunkPayload(stream, fileSummary, id);
            if (payload is { Length: > 0 })
                return payload;
        }

        return null;
    }

    public static int CountStringEntries(byte[]? payload)
    {
        if (payload is not { Length: > 0 })
            return 0;

        int count = 0;
        bool sawText = false;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (sawText)
                {
                    count++;
                    sawText = false;
                }

                continue;
            }

            sawText = true;
        }

        if (sawText)
            count++;

        return count;
    }

    public static int CountPlacements(byte[]? payload, int stride)
    {
        if (payload is not { Length: > 0 } || stride <= 0)
            return 0;

        return payload.Length / stride;
    }
}
