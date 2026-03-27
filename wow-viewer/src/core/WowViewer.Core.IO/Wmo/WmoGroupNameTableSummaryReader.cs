using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupNameTableSummaryReader
{
    public static WmoGroupNameTableSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupNameTableSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogn);
        int count = 0;
        int longest = 0;
        int maxOffset = 0;
        int currentStart = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (currentStart >= 0)
                {
                    int length = index - currentStart;
                    count++;
                    longest = Math.Max(longest, length);
                    maxOffset = Math.Max(maxOffset, currentStart);
                    currentStart = -1;
                }

                continue;
            }

            if (currentStart < 0)
                currentStart = index;
        }

        if (currentStart >= 0)
        {
            count++;
            longest = Math.Max(longest, payload.Length - currentStart);
            maxOffset = Math.Max(maxOffset, currentStart);
        }

        return new WmoGroupNameTableSummary(sourcePath, version, payload.Length, count, longest, maxOffset);
    }

}
