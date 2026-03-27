using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoTextureTableSummaryReader
{
    public static WmoTextureTableSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoTextureTableSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Motx);
        int textureCount = 0;
        int longestEntryLength = 0;
        int maxOffset = 0;
        HashSet<string> extensions = [];
        int blpEntryCount = 0;

        int currentStart = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (currentStart >= 0)
                {
                    int length = index - currentStart;
                    string entry = System.Text.Encoding.UTF8.GetString(payload, currentStart, length);
                    textureCount++;
                    longestEntryLength = Math.Max(longestEntryLength, length);
                    maxOffset = Math.Max(maxOffset, currentStart);

                    string extension = Path.GetExtension(entry).ToLowerInvariant();
                    if (!string.IsNullOrEmpty(extension))
                        extensions.Add(extension);

                    if (string.Equals(extension, ".blp", StringComparison.OrdinalIgnoreCase))
                        blpEntryCount++;

                    currentStart = -1;
                }

                continue;
            }

            if (currentStart < 0)
                currentStart = index;
        }

        if (currentStart >= 0)
        {
            int length = payload.Length - currentStart;
            string entry = System.Text.Encoding.UTF8.GetString(payload, currentStart, length);
            textureCount++;
            longestEntryLength = Math.Max(longestEntryLength, length);
            maxOffset = Math.Max(maxOffset, currentStart);

            string extension = Path.GetExtension(entry).ToLowerInvariant();
            if (!string.IsNullOrEmpty(extension))
                extensions.Add(extension);

            if (string.Equals(extension, ".blp", StringComparison.OrdinalIgnoreCase))
                blpEntryCount++;
        }

        return new WmoTextureTableSummary(
            sourcePath,
            version,
            payload.Length,
            textureCount,
            longestEntryLength,
            maxOffset,
            distinctExtensionCount: extensions.Count,
            blpEntryCount);
    }
}
