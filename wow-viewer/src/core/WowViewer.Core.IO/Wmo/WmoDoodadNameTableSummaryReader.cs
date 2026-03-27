using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadNameTableSummaryReader
{
    public static WmoDoodadNameTableSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadNameTableSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modn);
        int nameCount = 0;
        int longestEntryLength = 0;
        int maxOffset = 0;
        HashSet<string> extensions = [];
        int mdxEntryCount = 0;
        int m2EntryCount = 0;

        int currentStart = -1;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (currentStart >= 0)
                {
                    ProcessEntry(payload, currentStart, index - currentStart, ref nameCount, ref longestEntryLength, ref maxOffset, ref mdxEntryCount, ref m2EntryCount, extensions);
                    currentStart = -1;
                }

                continue;
            }

            if (currentStart < 0)
                currentStart = index;
        }

        if (currentStart >= 0)
            ProcessEntry(payload, currentStart, payload.Length - currentStart, ref nameCount, ref longestEntryLength, ref maxOffset, ref mdxEntryCount, ref m2EntryCount, extensions);

        return new WmoDoodadNameTableSummary(
            sourcePath,
            version,
            payload.Length,
            nameCount,
            longestEntryLength,
            maxOffset,
            distinctExtensionCount: extensions.Count,
            mdxEntryCount,
            m2EntryCount);
    }

    private static void ProcessEntry(byte[] payload, int offset, int length, ref int nameCount, ref int longestEntryLength, ref int maxOffset, ref int mdxEntryCount, ref int m2EntryCount, HashSet<string> extensions)
    {
        string entry = System.Text.Encoding.UTF8.GetString(payload, offset, length);
        nameCount++;
        longestEntryLength = Math.Max(longestEntryLength, length);
        maxOffset = Math.Max(maxOffset, offset);

        string extension = Path.GetExtension(entry).ToLowerInvariant();
        if (!string.IsNullOrEmpty(extension))
            extensions.Add(extension);

        if (string.Equals(extension, ".mdx", StringComparison.OrdinalIgnoreCase))
            mdxEntryCount++;
        else if (string.Equals(extension, ".m2", StringComparison.OrdinalIgnoreCase))
            m2EntryCount++;
    }

}
