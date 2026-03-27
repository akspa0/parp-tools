using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupVertexColorSummaryReader
{
    private const int ColorStride = 4;

    public static WmoGroupVertexColorSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupVertexColorSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);

        bool foundPrimary = false;
        int primaryPayloadSizeBytes = 0;
        int primaryColorCount = 0;
        int minRed = 0;
        int maxRed = 0;
        int minGreen = 0;
        int maxGreen = 0;
        int minBlue = 0;
        int maxBlue = 0;
        int minAlpha = 0;
        int maxAlpha = 0;
        int averageAlpha = 0;
        int additionalColorSetCount = 0;
        int totalAdditionalColorCount = 0;
        int maxAdditionalColorCount = 0;

        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Mocv)
                continue;

            int colorCount = checked((int)header.Size / ColorStride);
            if (!foundPrimary)
            {
                foundPrimary = true;
                primaryPayloadSizeBytes = checked((int)header.Size);
                primaryColorCount = colorCount;
                if (primaryColorCount > 0)
                {
                    minRed = minGreen = minBlue = minAlpha = byte.MaxValue;
                    maxRed = maxGreen = maxBlue = maxAlpha = byte.MinValue;
                    int totalAlpha = 0;
                    for (int index = 0; index < colorCount; index++)
                    {
                        int offset = dataOffset + index * ColorStride;
                        int blue = mogp[offset];
                        int green = mogp[offset + 1];
                        int red = mogp[offset + 2];
                        int alpha = mogp[offset + 3];

                        minRed = Math.Min(minRed, red);
                        maxRed = Math.Max(maxRed, red);
                        minGreen = Math.Min(minGreen, green);
                        maxGreen = Math.Max(maxGreen, green);
                        minBlue = Math.Min(minBlue, blue);
                        maxBlue = Math.Max(maxBlue, blue);
                        minAlpha = Math.Min(minAlpha, alpha);
                        maxAlpha = Math.Max(maxAlpha, alpha);
                        totalAlpha += alpha;
                    }

                    averageAlpha = totalAlpha / primaryColorCount;
                }

                continue;
            }

            additionalColorSetCount++;
            totalAdditionalColorCount += colorCount;
            maxAdditionalColorCount = Math.Max(maxAdditionalColorCount, colorCount);
        }

        if (!foundPrimary)
            throw new InvalidDataException("WMO group vertex-color summary requires at least one MOCV subchunk.");

        return new WmoGroupVertexColorSummary(
            sourcePath,
            version,
            primaryPayloadSizeBytes,
            primaryColorCount,
            minRed,
            maxRed,
            minGreen,
            maxGreen,
            minBlue,
            maxBlue,
            minAlpha,
            maxAlpha,
            averageAlpha,
            additionalColorSetCount,
            totalAdditionalColorCount,
            maxAdditionalColorCount);
    }
}
