using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupUvSummaryReader
{
    private const int UvStride = 8;

    public static WmoGroupUvSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupUvSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);

        bool foundPrimary = false;
        int primaryPayloadSizeBytes = 0;
        int primaryUvCount = 0;
        float minU = 0f;
        float maxU = 0f;
        float minV = 0f;
        float maxV = 0f;
        int additionalUvSetCount = 0;
        int totalAdditionalUvCount = 0;
        int maxAdditionalUvCount = 0;

        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Motv)
                continue;

            int uvCount = checked((int)header.Size / UvStride);
            if (!foundPrimary)
            {
                foundPrimary = true;
                primaryPayloadSizeBytes = checked((int)header.Size);
                primaryUvCount = uvCount;
                minU = float.MaxValue;
                maxU = float.MinValue;
                minV = float.MaxValue;
                maxV = float.MinValue;

                for (int index = 0; index < uvCount; index++)
                {
                    int offset = dataOffset + index * UvStride;
                    float u = BitConverter.ToSingle(mogp, offset);
                    float v = BitConverter.ToSingle(mogp, offset + 4);
                    minU = Math.Min(minU, u);
                    maxU = Math.Max(maxU, u);
                    minV = Math.Min(minV, v);
                    maxV = Math.Max(maxV, v);
                }

                if (primaryUvCount == 0)
                    minU = maxU = minV = maxV = 0f;

                continue;
            }

            additionalUvSetCount++;
            totalAdditionalUvCount += uvCount;
            maxAdditionalUvCount = Math.Max(maxAdditionalUvCount, uvCount);
        }

        if (!foundPrimary)
            throw new InvalidDataException("WMO group UV summary requires at least one MOTV subchunk.");

        return new WmoGroupUvSummary(
            sourcePath,
            version,
            primaryPayloadSizeBytes,
            primaryUvCount,
            minU,
            maxU,
            minV,
            maxV,
            additionalUvSetCount,
            totalAdditionalUvCount,
            maxAdditionalUvCount);
    }
}
