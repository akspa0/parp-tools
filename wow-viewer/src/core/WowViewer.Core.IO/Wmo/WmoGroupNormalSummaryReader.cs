using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupNormalSummaryReader
{
    private const int NormalStride = 12;
    private const float UnitTolerance = 0.05f;

    public static WmoGroupNormalSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupNormalSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);

        byte[]? monrPayload = null;
        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Monr)
                continue;

            monrPayload = mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            break;
        }

        if (monrPayload is null)
            throw new InvalidDataException("WMO group normal summary requires a MONR subchunk.");

        if (monrPayload.Length % NormalStride != 0)
            throw new InvalidDataException($"MONR payload size {monrPayload.Length} is not divisible by {NormalStride}.");

        int normalCount = monrPayload.Length / NormalStride;
        float minX = 0f;
        float maxX = 0f;
        float minY = 0f;
        float maxY = 0f;
        float minZ = 0f;
        float maxZ = 0f;
        float minLength = 0f;
        float maxLength = 0f;
        float averageLength = 0f;
        int nearUnitCount = 0;

        if (normalCount > 0)
        {
            minX = minY = minZ = minLength = float.MaxValue;
            maxX = maxY = maxZ = maxLength = float.MinValue;
            float totalLength = 0f;

            for (int index = 0; index < normalCount; index++)
            {
                int offset = index * NormalStride;
                float x = BitConverter.ToSingle(monrPayload, offset);
                float y = BitConverter.ToSingle(monrPayload, offset + 4);
                float z = BitConverter.ToSingle(monrPayload, offset + 8);
                float length = new Vector3(x, y, z).Length();

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
                minLength = Math.Min(minLength, length);
                maxLength = Math.Max(maxLength, length);
                totalLength += length;
                if (Math.Abs(length - 1f) <= UnitTolerance)
                    nearUnitCount++;
            }

            averageLength = totalLength / normalCount;
        }

        return new WmoGroupNormalSummary(
            sourcePath,
            version,
            monrPayload.Length,
            normalCount,
            minX,
            maxX,
            minY,
            maxY,
            minZ,
            maxZ,
            minLength,
            maxLength,
            averageLength,
            nearUnitCount);
    }
}
