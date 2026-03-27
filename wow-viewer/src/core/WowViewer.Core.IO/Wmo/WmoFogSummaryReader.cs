using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoFogSummaryReader
{
    private const int EntrySize = 48;

    public static WmoFogSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoFogSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mfog, out uint? version);
        if (payload.Length % EntrySize != 0)
            throw new InvalidDataException($"MFOG payload size {payload.Length} is not divisible by {EntrySize}.");

        int entryCount = payload.Length / EntrySize;
        int nonZeroFlagCount = 0;
        float minSmallRadius = 0f;
        float maxLargeRadius = 0f;
        float maxFogEnd = 0f;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;

        if (entryCount > 0)
        {
            minSmallRadius = float.MaxValue;
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * EntrySize;
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset, 4));
            Vector3 position = new(BitConverter.ToSingle(payload, offset + 4), BitConverter.ToSingle(payload, offset + 8), BitConverter.ToSingle(payload, offset + 12));
            float smallRadius = BitConverter.ToSingle(payload, offset + 16);
            float largeRadius = BitConverter.ToSingle(payload, offset + 20);
            float fogEnd = BitConverter.ToSingle(payload, offset + 24);

            if (flags != 0)
                nonZeroFlagCount++;

            minSmallRadius = Math.Min(minSmallRadius, smallRadius);
            maxLargeRadius = Math.Max(maxLargeRadius, largeRadius);
            maxFogEnd = Math.Max(maxFogEnd, fogEnd);
            boundsMin = Vector3.Min(boundsMin, position);
            boundsMax = Vector3.Max(boundsMax, position);
        }

        return new WmoFogSummary(sourcePath, version, payload.Length, entryCount, nonZeroFlagCount, minSmallRadius, maxLargeRadius, maxFogEnd, boundsMin, boundsMax);
    }
}
