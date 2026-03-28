using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoLightSummaryReader
{
    private const int LegacyEntrySize = 32;
    private const int StandardEntrySize = 48;

    public static WmoLightSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoLightSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Molt, out uint? version);
        int entrySize = InferEntrySize(payload.Length, version);
        if (payload.Length % entrySize != 0)
            throw new InvalidDataException($"MOLT payload size {payload.Length} is not divisible by inferred entry size {entrySize}.");

        int entryCount = payload.Length / entrySize;
        HashSet<byte> types = [];
        int attenuatedCount = 0;
        float minIntensity = 0f;
        float maxIntensity = 0f;
        float maxAttenEnd = 0f;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;

        if (entryCount > 0)
        {
            minIntensity = float.MaxValue;
            maxIntensity = float.MinValue;
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * entrySize;
            byte type = payload[offset];
            bool useAtten = payload[offset + 1] != 0;
            Vector3 position = new(BitConverter.ToSingle(payload, offset + 8), BitConverter.ToSingle(payload, offset + 12), BitConverter.ToSingle(payload, offset + 16));
            float intensity = BitConverter.ToSingle(payload, offset + 20);
            float attenEnd = BitConverter.ToSingle(payload, offset + 28);

            types.Add(type);
            if (useAtten)
                attenuatedCount++;

            minIntensity = Math.Min(minIntensity, intensity);
            maxIntensity = Math.Max(maxIntensity, intensity);
            maxAttenEnd = Math.Max(maxAttenEnd, attenEnd);
            boundsMin = Vector3.Min(boundsMin, position);
            boundsMax = Vector3.Max(boundsMax, position);
        }

        return new WmoLightSummary(sourcePath, version, payload.Length, entryCount, types.Count, attenuatedCount, minIntensity, maxIntensity, maxAttenEnd, boundsMin, boundsMax);
    }

    private static int InferEntrySize(int payloadLength, uint? version)
    {
        if (payloadLength == 0)
            return version is not null && version <= 14 ? LegacyEntrySize : StandardEntrySize;

        if (version is not null && version <= 14)
            return LegacyEntrySize;

        if (version is not null && version >= 17)
            return StandardEntrySize;

        bool divisibleByLegacy = payloadLength % LegacyEntrySize == 0;
        bool divisibleByStandard = payloadLength % StandardEntrySize == 0;

        if (divisibleByStandard && !divisibleByLegacy)
            return StandardEntrySize;

        if (divisibleByLegacy && !divisibleByStandard)
            return LegacyEntrySize;

        if (divisibleByStandard)
            return StandardEntrySize;

        if (divisibleByLegacy)
            return LegacyEntrySize;

        return StandardEntrySize;
    }
}
