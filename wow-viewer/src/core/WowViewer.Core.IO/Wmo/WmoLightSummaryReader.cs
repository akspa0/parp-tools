using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoLightSummaryReader
{
    public static WmoLightSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoLightSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Molt, out uint? version);
        IReadOnlyList<WmoLightDetail> details = WmoLightReaderCommon.ReadDetails(payload, sourcePath, version);
        int entryCount = details.Count;
        HashSet<byte> types = [];
        int attenuatedCount = 0;
        float minIntensity = 0f;
        float maxIntensity = 0f;
        float minAttenStart = 0f;
        float maxAttenStart = 0f;
        float maxAttenEnd = 0f;
        HashSet<ushort> headerFlagsWords = [];
        int nonZeroHeaderFlagsWordCount = 0;
        ushort minHeaderFlagsWord = 0;
        ushort maxHeaderFlagsWord = 0;
        int rotationEntryCount = 0;
        int nonIdentityRotationCount = 0;
        float minRotationLength = 0f;
        float maxRotationLength = 0f;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;
        bool hasStandardLayoutEntries = entryCount > 0 && details[0].HeaderFlagsWord.HasValue;

        if (entryCount > 0)
        {
            minIntensity = float.MaxValue;
            maxIntensity = float.MinValue;
            minAttenStart = float.MaxValue;
            maxAttenStart = float.MinValue;
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            if (hasStandardLayoutEntries)
            {
                minHeaderFlagsWord = ushort.MaxValue;
                maxHeaderFlagsWord = ushort.MinValue;
                minRotationLength = float.MaxValue;
                maxRotationLength = float.MinValue;
            }
        }

        foreach (WmoLightDetail detail in details)
        {
            types.Add(detail.LightType);
            if (detail.UsesAttenuation)
                attenuatedCount++;

            minIntensity = Math.Min(minIntensity, detail.Intensity);
            maxIntensity = Math.Max(maxIntensity, detail.Intensity);
            minAttenStart = Math.Min(minAttenStart, detail.AttenStart);
            maxAttenStart = Math.Max(maxAttenStart, detail.AttenStart);
            maxAttenEnd = Math.Max(maxAttenEnd, detail.AttenEnd);

            if (detail.HeaderFlagsWord is ushort headerFlagsWord && detail.RotationLength is float rotationLength)
            {
                headerFlagsWords.Add(headerFlagsWord);
                if (headerFlagsWord != 0)
                    nonZeroHeaderFlagsWordCount++;

                minHeaderFlagsWord = Math.Min(minHeaderFlagsWord, headerFlagsWord);
                maxHeaderFlagsWord = Math.Max(maxHeaderFlagsWord, headerFlagsWord);
                rotationEntryCount++;
                if (detail.Rotation is Quaternion rotation && !WmoLightReaderCommon.IsIdentityRotation(rotation))
                    nonIdentityRotationCount++;

                minRotationLength = Math.Min(minRotationLength, rotationLength);
                maxRotationLength = Math.Max(maxRotationLength, rotationLength);
            }

            boundsMin = Vector3.Min(boundsMin, detail.Position);
            boundsMax = Vector3.Max(boundsMax, detail.Position);
        }

        return new WmoLightSummary(
            sourcePath,
            version,
            payload.Length,
            entryCount,
            types.Count,
            attenuatedCount,
            minIntensity,
            maxIntensity,
            minAttenStart,
            maxAttenStart,
            maxAttenEnd,
            nonZeroHeaderFlagsWordCount,
            headerFlagsWords.Count,
            minHeaderFlagsWord,
            maxHeaderFlagsWord,
            rotationEntryCount,
            nonIdentityRotationCount,
            minRotationLength,
            maxRotationLength,
            boundsMin,
            boundsMax);
    }
}
