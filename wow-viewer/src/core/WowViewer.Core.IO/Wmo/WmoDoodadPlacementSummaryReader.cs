using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadPlacementSummaryReader
{
    private const int ModdEntrySize = 40;

    public static WmoDoodadPlacementSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadPlacementSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modd);
        if (payload.Length % ModdEntrySize != 0)
            throw new InvalidDataException($"MODD payload size {payload.Length} is not divisible by {ModdEntrySize}.");

        int entryCount = payload.Length / ModdEntrySize;
        HashSet<uint> nameIndices = [];
        int maxNameIndex = 0;
        float minScale = 0f;
        float maxScale = 0f;
        int minAlpha = 0;
        int maxAlpha = 0;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;

        if (entryCount > 0)
        {
            minScale = float.MaxValue;
            maxScale = float.MinValue;
            minAlpha = byte.MaxValue;
            maxAlpha = byte.MinValue;
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * ModdEntrySize;
            uint nameIndex = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset, 4)) & 0x00FFFFFFu;
            float x = BitConverter.ToSingle(payload, offset + 4);
            float y = BitConverter.ToSingle(payload, offset + 8);
            float z = BitConverter.ToSingle(payload, offset + 12);
            float scale = BitConverter.ToSingle(payload, offset + 32);
            uint color = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 36, 4));
            int alpha = (int)((color >> 24) & 0xFF);

            nameIndices.Add(nameIndex);
            maxNameIndex = Math.Max(maxNameIndex, checked((int)nameIndex));
            minScale = Math.Min(minScale, scale);
            maxScale = Math.Max(maxScale, scale);
            minAlpha = Math.Min(minAlpha, alpha);
            maxAlpha = Math.Max(maxAlpha, alpha);
            boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
            boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
        }

        return new WmoDoodadPlacementSummary(
            sourcePath,
            version,
            payload.Length,
            entryCount,
            distinctNameIndexCount: nameIndices.Count,
            maxNameIndex,
            minScale,
            maxScale,
            minAlpha,
            maxAlpha,
            boundsMin,
            boundsMax);
    }

}
