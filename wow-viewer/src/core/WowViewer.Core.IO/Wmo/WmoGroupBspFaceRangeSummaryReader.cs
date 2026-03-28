using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupBspFaceRangeSummaryReader
{
    private const int NodeStride = 16;
    private const int RefStride = 2;

    public static WmoGroupBspFaceRangeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupBspFaceRangeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupBspFaceRangeSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? mobnPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mobn);
        if (mobnPayload is null)
            throw new InvalidDataException("WMO group BSP-face range summary requires a MOBN subchunk.");

        byte[]? mobrPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mobr);
        if (mobrPayload is null)
            throw new InvalidDataException("WMO group BSP-face range summary requires a MOBR subchunk.");

        if (mobnPayload.Length % NodeStride != 0)
            throw new InvalidDataException($"MOBN payload size {mobnPayload.Length} is not divisible by {NodeStride}.");

        if (mobrPayload.Length % RefStride != 0)
            throw new InvalidDataException($"MOBR payload size {mobrPayload.Length} is not divisible by {RefStride}.");

        int nodeCount = mobnPayload.Length / NodeStride;
        int faceRefCount = mobrPayload.Length / RefStride;
        int zeroFaceNodeCount = 0;
        int coveredNodeCount = 0;
        int outOfRangeNodeCount = 0;
        int maxFaceEnd = 0;

        for (int index = 0; index < nodeCount; index++)
        {
            ReadOnlySpan<byte> node = mobnPayload.AsSpan(index * NodeStride, NodeStride);
            int faceCount = BinaryPrimitives.ReadUInt16LittleEndian(node[6..8]);
            int faceStart = BinaryPrimitives.ReadInt32LittleEndian(node[8..12]);
            int faceEnd = faceStart + faceCount;
            maxFaceEnd = Math.Max(maxFaceEnd, faceEnd);

            if (faceCount == 0)
            {
                zeroFaceNodeCount++;
                continue;
            }

            if (faceStart < 0 || faceEnd > faceRefCount)
                outOfRangeNodeCount++;
            else
                coveredNodeCount++;
        }

        return new WmoGroupBspFaceRangeSummary(
            sourcePath,
            version,
            nodeCount,
            faceRefCount,
            zeroFaceNodeCount,
            coveredNodeCount,
            outOfRangeNodeCount,
            maxFaceEnd);
    }
}