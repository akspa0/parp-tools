using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupBspNodeSummaryReader
{
    private const int NodeStride = 16;
    private const ushort LeafFlag = 0x4;
    private const short NoChild = -1;

    public static WmoGroupBspNodeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupBspNodeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupBspNodeSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? mobnPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mobn);
        if (mobnPayload is null)
            throw new InvalidDataException("WMO group BSP-node summary requires a MOBN subchunk.");

        if (mobnPayload.Length % NodeStride != 0)
            throw new InvalidDataException($"MOBN payload size {mobnPayload.Length} is not divisible by {NodeStride}.");

        int nodeCount = mobnPayload.Length / NodeStride;
        int leafNodeCount = 0;
        int childReferenceCount = 0;
        int noChildReferenceCount = 0;
        int outOfRangeChildReferenceCount = 0;
        int minFaceCount = 0;
        int maxFaceCount = 0;
        int minFaceStart = 0;
        int maxFaceStart = 0;
        int maxFaceEnd = 0;
        float minPlaneDistance = 0f;
        float maxPlaneDistance = 0f;
        if (nodeCount > 0)
        {
            minFaceCount = int.MaxValue;
            minFaceStart = int.MaxValue;
            minPlaneDistance = float.PositiveInfinity;
            maxPlaneDistance = float.NegativeInfinity;
        }

        for (int index = 0; index < nodeCount; index++)
        {
            ReadOnlySpan<byte> node = mobnPayload.AsSpan(index * NodeStride, NodeStride);
            ushort flags = BinaryPrimitives.ReadUInt16LittleEndian(node[0..2]);
            short negativeChild = BinaryPrimitives.ReadInt16LittleEndian(node[2..4]);
            short positiveChild = BinaryPrimitives.ReadInt16LittleEndian(node[4..6]);
            int faceCount = BinaryPrimitives.ReadUInt16LittleEndian(node[6..8]);
            int faceStart = BinaryPrimitives.ReadInt32LittleEndian(node[8..12]);
            float planeDistance = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(node[12..16]));

            if ((flags & LeafFlag) != 0)
                leafNodeCount++;

            TrackChild(negativeChild, nodeCount, ref childReferenceCount, ref noChildReferenceCount, ref outOfRangeChildReferenceCount);
            TrackChild(positiveChild, nodeCount, ref childReferenceCount, ref noChildReferenceCount, ref outOfRangeChildReferenceCount);

            minFaceCount = Math.Min(minFaceCount, faceCount);
            maxFaceCount = Math.Max(maxFaceCount, faceCount);
            minFaceStart = Math.Min(minFaceStart, faceStart);
            maxFaceStart = Math.Max(maxFaceStart, faceStart);
            maxFaceEnd = Math.Max(maxFaceEnd, faceStart + faceCount);
            minPlaneDistance = Math.Min(minPlaneDistance, planeDistance);
            maxPlaneDistance = Math.Max(maxPlaneDistance, planeDistance);
        }

        if (nodeCount == 0)
        {
            minPlaneDistance = 0f;
            maxPlaneDistance = 0f;
        }

        return new WmoGroupBspNodeSummary(
            sourcePath,
            version,
            mobnPayload.Length,
            nodeCount,
            leafNodeCount,
            nodeCount - leafNodeCount,
            childReferenceCount,
            noChildReferenceCount,
            outOfRangeChildReferenceCount,
            minFaceCount == int.MaxValue ? 0 : minFaceCount,
            maxFaceCount,
            minFaceStart == int.MaxValue ? 0 : minFaceStart,
            maxFaceStart,
            maxFaceEnd,
            minPlaneDistance,
            maxPlaneDistance);
    }

    private static void TrackChild(short childIndex, int nodeCount, ref int childReferenceCount, ref int noChildReferenceCount, ref int outOfRangeChildReferenceCount)
    {
        if (childIndex == NoChild)
        {
            noChildReferenceCount++;
            return;
        }

        if (childIndex >= 0)
        {
            childReferenceCount++;
            if (childIndex >= nodeCount)
                outOfRangeChildReferenceCount++;
            return;
        }

        noChildReferenceCount++;
    }
}