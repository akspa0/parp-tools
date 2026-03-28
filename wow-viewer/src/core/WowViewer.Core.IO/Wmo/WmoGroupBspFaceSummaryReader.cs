using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupBspFaceSummaryReader
{
    private const int RefStride = 2;

    public static WmoGroupBspFaceSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupBspFaceSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupBspFaceSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? mobrPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mobr);
        if (mobrPayload is null)
            throw new InvalidDataException("WMO group BSP-face summary requires a MOBR subchunk.");

        if (mobrPayload.Length % RefStride != 0)
            throw new InvalidDataException($"MOBR payload size {mobrPayload.Length} is not divisible by {RefStride}.");

        int refCount = mobrPayload.Length / RefStride;
        HashSet<ushort> refs = [];
        int minFaceRef = 0;
        int maxFaceRef = 0;
        if (refCount > 0)
        {
            minFaceRef = ushort.MaxValue;
            maxFaceRef = ushort.MinValue;
        }

        for (int index = 0; index < refCount; index++)
        {
            ushort value = BinaryPrimitives.ReadUInt16LittleEndian(mobrPayload.AsSpan(index * RefStride, RefStride));
            refs.Add(value);
            minFaceRef = Math.Min(minFaceRef, value);
            maxFaceRef = Math.Max(maxFaceRef, value);
        }

        return new WmoGroupBspFaceSummary(
            sourcePath,
            version,
            mobrPayload.Length,
            refCount,
            refs.Count,
            minFaceRef,
            maxFaceRef,
            refCount - refs.Count);
    }
}