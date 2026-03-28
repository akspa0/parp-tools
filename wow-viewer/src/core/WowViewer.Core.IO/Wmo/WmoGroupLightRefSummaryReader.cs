using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupLightRefSummaryReader
{
    private const int RefStride = 2;

    public static WmoGroupLightRefSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupLightRefSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupLightRefSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? molrPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Molr);
        if (molrPayload is null)
            throw new InvalidDataException("WMO group light-ref summary requires a MOLR subchunk.");

        if (molrPayload.Length % RefStride != 0)
            throw new InvalidDataException($"MOLR payload size {molrPayload.Length} is not divisible by {RefStride}.");

        int refCount = molrPayload.Length / RefStride;
        HashSet<ushort> refs = [];
        int minRef = 0;
        int maxRef = 0;
        if (refCount > 0)
        {
            minRef = ushort.MaxValue;
            maxRef = ushort.MinValue;
        }

        for (int index = 0; index < refCount; index++)
        {
            ushort value = BinaryPrimitives.ReadUInt16LittleEndian(molrPayload.AsSpan(index * RefStride, RefStride));
            refs.Add(value);
            minRef = Math.Min(minRef, value);
            maxRef = Math.Max(maxRef, value);
        }

        return new WmoGroupLightRefSummary(
            sourcePath,
            version,
            molrPayload.Length,
            refCount,
            refs.Count,
            minRef,
            maxRef,
            refCount - refs.Count);
    }
}