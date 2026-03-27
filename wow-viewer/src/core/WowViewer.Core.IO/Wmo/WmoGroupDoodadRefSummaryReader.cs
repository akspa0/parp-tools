using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupDoodadRefSummaryReader
{
    private const int RefStride = 2;

    public static WmoGroupDoodadRefSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupDoodadRefSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? modrPayload = null;
        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Modr)
                continue;

            modrPayload = mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            break;
        }

        if (modrPayload is null)
            throw new InvalidDataException("WMO group doodad-ref summary requires a MODR subchunk.");

        if (modrPayload.Length % RefStride != 0)
            throw new InvalidDataException($"MODR payload size {modrPayload.Length} is not divisible by {RefStride}.");

        int refCount = modrPayload.Length / RefStride;
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
            ushort value = BinaryPrimitives.ReadUInt16LittleEndian(modrPayload.AsSpan(index * RefStride, RefStride));
            refs.Add(value);
            minRef = Math.Min(minRef, value);
            maxRef = Math.Max(maxRef, value);
        }

        return new WmoGroupDoodadRefSummary(
            sourcePath,
            version,
            modrPayload.Length,
            refCount,
            distinctRefCount: refs.Count,
            minRef,
            maxRef,
            duplicateRefCount: refCount - refs.Count);
    }
}
