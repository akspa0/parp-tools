using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupSummaryReader
{
    private const int VertexStride = 12;
    private const int IndexStride = 2;
    private const int NormalStride = 12;
    private const int UvStride = 8;
    private const int BatchStride = 24;
    private const int VertexColorStride = 4;
    private const int DoodadRefStride = 2;

    public static WmoGroupSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        int faceMaterialCount = 0;
        int vertexCount = 0;
        int indexCount = 0;
        int normalCount = 0;
        int primaryUvCount = 0;
        int additionalUvSetCount = 0;
        int batchCount = 0;
        int vertexColorCount = 0;
        int doodadRefCount = 0;
        bool hasLiquid = false;

        foreach ((var header, _) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id == WmoChunkIds.Mopy)
                faceMaterialCount += WmoGroupReaderCommon.CountMopyEntries(header.Size, version);
            else if (header.Id == WmoChunkIds.Movi || header.Id == WmoChunkIds.Moin)
                indexCount += checked((int)header.Size / IndexStride);
            else if (header.Id == WmoChunkIds.Movt)
                vertexCount += checked((int)header.Size / VertexStride);
            else if (header.Id == WmoChunkIds.Monr)
                normalCount += checked((int)header.Size / NormalStride);
            else if (header.Id == WmoChunkIds.Motv)
            {
                int uvCount = checked((int)header.Size / UvStride);
                if (primaryUvCount == 0)
                    primaryUvCount = uvCount;
                else
                    additionalUvSetCount++;
            }
            else if (header.Id == WmoChunkIds.Moba)
                batchCount += checked((int)header.Size / BatchStride);
            else if (header.Id == WmoChunkIds.Mocv)
                vertexColorCount += checked((int)header.Size / VertexColorStride);
            else if (header.Id == WmoChunkIds.Modr)
                doodadRefCount += checked((int)header.Size / DoodadRefStride);
            else if (header.Id == WmoChunkIds.Mliq)
                hasLiquid = true;
        }

        return new WmoGroupSummary(
            sourcePath,
            version,
            headerSizeBytes,
            nameOffset: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x00, 4)),
            descriptiveNameOffset: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x04, 4)),
            flags: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x08, 4)),
            boundsMin: WmoGroupReaderCommon.ReadVector3(mogp.AsSpan(0x0C, 12)),
            boundsMax: WmoGroupReaderCommon.ReadVector3(mogp.AsSpan(0x18, 12)),
            portalStart: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x24, 2)),
            portalCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x26, 2)),
            transparentBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x28, 2)),
            interiorBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x2A, 2)),
            exteriorBatchCount: BinaryPrimitives.ReadUInt16LittleEndian(mogp.AsSpan(0x2C, 2)),
            groupLiquid: BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(0x34, 4)),
            faceMaterialCount,
            vertexCount,
            indexCount,
            normalCount,
            primaryUvCount,
            additionalUvSetCount,
            batchCount,
            vertexColorCount,
            doodadRefCount,
            hasLiquid);
    }
}
