using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoEmbeddedGroupMeshDetailReader
{
    public static IReadOnlyList<WmoEmbeddedGroupMeshDetail> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoEmbeddedGroupMeshDetail> Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        List<ChunkSpan> groupChunks = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        if (groupChunks.Count == 0)
            return [];

        List<WmoEmbeddedGroupMeshDetail> details = new(groupChunks.Count);
        for (int groupIndex = 0; groupIndex < groupChunks.Count; groupIndex++)
        {
            ChunkSpan groupChunk = groupChunks[groupIndex];
            string detailSourcePath = $"{sourcePath}#MOGP[{groupIndex}]@{groupChunk.HeaderOffset}";
            byte[] mogp = WmoGroupReaderCommon.ReadChunkPayload(stream, groupChunk);

            WmoGroupSummary groupSummary = WmoGroupSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version);
            WmoGroupMeshDetail mesh = WmoGroupMeshDetailReader.ReadMogpPayload(mogp, detailSourcePath, version);
            WmoGroupLiquidSummary? liquidSummary = groupSummary.HasLiquid
                ? WmoGroupLiquidSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;

            details.Add(new WmoEmbeddedGroupMeshDetail(
                groupIndex,
                groupChunk.HeaderOffset,
                groupSummary,
                mesh,
                liquidSummary,
                ReadRefs(mogp, mesh.HeaderSizeBytes, WmoChunkIds.Modr),
                ReadRefs(mogp, mesh.HeaderSizeBytes, WmoChunkIds.Molr)));
        }

        return details;
    }

    private static List<ushort> ReadRefs(byte[] mogp, int headerSizeBytes, FourCC chunkId)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, chunkId);
        if (payload is null)
            return [];

        if (payload.Length % 2 != 0)
            throw new InvalidDataException($"{chunkId} payload size {payload.Length} is not divisible by 2.");

        List<ushort> refs = new(payload.Length / 2);
        for (int offset = 0; offset < payload.Length; offset += 2)
            refs.Add(BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2)));

        return refs;
    }
}