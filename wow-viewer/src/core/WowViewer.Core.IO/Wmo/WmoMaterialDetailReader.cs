using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoMaterialDetailReader
{
    private const int MohdSize = 64;
    private const int StandardMomtEntrySize = 64;
    private const int LegacyMomtEntrySize = 48;
    private const int VintageMomtEntrySize = 44;

    public static IReadOnlyList<WmoMaterialDetail> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoMaterialDetail> Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedMaterialCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(0, 4)));
        byte[] momt = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Momt);
        byte[] motx = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Motx) ?? [];
        int entrySize = InferMomtEntrySize(momt, reportedMaterialCount);
        if (entrySize <= 0 || momt.Length % entrySize != 0)
            throw new InvalidDataException($"MOMT payload size {momt.Length} is not compatible with inferred entry size {entrySize}.");

        int entryCount = momt.Length / entrySize;
        List<WmoMaterialDetail> details = new(entryCount);
        for (int materialIndex = 0; materialIndex < entryCount; materialIndex++)
        {
            int offset = materialIndex * entrySize;
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 0, 4));
            uint shader = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 4, 4));
            uint blendMode = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 8, 4));
            uint texture1Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 12, 4));
            uint texture2Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 24, 4));
            uint texture3Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 36, 4));

            details.Add(new WmoMaterialDetail(
                materialIndex,
                offset,
                entrySize,
                flags,
                shader,
                blendMode,
                texture1Offset,
                WmoRootReaderCommon.ResolveStringAtOffset(motx, texture1Offset),
                texture2Offset,
                WmoRootReaderCommon.ResolveStringAtOffset(motx, texture2Offset),
                texture3Offset,
                WmoRootReaderCommon.ResolveStringAtOffset(motx, texture3Offset)));
        }

        return details;
    }

    private static int InferMomtEntrySize(byte[] payload, int reportedMaterialCount)
    {
        if (payload.Length == 0)
            return 0;

        if (reportedMaterialCount > 0)
        {
            if (payload.Length == reportedMaterialCount * StandardMomtEntrySize)
                return StandardMomtEntrySize;

            if (payload.Length == reportedMaterialCount * LegacyMomtEntrySize)
                return LegacyMomtEntrySize;

            if (payload.Length == reportedMaterialCount * VintageMomtEntrySize)
                return VintageMomtEntrySize;
        }

        if (payload.Length % StandardMomtEntrySize == 0)
            return StandardMomtEntrySize;

        if (payload.Length % LegacyMomtEntrySize == 0)
            return LegacyMomtEntrySize;

        if (payload.Length % VintageMomtEntrySize == 0)
            return VintageMomtEntrySize;

        return 0;
    }
}