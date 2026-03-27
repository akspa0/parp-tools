using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoSummaryReader
{
    private const int MohdSize = 64;
    private const int ModsEntrySize = 32;
    private const int ModdEntrySize = 40;
    private const int StandardMogiEntrySize = 32;
    private const int LegacyMogiEntrySize = 40;
    private const int StandardMomtEntrySize = 64;
    private const int LegacyMomtEntrySize = 48;
    private const int VintageMomtEntrySize = 44;

    public static WmoSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO semantic summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? mohdChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mohd);
        if (mohdChunk is null)
            throw new InvalidDataException("WMO semantic summary requires an MOHD chunk.");

        byte[] mohd = ReadChunkPayload(stream, mohdChunk.Value);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedMaterialCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(0, 4)));
        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(4, 4)));
        int reportedPortalCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(8, 4)));
        int reportedLightCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(12, 4)));
        int reportedDoodadNameCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(16, 4)));
        int reportedDoodadPlacementCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(20, 4)));
        int reportedDoodadSetCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(24, 4)));
        Vector3 boundsMin = ReadVector3(mohd.AsSpan(36, 12));
        Vector3 boundsMax = ReadVector3(mohd.AsSpan(48, 12));
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(60, 4));

        byte[]? motx = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Motx);
        byte[]? modn = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Modn);
        byte[]? moms = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Mods);
        byte[]? modd = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Modd);
        byte[]? momt = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Momt);
        byte[]? mogi = TryReadFirstChunkPayload(stream, chunks, WmoChunkIds.Mogi);

        return new WmoSummary(
            sourcePath,
            version,
            reportedMaterialCount,
            materialEntryCount: CountEntries(momt, InferMomtEntrySize(momt, reportedMaterialCount)),
            reportedGroupCount,
            groupInfoCount: CountEntries(mogi, InferMogiEntrySize(mogi, reportedGroupCount)),
            reportedPortalCount,
            reportedLightCount,
            textureNameCount: MapSummaryReaderCommon.CountStringEntries(motx),
            reportedDoodadNameCount,
            doodadNameTableCount: MapSummaryReaderCommon.CountStringEntries(modn),
            reportedDoodadPlacementCount,
            doodadPlacementEntryCount: CountEntries(modd, ModdEntrySize),
            reportedDoodadSetCount,
            doodadSetEntryCount: CountEntries(moms, ModsEntrySize),
            flags,
            boundsMin,
            boundsMax);
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != FourCC.FromString("MVER"))
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    private static byte[]? TryReadFirstChunkPayload(Stream stream, IReadOnlyList<ChunkSpan> chunks, FourCC id)
    {
        ChunkSpan? chunk = chunks.FirstOrDefault(location => location.Header.Id == id);
        return chunk is null ? null : ReadChunkPayload(stream, chunk.Value);
    }

    private static byte[] ReadChunkPayload(Stream stream, ChunkSpan chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Header.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static int InferMogiEntrySize(byte[]? payload, int reportedGroupCount)
    {
        if (payload is not { Length: > 0 })
            return 0;

        if (reportedGroupCount > 0)
        {
            if (payload.Length == reportedGroupCount * StandardMogiEntrySize)
                return StandardMogiEntrySize;

            if (payload.Length == reportedGroupCount * LegacyMogiEntrySize)
                return LegacyMogiEntrySize;
        }

        if (payload.Length % StandardMogiEntrySize == 0)
            return StandardMogiEntrySize;

        if (payload.Length % LegacyMogiEntrySize == 0)
            return LegacyMogiEntrySize;

        return 0;
    }

    private static int InferMomtEntrySize(byte[]? payload, int reportedMaterialCount)
    {
        if (payload is not { Length: > 0 })
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

    private static int CountEntries(byte[]? payload, int stride)
    {
        if (payload is not { Length: > 0 } || stride <= 0)
            return 0;

        return payload.Length / stride;
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }
}
