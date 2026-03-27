using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoMaterialSummaryReader
{
    private const int MohdSize = 64;
    private const int StandardMomtEntrySize = 64;
    private const int LegacyMomtEntrySize = 48;
    private const int VintageMomtEntrySize = 44;

    public static WmoMaterialSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoMaterialSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO material summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? mohdChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mohd);
        ChunkSpan? momtChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Momt);
        if (mohdChunk is null)
            throw new InvalidDataException("WMO material summary requires an MOHD chunk.");

        if (momtChunk is null)
            throw new InvalidDataException("WMO material summary requires a MOMT chunk.");

        byte[] mohd = ReadChunkPayload(stream, mohdChunk.Value);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedMaterialCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(0, 4)));
        byte[] momt = ReadChunkPayload(stream, momtChunk.Value);
        int entrySize = InferMomtEntrySize(momt, reportedMaterialCount);
        if (entrySize <= 0 || momt.Length % entrySize != 0)
            throw new InvalidDataException($"MOMT payload size {momt.Length} is not compatible with inferred entry size {entrySize}.");

        int entryCount = momt.Length / entrySize;
        HashSet<uint> shaders = [];
        HashSet<uint> blendModes = [];
        int nonZeroFlagCount = 0;
        uint maxTexture1Offset = 0;
        uint maxTexture2Offset = 0;
        uint maxTexture3Offset = 0;

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * entrySize;
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 0, 4));
            uint shader = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 4, 4));
            uint blendMode = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 8, 4));
            uint texture1Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 12, 4));
            uint texture2Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 24, 4));
            uint texture3Offset = BinaryPrimitives.ReadUInt32LittleEndian(momt.AsSpan(offset + 36, 4));

            shaders.Add(shader);
            blendModes.Add(blendMode);
            if (flags != 0)
                nonZeroFlagCount++;

            maxTexture1Offset = Math.Max(maxTexture1Offset, texture1Offset);
            maxTexture2Offset = Math.Max(maxTexture2Offset, texture2Offset);
            maxTexture3Offset = Math.Max(maxTexture3Offset, texture3Offset);
        }

        return new WmoMaterialSummary(
            sourcePath,
            version,
            momt.Length,
            entrySize,
            entryCount,
            distinctShaderCount: shaders.Count,
            distinctBlendModeCount: blendModes.Count,
            nonZeroFlagCount,
            maxTexture1Offset: checked((int)maxTexture1Offset),
            maxTexture2Offset: checked((int)maxTexture2Offset),
            maxTexture3Offset: checked((int)maxTexture3Offset));
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != WmoChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
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
