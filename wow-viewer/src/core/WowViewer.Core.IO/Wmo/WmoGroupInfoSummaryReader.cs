using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupInfoSummaryReader
{
    private const int MohdSize = 64;
    private const int StandardMogiEntrySize = 32;
    private const int LegacyMogiEntrySize = 40;

    public static WmoGroupInfoSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupInfoSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        uint? version = TryReadVersion(stream, chunks);
        WowFileDetection detection = WowFileDetector.Detect(sourcePath, chunks, version);
        if (detection.Kind != WowFileKind.Wmo)
            throw new InvalidDataException($"WMO group-info summary requires a WMO root file, but found {detection.Kind}.");

        ChunkSpan? mohdChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mohd);
        ChunkSpan? mogiChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mogi);
        if (mohdChunk is null)
            throw new InvalidDataException("WMO group-info summary requires an MOHD chunk.");

        if (mogiChunk is null)
            throw new InvalidDataException("WMO group-info summary requires an MOGI chunk.");

        byte[] mohd = ReadChunkPayload(stream, mohdChunk.Value);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(4, 4)));
        byte[] mogi = ReadChunkPayload(stream, mogiChunk.Value);
        int entrySize = InferMogiEntrySize(mogi, reportedGroupCount);
        if (entrySize <= 0 || mogi.Length % entrySize != 0)
            throw new InvalidDataException($"MOGI payload size {mogi.Length} is not compatible with inferred entry size {entrySize}.");

        int entryCount = mogi.Length / entrySize;
        HashSet<uint> flagsSeen = [];
        int nonZeroFlagCount = 0;
        int minNameOffset = 0;
        int maxNameOffset = 0;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;

        if (entryCount > 0)
        {
            minNameOffset = int.MaxValue;
            maxNameOffset = int.MinValue;
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * entrySize;
            if (entrySize == LegacyMogiEntrySize)
                offset += 8;

            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(mogi.AsSpan(offset, 4));
            Vector3 groupBoundsMin = ReadVector3(mogi.AsSpan(offset + 4, 12));
            Vector3 groupBoundsMax = ReadVector3(mogi.AsSpan(offset + 16, 12));
            int nameOffset = BinaryPrimitives.ReadInt32LittleEndian(mogi.AsSpan(offset + 28, 4));

            flagsSeen.Add(flags);
            if (flags != 0)
                nonZeroFlagCount++;

            minNameOffset = Math.Min(minNameOffset, nameOffset);
            maxNameOffset = Math.Max(maxNameOffset, nameOffset);
            boundsMin = Vector3.Min(boundsMin, groupBoundsMin);
            boundsMax = Vector3.Max(boundsMax, groupBoundsMax);
        }

        if (entryCount == 0)
            minNameOffset = maxNameOffset = 0;

        return new WmoGroupInfoSummary(
            sourcePath,
            version,
            mogi.Length,
            entrySize,
            entryCount,
            distinctFlagCount: flagsSeen.Count,
            nonZeroFlagCount,
            minNameOffset,
            maxNameOffset,
            boundsMin,
            boundsMax);
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

    private static int InferMogiEntrySize(byte[] payload, int reportedGroupCount)
    {
        if (payload.Length == 0)
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

    private static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }
}
