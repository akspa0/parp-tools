using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoEmbeddedGroupLinkageSummaryReader
{
    private const int MohdSize = 64;
    private const int LegacyMogiEntrySize = 40;
    private const float BoundsMatchTolerance = 0.001f;

    public static WmoEmbeddedGroupLinkageSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoEmbeddedGroupLinkageSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(4, 4)));
        byte[] mogi = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogi);
        int entrySize = WmoRootReaderCommon.InferMogiEntrySize(mogi, reportedGroupCount);
        if (entrySize <= 0 || mogi.Length % entrySize != 0)
            throw new InvalidDataException($"MOGI payload size {mogi.Length} is not compatible with inferred entry size {entrySize}.");

        List<MogiEntry> groupInfos = ReadMogiEntries(mogi, entrySize);
        List<ChunkSpan> embeddedGroupChunks = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        if (embeddedGroupChunks.Count == 0)
            throw new InvalidDataException("WMO embedded-group linkage summary requires one or more MOGP chunks in the root file.");

        int coveredPairCount = Math.Min(groupInfos.Count, embeddedGroupChunks.Count);
        int missingEmbeddedGroupCount = Math.Max(0, groupInfos.Count - embeddedGroupChunks.Count);
        int extraEmbeddedGroupCount = Math.Max(0, embeddedGroupChunks.Count - groupInfos.Count);
        int flagMatchCount = 0;
        int boundsMatchCount = 0;
        float maxBoundsDelta = 0f;

        for (int index = 0; index < coveredPairCount; index++)
        {
            WmoGroupSummary groupSummary = WmoGroupSummaryReader.ReadMogpPayload(
                WmoGroupReaderCommon.ReadChunkPayload(stream, embeddedGroupChunks[index]),
                $"{sourcePath}#MOGP@{embeddedGroupChunks[index].HeaderOffset}",
                version);

            MogiEntry groupInfo = groupInfos[index];
            if (groupInfo.Flags == groupSummary.Flags)
                flagMatchCount++;

            float pairDelta = MaxBoundsDelta(groupInfo.BoundsMin, groupInfo.BoundsMax, groupSummary.BoundsMin, groupSummary.BoundsMax);
            maxBoundsDelta = Math.Max(maxBoundsDelta, pairDelta);
            if (pairDelta <= BoundsMatchTolerance)
                boundsMatchCount++;
        }

        return new WmoEmbeddedGroupLinkageSummary(
            sourcePath,
            version,
            groupInfos.Count,
            embeddedGroupChunks.Count,
            coveredPairCount,
            missingEmbeddedGroupCount,
            extraEmbeddedGroupCount,
            flagMatchCount,
            boundsMatchCount,
            maxBoundsDelta);
    }

    private static List<MogiEntry> ReadMogiEntries(byte[] mogi, int entrySize)
    {
        List<MogiEntry> entries = new(mogi.Length / entrySize);
        for (int index = 0; index < mogi.Length; index += entrySize)
        {
            int offset = index;
            if (entrySize == LegacyMogiEntrySize)
                offset += 8;

            entries.Add(new MogiEntry(
                BinaryPrimitives.ReadUInt32LittleEndian(mogi.AsSpan(offset, 4)),
                ReadVector3(mogi.AsSpan(offset + 4, 12)),
                ReadVector3(mogi.AsSpan(offset + 16, 12))));
        }

        return entries;
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }

    private static float MaxBoundsDelta(Vector3 infoMin, Vector3 infoMax, Vector3 groupMin, Vector3 groupMax)
    {
        return Math.Max(
            Math.Max(MaxVectorDelta(infoMin, groupMin), MaxVectorDelta(infoMax, groupMax)),
            0f);
    }

    private static float MaxVectorDelta(Vector3 left, Vector3 right)
    {
        return Math.Max(Math.Max(MathF.Abs(left.X - right.X), MathF.Abs(left.Y - right.Y)), MathF.Abs(left.Z - right.Z));
    }

    private sealed record MogiEntry(uint Flags, Vector3 BoundsMin, Vector3 BoundsMax);
}
