using System.Buffers.Binary;
using System.Numerics;
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

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohd.Length < MohdSize)
            throw new InvalidDataException($"MOHD payload is too short ({mohd.Length} bytes). Expected at least {MohdSize} bytes.");

        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohd.AsSpan(4, 4)));
        byte[] mogi = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogi);
        int entrySize = WmoRootReaderCommon.InferMogiEntrySize(mogi, reportedGroupCount);
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

    private static Vector3 ReadVector3(ReadOnlySpan<byte> bytes)
    {
        return new Vector3(
            BitConverter.ToSingle(bytes[0..4]),
            BitConverter.ToSingle(bytes[4..8]),
            BitConverter.ToSingle(bytes[8..12]));
    }
}
