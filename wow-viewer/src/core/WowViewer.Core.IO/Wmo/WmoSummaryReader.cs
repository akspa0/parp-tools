using System.Buffers.Binary;
using System.Numerics;
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

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
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

        byte[]? motx = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Motx);
        byte[]? modn = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Modn);
        byte[]? moms = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Mods);
        byte[]? modd = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Modd);
        byte[]? momt = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Momt);
        byte[]? mogi = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Mogi);
        byte[]? mosb = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, WmoChunkIds.Mosb);

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
                hasSkybox: mosb is { Length: > 0 },
            flags,
            boundsMin,
            boundsMax);
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
