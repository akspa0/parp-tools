using System.Buffers.Binary;
using System.Numerics;

namespace WowViewer.Core.IO.Maps;

public static class RosettaAlphaWdtPlacementPatcher
{
    private const int TilesPerAxis = 64;
    private const int MainEntrySize = 16;
    private const int MainPayloadOffset = 12 + 8 + 128 + 8;
    private const int MddfMhdrOffset = 0x0C;
    private const int ModfMhdrOffset = 0x14;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;

    /// <summary>
    /// Rosetta-only compatibility shim for 0.5.3 client inspection. The base Alpha writer consumes
    /// renderer-space placements; this patches the generated Rosetta WDT bytes to the tile-local file
    /// axes the alpha client appears to use without changing the protected writer itself.
    /// </summary>
    public static void PatchPlacementFileAxes(byte[] wdtBytes, IEnumerable<RosettaTilePlan> tiles)
    {
        ArgumentNullException.ThrowIfNull(wdtBytes);
        ArgumentNullException.ThrowIfNull(tiles);

        foreach (RosettaTilePlan tile in tiles)
        {
            Dictionary<int, RosettaPlacementRecord> placementById =
                tile.Placements.ToDictionary(static p => p.UniqueId);

            PatchPlacementChunk(
                wdtBytes,
                tile.TileX,
                tile.TileY,
                MddfMhdrOffset,
                "MDDF",
                MddfEntrySize,
                placementById,
                PatchMddfEntry);

            PatchPlacementChunk(
                wdtBytes,
                tile.TileX,
                tile.TileY,
                ModfMhdrOffset,
                "MODF",
                ModfEntrySize,
                placementById,
                PatchModfEntry);
        }
    }

    private static void PatchPlacementChunk(
        byte[] wdtBytes,
        int tileX,
        int tileY,
        int mhdrFieldOffset,
        string expectedTag,
        int entrySize,
        IReadOnlyDictionary<int, RosettaPlacementRecord> placementById,
        Action<byte[], int, RosettaPlacementRecord> patchEntry)
    {
        int payloadOffset = FindAlphaTileSubchunkPayload(
            wdtBytes, tileX, tileY, mhdrFieldOffset, expectedTag, out int payloadSize);

        if (payloadSize == 0)
            return;

        if (payloadSize % entrySize != 0)
            throw new InvalidDataException($"{expectedTag} payload size {payloadSize} is not a multiple of {entrySize}.");

        for (int offset = payloadOffset; offset < payloadOffset + payloadSize; offset += entrySize)
        {
            int uniqueId = BinaryPrimitives.ReadInt32LittleEndian(wdtBytes.AsSpan(offset + 4));
            if (!placementById.TryGetValue(uniqueId, out RosettaPlacementRecord? placement))
                continue;

            patchEntry(wdtBytes, offset, placement);
        }
    }

    private static void PatchMddfEntry(byte[] wdtBytes, int offset, RosettaPlacementRecord placement)
    {
        Vector3 position = RosettaTilesetGenerator.GetAlphaClientFilePosition(placement);
        WriteVector3(wdtBytes, offset + 0x08, position);
    }

    private static void PatchModfEntry(byte[] wdtBytes, int offset, RosettaPlacementRecord placement)
    {
        Vector3 position = RosettaTilesetGenerator.GetAlphaClientFilePosition(placement);
        WriteVector3(wdtBytes, offset + 0x08, position);

        float extentU = MathF.Abs(placement.Asset.BoundsMax.X - placement.Asset.BoundsMin.X);
        float extentV = MathF.Abs(placement.Asset.BoundsMax.Y - placement.Asset.BoundsMin.Y);
        float half = MathF.Max(extentU, extentV) / 2f;
        float minY = position.Y + MathF.Min(placement.Asset.BoundsMin.Z, 0f);
        float maxY = position.Y + MathF.Max(placement.Asset.BoundsMax.Z, 0f);

        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x20), position.X + half);
        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x24), maxY);
        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x28), position.Z + half);
        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x2C), position.X - half);
        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x30), minY);
        BinaryPrimitives.WriteSingleLittleEndian(wdtBytes.AsSpan(offset + 0x34), position.Z - half);
    }

    private static int FindAlphaTileSubchunkPayload(
        byte[] wdtBytes,
        int tileX,
        int tileY,
        int mhdrFieldOffset,
        string expectedTag,
        out int payloadSize)
    {
        if ((uint)tileX >= TilesPerAxis || (uint)tileY >= TilesPerAxis)
            throw new ArgumentOutOfRangeException(nameof(tileX), $"Tile ({tileX},{tileY}) is outside the 64x64 Alpha grid.");

        int mainEntryOffset = MainPayloadOffset + (((tileY * TilesPerAxis) + tileX) * MainEntrySize);
        EnsureRange(wdtBytes, mainEntryOffset, MainEntrySize, "MAIN tile entry");

        int tileOffset = BinaryPrimitives.ReadInt32LittleEndian(wdtBytes.AsSpan(mainEntryOffset));
        if (tileOffset <= 0)
            throw new InvalidDataException($"Alpha WDT MAIN entry for tile ({tileX},{tileY}) does not point at an embedded tile.");

        int mhdrPayloadOffset = tileOffset + 8;
        EnsureRange(wdtBytes, mhdrPayloadOffset, 64, "MHDR payload");

        int relativeOffset = BinaryPrimitives.ReadInt32LittleEndian(wdtBytes.AsSpan(mhdrPayloadOffset + mhdrFieldOffset));
        int chunkOffset = mhdrPayloadOffset + relativeOffset;
        EnsureRange(wdtBytes, chunkOffset, 8, expectedTag);

        string actualTag = ReadChunkId(wdtBytes, chunkOffset);
        if (!string.Equals(actualTag, expectedTag, StringComparison.Ordinal))
            throw new InvalidDataException($"Expected {expectedTag} at 0x{chunkOffset:X}, found {actualTag}.");

        payloadSize = BinaryPrimitives.ReadInt32LittleEndian(wdtBytes.AsSpan(chunkOffset + 4));
        EnsureRange(wdtBytes, chunkOffset + 8, payloadSize, $"{expectedTag} payload");
        return chunkOffset + 8;
    }

    private static void WriteVector3(byte[] bytes, int offset, Vector3 value)
    {
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(offset), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(offset + 4), value.Y);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(offset + 8), value.Z);
    }

    private static void EnsureRange(byte[] data, int offset, int count, string label)
    {
        if (offset < 0 || count < 0 || offset > data.Length - count)
            throw new InvalidDataException($"{label} at 0x{offset:X} exceeds Alpha WDT length {data.Length}.");
    }

    private static string ReadChunkId(byte[] data, int offset)
        => new(new[]
        {
            (char)data[offset + 3],
            (char)data[offset + 2],
            (char)data[offset + 1],
            (char)data[offset]
        });
}
