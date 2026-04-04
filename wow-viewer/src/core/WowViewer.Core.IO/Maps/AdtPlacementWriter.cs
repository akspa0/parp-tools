using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtPlacementWriter
{
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;

    public static byte[] ApplyTransaction(string path, AdtPlacementEditTransaction transaction)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(transaction);

        return ApplyTransaction(File.ReadAllBytes(path), Path.GetFullPath(path), transaction);
    }

    public static byte[] ApplyTransaction(byte[] sourceBytes, string sourcePath, AdtPlacementEditTransaction transaction)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(transaction);

        byte[] updatedBytes = sourceBytes.ToArray();
        if (transaction.Moves.Count == 0)
            return updatedBytes;

        using MemoryStream stream = new(updatedBytes, writable: true);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtObj))
            throw new InvalidDataException($"ADT placement writing requires an ADT or ADTOBJ file, but found {fileSummary.Kind}.");

        ValidateDistinctMoves(transaction.Moves);

        MapChunkLocation? mddfChunk = FindChunk(fileSummary, MapChunkIds.Mddf);
        MapChunkLocation? modfChunk = FindChunk(fileSummary, MapChunkIds.Modf);

        foreach (AdtPlacementMove move in transaction.Moves)
        {
            switch (move.Placement.Kind)
            {
                case AdtPlacementKind.Model:
                    if (!mddfChunk.HasValue)
                        throw new InvalidDataException("Cannot apply a model placement move because the file has no MDDF chunk.");

                    ApplyModelMove(updatedBytes, mddfChunk.Value, move);
                    break;

                case AdtPlacementKind.WorldModel:
                    if (!modfChunk.HasValue)
                        throw new InvalidDataException("Cannot apply a world-model placement move because the file has no MODF chunk.");

                    ApplyWorldModelMove(updatedBytes, modfChunk.Value, move);
                    break;

                default:
                    throw new InvalidOperationException($"Unsupported placement kind {move.Placement.Kind}.");
            }
        }

        return updatedBytes;
    }

    public static void Write(string inputPath, string outputPath, AdtPlacementEditTransaction transaction)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(transaction);

        byte[] updatedBytes = ApplyTransaction(inputPath, transaction);
        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllBytes(outputPath, updatedBytes);
    }

    private static void ValidateDistinctMoves(IReadOnlyList<AdtPlacementMove> moves)
    {
        HashSet<(AdtPlacementKind kind, int entryIndex, int uniqueId)> seen = [];
        foreach (AdtPlacementMove move in moves)
        {
            var key = (move.Placement.Kind, move.Placement.EntryIndex, move.Placement.UniqueId);
            if (!seen.Add(key))
                throw new InvalidOperationException($"Duplicate placement move detected for {move.Placement.Kind} entry {move.Placement.EntryIndex} with UniqueId {move.Placement.UniqueId}.");
        }
    }

    private static MapChunkLocation? FindChunk(MapFileSummary summary, WowViewer.Core.Chunks.FourCC id)
    {
        foreach (MapChunkLocation chunk in summary.Chunks)
        {
            if (chunk.Id == id)
                return chunk;
        }

        return null;
    }

    private static void ApplyModelMove(byte[] bytes, MapChunkLocation chunk, AdtPlacementMove move)
    {
        int entryOffset = ResolveEntryOffset(chunk, MddfEntrySize, move.Placement.EntryIndex, "MDDF");
        ValidateUniqueId(bytes, entryOffset + 4, move.Placement.UniqueId, "MDDF", move.Placement.EntryIndex);

        WriteModelPosition(bytes, entryOffset, move.NewPosition);
    }

    private static void ApplyWorldModelMove(byte[] bytes, MapChunkLocation chunk, AdtPlacementMove move)
    {
        int entryOffset = ResolveEntryOffset(chunk, ModfEntrySize, move.Placement.EntryIndex, "MODF");
        ValidateUniqueId(bytes, entryOffset + 4, move.Placement.UniqueId, "MODF", move.Placement.EntryIndex);

        Vector3 oldPosition = ReadWorldModelPosition(bytes.AsSpan(entryOffset, ModfEntrySize));
        Vector3 delta = move.NewPosition - oldPosition;

        WriteWorldModelPosition(bytes, entryOffset, move.NewPosition);
        TranslateWorldModelBounds(bytes, entryOffset, delta);
    }

    private static int ResolveEntryOffset(MapChunkLocation chunk, int entrySize, int entryIndex, string chunkName)
    {
        int entryCount = checked((int)(chunk.Size / entrySize));
        if (entryIndex >= entryCount)
            throw new ArgumentOutOfRangeException(nameof(entryIndex), $"{chunkName} entry index {entryIndex} is outside the available range 0..{Math.Max(0, entryCount - 1)}.");

        return checked((int)chunk.DataOffset + entryIndex * entrySize);
    }

    private static void ValidateUniqueId(byte[] bytes, int uniqueIdOffset, int expectedUniqueId, string chunkName, int entryIndex)
    {
        int actualUniqueId = unchecked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(uniqueIdOffset, 4)));
        if (actualUniqueId != expectedUniqueId)
        {
            throw new InvalidDataException(
                $"{chunkName} entry {entryIndex} no longer matches the requested UniqueId. Expected {expectedUniqueId}, found {actualUniqueId}.");
        }
    }

    private static void WriteModelPosition(byte[] bytes, int entryOffset, Vector3 worldPosition)
    {
        float rawX = MapOrigin - worldPosition.Y;
        float rawY = MapOrigin - worldPosition.X;
        float rawZ = worldPosition.Z;

        WriteSingle(bytes, entryOffset + 8, rawX);
        WriteSingle(bytes, entryOffset + 12, rawZ);
        WriteSingle(bytes, entryOffset + 16, rawY);
    }

    private static Vector3 ReadWorldModelPosition(ReadOnlySpan<byte> entry)
    {
        float rawX = ReadSingle(entry[8..12]);
        float rawZ = ReadSingle(entry[12..16]);
        float rawY = ReadSingle(entry[16..20]);
        return new Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ);
    }

    private static void WriteWorldModelPosition(byte[] bytes, int entryOffset, Vector3 worldPosition)
    {
        float rawX = MapOrigin - worldPosition.Y;
        float rawY = MapOrigin - worldPosition.X;
        float rawZ = worldPosition.Z;

        WriteSingle(bytes, entryOffset + 8, rawX);
        WriteSingle(bytes, entryOffset + 12, rawZ);
        WriteSingle(bytes, entryOffset + 16, rawY);
    }

    private static void TranslateWorldModelBounds(byte[] bytes, int entryOffset, Vector3 delta)
    {
        float rawDeltaX = -delta.Y;
        float rawDeltaY = -delta.X;
        float rawDeltaZ = delta.Z;

        AddSingle(bytes, entryOffset + 32, rawDeltaX);
        AddSingle(bytes, entryOffset + 36, rawDeltaZ);
        AddSingle(bytes, entryOffset + 40, rawDeltaY);
        AddSingle(bytes, entryOffset + 44, rawDeltaX);
        AddSingle(bytes, entryOffset + 48, rawDeltaZ);
        AddSingle(bytes, entryOffset + 52, rawDeltaY);
    }

    private static float ReadSingle(ReadOnlySpan<byte> bytes)
    {
        return BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes));
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }

    private static void AddSingle(byte[] bytes, int offset, float delta)
    {
        float value = ReadSingle(bytes.AsSpan(offset, 4));
        WriteSingle(bytes, offset, value + delta);
    }
}