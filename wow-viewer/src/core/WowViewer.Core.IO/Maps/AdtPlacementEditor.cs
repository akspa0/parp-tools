using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public enum PlacementEditKind
{
    Move,
    Rotate,
    Scale,
    Add,
    Delete,
}

/// <summary>A single authoring edit against a loaded ADT placement set.</summary>
public abstract record AdtPlacementEdit
{
    public abstract PlacementEditKind Kind_ { get; }
}

public sealed record AdtPlacementMoveEdit(AdtPlacementKind Kind, int EntryIndex, int UniqueId, Vector3 NewPosition) : AdtPlacementEdit
{
    public override PlacementEditKind Kind_ => PlacementEditKind.Move;
}

public sealed record AdtPlacementRotateEdit(AdtPlacementKind Kind, int EntryIndex, int UniqueId, Vector3 NewRotation) : AdtPlacementEdit
{
    public override PlacementEditKind Kind_ => PlacementEditKind.Rotate;
}

public sealed record AdtPlacementScaleEdit(AdtPlacementKind Kind, int EntryIndex, int UniqueId, float NewScale) : AdtPlacementEdit
{
    public override PlacementEditKind Kind_ => PlacementEditKind.Scale;
}

public sealed record AdtPlacementDeleteEdit(AdtPlacementKind Kind, int EntryIndex, int UniqueId) : AdtPlacementEdit
{
    public override PlacementEditKind Kind_ => PlacementEditKind.Delete;
}

public sealed record AdtPlacementAddEdit(AdtPlacementKind Kind, string ModelName, Vector3 Position, Vector3 Rotation, float Scale) : AdtPlacementEdit
{
    public override PlacementEditKind Kind_ => PlacementEditKind.Add;
}

/// <summary>Outcome of an in-memory placement edit: the updated bytes plus the reported remaps.</summary>
public sealed record AdtPlacementEditResult(
    byte[] Bytes,
    IReadOnlyList<string> AddedModelNames,
    IReadOnlyList<string> AddedWorldModelNames,
    IReadOnlyList<int> AllocatedIds);

/// <summary>
/// Library-first placement authoring over the existing ADT representation (Specs 175 and 176
/// Phase 2). It rebuilds only the placement and name-table chunks (MMDX/MMID/MDDF and
/// MWMO/MWID/MODF) and copies every other chunk byte-identically, so an edit never touches the
/// terrain/render data it did not intend to change. No ADT serializer is added — this completes the
/// placement write surface the CLI-only writers already expose.
/// </summary>
public static class AdtPlacementEditor
{
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;

    public static AdtPlacementEditResult Apply(byte[] source, string sourcePath, IReadOnlyList<AdtPlacementEdit> edits)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(edits);

        MapFileSummary summary;
        AdtPlacementCatalog catalog;
        using (MemoryStream stream = new(source, writable: false))
        {
            summary = MapFileSummaryReader.Read(stream, sourcePath);
            catalog = AdtPlacementReader.Read(stream, summary);
        }

        var modelNames = new List<string>(catalog.ModelNames);
        var wmoNames = new List<string>(catalog.WorldModelNames);

        var models = ToMutableModelRows(catalog.ModelPlacements);
        var wmos = ToMutableWmoRows(catalog.WorldModelPlacements);

        // Allocation continues the tile's existing uniqueId chronology even when earlier edits in
        // this batch deleted the highest-id row (e.g. a substitute's delete + add pair).
        int idHighWaterMark = Math.Max(MaxUniqueId(models), MaxUniqueId(wmos));

        var addedModelNames = new List<string>();
        var addedWmoNames = new List<string>();
        var allocatedIds = new List<int>();

        foreach (AdtPlacementEdit edit in edits)
        {
            switch (edit)
            {
                case AdtPlacementMoveEdit move:
                    ApplyMove(models, wmos, move);
                    break;
                case AdtPlacementRotateEdit rotate:
                    ApplyRotate(models, wmos, rotate);
                    break;
                case AdtPlacementScaleEdit scale:
                    ApplyScale(models, wmos, scale);
                    break;
                case AdtPlacementDeleteEdit delete:
                    ApplyDelete(models, wmos, delete);
                    break;
                case AdtPlacementAddEdit add:
                    ApplyAdd(modelNames, wmoNames, models, wmos, add, addedModelNames, addedWmoNames, allocatedIds, ref idHighWaterMark);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported placement edit {edit.GetType().Name}.");
            }
        }

        byte[] mmdx = SerializeStringBlock(modelNames);
        byte[] mmid = SerializeOffsets(modelNames);
        byte[] mwmo = SerializeStringBlock(wmoNames);
        byte[] mwid = SerializeOffsets(wmoNames);
        byte[] mddf = SerializeMddf(models);
        byte[] modf = SerializeModf(wmos);

        byte[] result = Reassemble(source, summary, mmdx, mmid, mwmo, mwid, mddf, modf);

        return new AdtPlacementEditResult(result, addedModelNames, addedWmoNames, allocatedIds);
    }

    // --- edit application ---

    private static void ApplyMove(List<MutableModelRow> models, List<MutableWmoRow> wmos, AdtPlacementMoveEdit edit)
    {
        if (edit.Kind == AdtPlacementKind.Model)
        {
            var row = FindModel(models, edit.EntryIndex, edit.UniqueId);
            row.Position = edit.NewPosition;
        }
        else
        {
            var row = FindWmo(wmos, edit.EntryIndex, edit.UniqueId);
            row.Position = edit.NewPosition;
        }
    }

    private static void ApplyRotate(List<MutableModelRow> models, List<MutableWmoRow> wmos, AdtPlacementRotateEdit edit)
    {
        if (edit.Kind == AdtPlacementKind.Model)
        {
            FindModel(models, edit.EntryIndex, edit.UniqueId).Rotation = edit.NewRotation;
        }
        else
        {
            FindWmo(wmos, edit.EntryIndex, edit.UniqueId).Rotation = edit.NewRotation;
        }
    }

    private static void ApplyScale(List<MutableModelRow> models, List<MutableWmoRow> wmos, AdtPlacementScaleEdit edit)
    {
        if (edit.Kind == AdtPlacementKind.Model)
        {
            FindModel(models, edit.EntryIndex, edit.UniqueId).Scale = edit.NewScale;
        }
        else
        {
            FindWmo(wmos, edit.EntryIndex, edit.UniqueId).ScaleRaw = ToScaleUshort(edit.NewScale);
        }
    }

    private static void ApplyDelete(List<MutableModelRow> models, List<MutableWmoRow> wmos, AdtPlacementDeleteEdit edit)
    {
        if (edit.Kind == AdtPlacementKind.Model)
        {
            int index = FindModelIndex(models, edit.EntryIndex, edit.UniqueId);
            models.RemoveAt(index);
        }
        else
        {
            int index = FindWmoIndex(wmos, edit.EntryIndex, edit.UniqueId);
            wmos.RemoveAt(index);
        }
    }

    private static void ApplyAdd(
        List<string> modelNames,
        List<string> wmoNames,
        List<MutableModelRow> models,
        List<MutableWmoRow> wmos,
        AdtPlacementAddEdit edit,
        List<string> addedModelNames,
        List<string> addedWmoNames,
        List<int> allocatedIds,
        ref int idHighWaterMark)
    {
        int uniqueId = ++idHighWaterMark;

        if (edit.Kind == AdtPlacementKind.Model)
        {
            int nameId = ResolveOrAddName(modelNames, edit.ModelName, addedModelNames);
            models.Add(new MutableModelRow
            {
                NameId = nameId,
                UniqueId = uniqueId,
                Position = edit.Position,
                Rotation = edit.Rotation,
                Scale = edit.Scale,
            });
        }
        else
        {
            int nameId = ResolveOrAddName(wmoNames, edit.ModelName, addedWmoNames);
            wmos.Add(new MutableWmoRow
            {
                NameId = nameId,
                UniqueId = uniqueId,
                Position = edit.Position,
                Rotation = edit.Rotation,
                ScaleRaw = ToScaleUshort(edit.Scale),
                Flags = 0,
                DoodadSet = 0,
                NameSet = 0,
            });
        }

        allocatedIds.Add(uniqueId);
    }

    private static int MaxUniqueId(List<MutableModelRow> models)
    {
        int max = 0;
        foreach (var model in models)
            max = Math.Max(max, model.UniqueId);
        return max;
    }

    private static int MaxUniqueId(List<MutableWmoRow> wmos)
    {
        int max = 0;
        foreach (var wmo in wmos)
            max = Math.Max(max, wmo.UniqueId);
        return max;
    }

    private static int ResolveOrAddName(List<string> names, string name, List<string> addedNames)
    {
        for (int index = 0; index < names.Count; index++)
        {
            if (string.Equals(names[index], name, StringComparison.Ordinal))
                return index;
        }

        names.Add(name);
        addedNames.Add(name);
        return names.Count - 1;
    }

    // --- row models ---

    private sealed class MutableModelRow
    {
        public int NameId;
        public int UniqueId;
        public Vector3 Position;
        public Vector3 Rotation;
        public float Scale;
    }

    private sealed class MutableWmoRow
    {
        public int NameId;
        public int UniqueId;
        public Vector3 Position;
        public Vector3 Rotation;
        public ushort ScaleRaw;
        public ushort Flags;
        public ushort DoodadSet;
        public ushort NameSet;
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
    }

    private static List<MutableModelRow> ToMutableModelRows(IReadOnlyList<AdtModelPlacement> placements)
    {
        var rows = new List<MutableModelRow>(placements.Count);
        foreach (AdtModelPlacement placement in placements)
        {
            rows.Add(new MutableModelRow
            {
                NameId = placement.NameId,
                UniqueId = placement.UniqueId,
                Position = placement.Position,
                Rotation = placement.Rotation,
                Scale = placement.Scale,
            });
        }

        return rows;
    }

    private static List<MutableWmoRow> ToMutableWmoRows(IReadOnlyList<AdtWorldModelPlacement> placements)
    {
        var rows = new List<MutableWmoRow>(placements.Count);
        foreach (AdtWorldModelPlacement placement in placements)
        {
            rows.Add(new MutableWmoRow
            {
                NameId = placement.NameId,
                UniqueId = placement.UniqueId,
                Position = placement.Position,
                Rotation = placement.Rotation,
                ScaleRaw = placement.Scale,
                Flags = placement.Flags,
                DoodadSet = placement.DoodadSet,
                NameSet = placement.NameSet,
                BoundsMin = placement.BoundsMin,
                BoundsMax = placement.BoundsMax,
            });
        }

        return rows;
    }

    private static MutableModelRow FindModel(List<MutableModelRow> rows, int entryIndex, int uniqueId)
    {
        int index = FindModelIndex(rows, entryIndex, uniqueId);
        return rows[index];
    }

    private static int FindModelIndex(List<MutableModelRow> rows, int entryIndex, int uniqueId)
    {
        if (entryIndex < 0 || entryIndex >= rows.Count)
            throw new ArgumentOutOfRangeException(nameof(entryIndex), $"MDDF entry index {entryIndex} is outside 0..{rows.Count - 1}.");

        if (rows[entryIndex].UniqueId != uniqueId)
            throw new InvalidDataException($"MDDF entry {entryIndex} no longer matches UniqueId {uniqueId}; found {rows[entryIndex].UniqueId}.");

        return entryIndex;
    }

    private static MutableWmoRow FindWmo(List<MutableWmoRow> rows, int entryIndex, int uniqueId)
    {
        int index = FindWmoIndex(rows, entryIndex, uniqueId);
        return rows[index];
    }

    private static int FindWmoIndex(List<MutableWmoRow> rows, int entryIndex, int uniqueId)
    {
        if (entryIndex < 0 || entryIndex >= rows.Count)
            throw new ArgumentOutOfRangeException(nameof(entryIndex), $"MODF entry index {entryIndex} is outside 0..{rows.Count - 1}.");

        if (rows[entryIndex].UniqueId != uniqueId)
            throw new InvalidDataException($"MODF entry {entryIndex} no longer matches UniqueId {uniqueId}; found {rows[entryIndex].UniqueId}.");

        return entryIndex;
    }

    // --- serialization ---

    private static byte[] SerializeStringBlock(IReadOnlyList<string> names)
    {
        using MemoryStream stream = new();
        foreach (string name in names)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(name);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] SerializeOffsets(IReadOnlyList<string> names)
    {
        byte[] bytes = new byte[names.Count * sizeof(uint)];
        int offset = 0;
        for (int index = 0; index < names.Count; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint), sizeof(uint)), (uint)offset);
            offset += Encoding.ASCII.GetByteCount(names[index]) + 1;
        }

        return bytes;
    }

    private static byte[] SerializeMddf(IReadOnlyList<MutableModelRow> models)
    {
        byte[] bytes = new byte[models.Count * MddfEntrySize];
        for (int index = 0; index < models.Count; index++)
        {
            MutableModelRow row = models[index];
            int offset = index * MddfEntrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset, 4), (uint)row.NameId);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 4, 4), (uint)row.UniqueId);
            WriteSingle(bytes, offset + 8, MapOrigin - row.Position.Y);
            WriteSingle(bytes, offset + 12, row.Position.Z);
            WriteSingle(bytes, offset + 16, MapOrigin - row.Position.X);
            WriteSingle(bytes, offset + 20, row.Rotation.X);
            WriteSingle(bytes, offset + 24, row.Rotation.Z);
            WriteSingle(bytes, offset + 28, row.Rotation.Y);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 32, 2), ToScaleUshort(row.Scale));
        }

        return bytes;
    }

    private static byte[] SerializeModf(IReadOnlyList<MutableWmoRow> wmos)
    {
        byte[] bytes = new byte[wmos.Count * ModfEntrySize];
        for (int index = 0; index < wmos.Count; index++)
        {
            MutableWmoRow row = wmos[index];
            int offset = index * ModfEntrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset, 4), (uint)row.NameId);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 4, 4), (uint)row.UniqueId);
            WriteSingle(bytes, offset + 8, MapOrigin - row.Position.Y);
            WriteSingle(bytes, offset + 12, row.Position.Z);
            WriteSingle(bytes, offset + 16, MapOrigin - row.Position.X);
            WriteSingle(bytes, offset + 20, row.Rotation.X);
            WriteSingle(bytes, offset + 24, row.Rotation.Z);
            WriteSingle(bytes, offset + 28, row.Rotation.Y);
            WriteSingle(bytes, offset + 32, MapOrigin - row.BoundsMax.Y);
            WriteSingle(bytes, offset + 36, row.BoundsMin.Z);
            WriteSingle(bytes, offset + 40, MapOrigin - row.BoundsMax.X);
            WriteSingle(bytes, offset + 44, MapOrigin - row.BoundsMin.Y);
            WriteSingle(bytes, offset + 48, row.BoundsMax.Z);
            WriteSingle(bytes, offset + 52, MapOrigin - row.BoundsMin.X);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 56, 2), row.Flags);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 58, 2), row.DoodadSet);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 60, 2), row.NameSet);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 62, 2), row.ScaleRaw);
        }

        return bytes;
    }

    private static ushort ToScaleUshort(float scale)
        => (ushort)Math.Clamp((int)MathF.Round(scale * 1024f), ushort.MinValue, ushort.MaxValue);

    private static byte[] Reassemble(
        byte[] source,
        MapFileSummary summary,
        byte[] mmdx,
        byte[] mmid,
        byte[] mwmo,
        byte[] mwid,
        byte[] mddf,
        byte[] modf)
    {
        var rebuilt = new Dictionary<FourCC, byte[]>
        {
            [MapChunkIds.Mmdx] = mmdx,
            [MapChunkIds.Mmid] = mmid,
            [MapChunkIds.Mwmo] = mwmo,
            [MapChunkIds.Mwid] = mwid,
            [MapChunkIds.Mddf] = mddf,
            [MapChunkIds.Modf] = modf,
        };

        using MemoryStream output = new(source.Length + 128);
        var emitted = new HashSet<FourCC>();

        foreach (MapChunkLocation chunk in summary.Chunks)
        {
            if (rebuilt.TryGetValue(chunk.Id, out byte[]? newPayload))
            {
                if (newPayload.Length > 0)
                    WriteChunk(output, chunk.Id, newPayload);
                emitted.Add(chunk.Id);
            }
            else
            {
                CopyChunk(output, source, chunk);
            }
        }

        // Append any placement chunk that did not exist in the source (e.g. first MDDF added).
        foreach ((FourCC id, byte[] payload) in rebuilt)
        {
            if (!emitted.Contains(id) && payload.Length > 0)
                WriteChunk(output, id, payload);
        }

        return output.ToArray();
    }

    private static void CopyChunk(Stream output, byte[] source, MapChunkLocation chunk)
    {
        int headerOffset = checked((int)chunk.HeaderOffset);
        int length = checked((int)chunk.EndOffset - headerOffset);
        output.Write(source, headerOffset, length);
    }

    private static void WriteChunk(Stream output, FourCC id, byte[] payload)
    {
        output.Write(id.ToFileBytes(), 0, 4);
        Span<byte> size = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(size, (uint)payload.Length);
        output.Write(size);
        output.Write(payload, 0, payload.Length);
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}