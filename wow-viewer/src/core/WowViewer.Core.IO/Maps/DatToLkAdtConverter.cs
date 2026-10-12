using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>Options for <see cref="DatToLkAdtConverter"/>.</summary>
public sealed class DatToLkConversionOptions
{
    /// <summary>DAT heights are in inches; 36 converts to the ADT's yards (see AhdrTerrainAdapter.HeightDivisor).</summary>
    public float HeightDivisor { get; init; } = 36f;

    /// <summary>Carry ACDO object placements into MDDF/MODF.</summary>
    public bool IncludeObjects { get; init; } = true;

    /// <summary>
    /// Swap ACNK IndexX/IndexY when writing MCNK. The DAT grid's row axis runs along ALOC tile Y while the
    /// renderer's runs along its tile X (see AhdrTerrainAdapter), so which way round a standalone ADT should be
    /// written is not established by measurement. Default false keeps the source's own indices.
    /// </summary>
    public bool TransposeChunks { get; init; }
}

/// <summary>What the conversion carried and what it could not, for the §9.2 receipt and the user-facing manifest.</summary>
public sealed class DatToLkConversionReport
{
    public int ChunksInSource { get; set; }
    public int ChunksWritten { get; set; }
    public int ChunksSynthesizedEmpty { get; set; }
    public int LayersWritten { get; set; }
    public int LayersDroppedNoAlpha { get; set; }
    public int AlphaMapsWritten { get; set; }
    public int ShadowMapsCarried { get; set; }
    public int AreaIdsCarried { get; set; }
    public int VertexColourChunks { get; set; }
    public int ObjectsPlaced { get; set; }
    public int ObjectsSkippedUnnamed { get; set; }

    /// <summary>
    /// Placements whose DAT uniqueId is negative. v26 allocates 22.85% of its ids as a descending
    /// negative block (-1,233..-2) alongside the positive one; LK MDDF uniqueId is a uint32, so these are
    /// written as very large unsigned values. See spec 237 evidence/acdo-negative-uniqueids-2026-09-20.md.
    /// </summary>
    public int ObjectsWithNegativeUniqueId { get; set; }

    /// <summary>ADST rows seen: uniqueId + model FileDataID with no position. LK has no equivalent chunk.</summary>
    public int AdstRowsDropped { get; set; }

    /// <summary>True when any source tile carried an AOCH chunk (2048 bytes, all-zero and unexplained).</summary>
    public bool SawAoch { get; set; }
    public List<string> Notes { get; } = [];

    public void Note(string note)
    {
        if (!Notes.Contains(note))
            Notes.Add(note);
    }
}

/// <summary>
/// Spec 247: one-way conversion of an AHDR-family DAT tile (v22/v23/v26) to Wrath-of-the-Lich-King ADT data,
/// so recovered DAT terrain can be saved as files other tools read. Writing is left to
/// <see cref="LkAdtWriter"/>; this only maps the model.
/// <para>
/// Chunks are addressed by their <c>ACNK</c> index fields, never by ordinal position: v22 omits empty chunks
/// (243-255 of 256 observed), so the n-th ACNK is not chunk n. Missing chunks are written as empty MCNK so the
/// ADT keeps its full 16x16 grid.
/// </para>
/// <para>
/// Alpha: v23/v26 store per-layer <b>weights</b> in a 4096-byte AMAP, converted to the ADT's sequential alpha by
/// <see cref="AdtAhdrAlpha.WeightsToSequentialAlpha"/> and written as big (8-bit, uncompressed) MCAL. v22's AMAP
/// is an unidentified encoding, so no alpha is available; rather than emit opaque upper layers that would hide
/// layer 0, only layer 0 is written and the loss is reported.
/// </para>
/// </summary>
public static class DatToLkAdtConverter
{
    private const int VerticesPerChunk = 145;
    private const int McnrByteCount = 448;
    private const int AlphaBytes = 64 * 64;
    private const float ChunkSize = 533.33333f;
    private const float ChunkSubSize = ChunkSize / 16f;
    private const float MapOrigin = 32f * ChunkSize;
    private const float VertexSpacing = ChunkSubSize / 8f;

    /// <summary>MCLY flag: this layer has an alpha map in MCAL.</summary>
    private const uint MclyUseAlphaMap = 0x100;

    /// <summary>MCNK flag: MCSH is present.</summary>
    private const uint McnkHasMcsh = 0x01;

    public static LkAdtData Convert(
        AdtAhdrTile tile,
        int tileX,
        int tileY,
        string mapName,
        DatToLkConversionOptions? options,
        DatToLkConversionReport report)
    {
        ArgumentNullException.ThrowIfNull(tile);
        ArgumentNullException.ThrowIfNull(report);
        options ??= new DatToLkConversionOptions();

        float divisor = options.HeightDivisor > 0f ? options.HeightDivisor : 1f;
        report.ChunksInSource += tile.Chunks.Count;

        // ACNK index -> source chunk. v22 omits empties, so this is a lookup, not an ordinal walk.
        var byIndex = new Dictionary<(int X, int Y), AdtAhdrChunk>();
        foreach (AdtAhdrChunk c in tile.Chunks)
            byIndex.TryAdd((c.IndexX, c.IndexY), c);

        var textureNames = new List<string>(tile.TextureNames);
        var modelNames = new List<string>();
        var worldModelNames = new List<string>();
        var modelPlacements = new List<LkMddfEntry>();
        var worldModelPlacements = new List<LkModfEntry>();
        var modelIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var worldModelIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        var chunks = new List<LkMcnkData>(256);
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                (int sx, int sy) = options.TransposeChunks ? (cy, cx) : (cx, cy);
                if (!byIndex.TryGetValue((sx, sy), out AdtAhdrChunk? source))
                {
                    chunks.Add(EmptyChunk(cx, cy, tileX, tileY));
                    report.ChunksSynthesizedEmpty++;
                    continue;
                }

                chunks.Add(BuildChunk(tile, source, cx, cy, sx, sy, tileX, tileY, divisor, options, report,
                    modelNames, worldModelNames, modelPlacements, worldModelPlacements, modelIndex, worldModelIndex));
                report.ChunksWritten++;
            }
        }

        if (tile.Version == 22)
            report.Note("v22: AMAP is an unidentified encoding, so only layer 0 is written per chunk (spec 247 FR-001).");
        if (tile.Chunks.Count < 256)
            report.Note($"Source omits {256 - tile.Chunks.Count} empty ACNK; written as empty MCNK to keep a full 16x16 grid.");
        report.Note($"Heights divided by {divisor:0.##} (DAT inches -> ADT yards).");
        report.Note("Not carried: liquids (no liquid chunk exists in any DAT sample) and MFBO flight bounds.");
        report.AdstRowsDropped += tile.ModelFileReferences.Count;
        report.SawAoch |= tile.AochRaw is { Length: > 0 };


        return new LkAdtData
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY,
            TextureNames = textureNames,
            ModelNames = modelNames,
            WorldModelNames = worldModelNames,
            ModelPlacements = modelPlacements,
            WorldModelPlacements = worldModelPlacements,
            Chunks = chunks,
        };
    }

    private static LkMcnkData BuildChunk(
        AdtAhdrTile tile, AdtAhdrChunk source, int cx, int cy, int sx, int sy, int tileX, int tileY,
        float divisor, DatToLkConversionOptions options, DatToLkConversionReport report,
        List<string> modelNames, List<string> worldModelNames,
        List<LkMddfEntry> modelPlacements, List<LkModfEntry> worldModelPlacements,
        Dictionary<string, int> modelIndex, Dictionary<string, int> worldModelIndex)
    {
        float[] heights = AdtAhdrTileSlicer.SliceHeights(tile, sx, sy);
        if (heights.Length != VerticesPerChunk)
            heights = new float[VerticesPerChunk];

        for (int i = 0; i < heights.Length; i++)
            heights[i] /= divisor;

        float baseHeight = MinNonZero(heights);
        for (int i = 0; i < heights.Length; i++)
            heights[i] -= baseHeight;

        byte[] normals = EncodeNormals(AdtAhdrTileSlicer.SliceStoredNormals(tile, sx, sy));

        byte[]? shadow = source.ShadowRaw is { Length: 512 } s && AnyNonZero(s) ? s.ToArray() : null;
        if (shadow is not null)
            report.ShadowMapsCarried++;

        int areaId = ReadHeaderInt32(source.HeaderRaw, 0x0C);
        if (areaId != 0)
            report.AreaIdsCarried++;

        int holeMask = ReadHeaderUInt16(source.HeaderRaw, 0x10);

        byte[]? mccv = AdtAhdrTileSlicer.SliceVertexColors(tile, sx, sy);
        if (mccv is not null)
            report.VertexColourChunks++;

        (List<LkMclyEntry> layers, byte[]? alphaData) = BuildLayers(source, report);
        report.LayersWritten += layers.Count;
        if (alphaData is { Length: > 0 })
            report.AlphaMapsWritten += alphaData.Length / AlphaBytes;

        var doodadRefs = new List<int>();
        var worldModelRefs = new List<int>();
        if (options.IncludeObjects)
        {
            AddObjects(tile, source, divisor, tileX, tileY, report,
                modelNames, worldModelNames, modelPlacements, worldModelPlacements,
                modelIndex, worldModelIndex, doodadRefs, worldModelRefs);
        }

        uint flags = 0;
        if (shadow is not null)
            flags |= McnkHasMcsh;

        return new LkMcnkData
        {
            IndexX = cx,
            IndexY = cy,
            Flags = (int)flags,
            AreaId = areaId,
            NLayers = layers.Count,
            HoleMask = holeMask,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = normals,
            ShadowMap = shadow,
            AlphaMapData = alphaData,
            AlphaMapSize = alphaData?.Length ?? 0,
            Layers = layers,
            DoodadRefs = doodadRefs,
            WorldModelRefs = worldModelRefs,
            MccvColors = mccv,
            PosX = -((ChunkSubSize * cy) + ChunkSize * tileY - ChunkSize * 32f),
            PosY = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32f),
            PosZ = baseHeight,
        };
    }

    /// <summary>
    /// MCLY plus the packed MCAL payload. Alpha is only available when every source layer carries a 4096-byte
    /// AMAP (v23/v26), because the weight->sequential conversion needs all layers at once. Otherwise only layer 0
    /// is emitted: writing upper layers with no alpha would make them opaque and hide the base layer entirely.
    /// </summary>
    private static (List<LkMclyEntry> Layers, byte[]? Alpha) BuildLayers(AdtAhdrChunk source, DatToLkConversionReport report)
    {
        var layers = new List<LkMclyEntry>();
        if (source.Layers.Count == 0)
            return (layers, null);

        bool everyLayerHasWeights = source.Layers.Count > 1
            && source.Layers.All(static l => l.AlphaMap is { Length: AlphaBytes });

        if (!everyLayerHasWeights)
        {
            AdtAhdrLayer first = source.Layers[0];
            layers.Add(new LkMclyEntry((uint)first.TextureIndex, 0, 0, 0));
            if (source.Layers.Count > 1)
                report.LayersDroppedNoAlpha += source.Layers.Count - 1;
            return (layers, null);
        }

        byte[][] sequential = AdtAhdrAlpha.WeightsToSequentialAlpha(
            source.Layers.Select(static l => l.AlphaMap!).ToArray());

        var alpha = new byte[sequential.Length * AlphaBytes];
        layers.Add(new LkMclyEntry((uint)source.Layers[0].TextureIndex, 0, 0, 0));
        for (int i = 1; i < source.Layers.Count; i++)
        {
            int offset = (i - 1) * AlphaBytes;
            sequential[i - 1].CopyTo(alpha, offset);
            layers.Add(new LkMclyEntry((uint)source.Layers[i].TextureIndex, MclyUseAlphaMap, (uint)offset, 0));
        }

        return (layers, alpha);
    }

    /// <summary>
    /// ACDO -> MDDF/MODF using the frame proven in AhdrTerrainAdapter: renderer TileX follows the grid row axis
    /// and TileY the column axis, and ACDO rotation is in position order (column, vertical, row) like MDDF's.
    /// </summary>
    private static void AddObjects(
        AdtAhdrTile tile, AdtAhdrChunk source, float divisor, int tileX, int tileY,
        DatToLkConversionReport report,
        List<string> modelNames, List<string> worldModelNames,
        List<LkMddfEntry> modelPlacements, List<LkModfEntry> worldModelPlacements,
        Dictionary<string, int> modelIndex, Dictionary<string, int> worldModelIndex,
        List<int> doodadRefs, List<int> worldModelRefs)
    {
        foreach (AdtAhdrObjectDefinition obj in source.Objects)
        {
            if ((uint)obj.ModelIndex >= (uint)tile.ModelNames.Count
                || string.IsNullOrWhiteSpace(tile.ModelNames[obj.ModelIndex]))
            {
                report.ObjectsSkippedUnnamed++;
                continue;
            }

            (float column, float row, float height) = AdtAhdrTileSlicer.ResolveObjectGridPosition(tile, source, obj);
            var position = new Vector3(
                MapOrigin - tileX * ChunkSize - row * VertexSpacing,
                MapOrigin - tileY * ChunkSize - column * VertexSpacing,
                height / divisor);
            var rotation = new Vector3(obj.RotationDegrees.X, obj.RotationDegrees.Z, obj.RotationDegrees.Y);
            string name = tile.ModelNames[obj.ModelIndex];

            if (name.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            {
                worldModelRefs.Add(worldModelPlacements.Count);
                worldModelPlacements.Add(new LkModfEntry(
                    GetOrAdd(name, worldModelNames, worldModelIndex),
                    unchecked((int)obj.UniqueId),
                    position, rotation, position, position, 0, 0, 0, 1024f));
            }
            else
            {
                doodadRefs.Add(modelPlacements.Count);
                modelPlacements.Add(new LkMddfEntry(
                    GetOrAdd(name, modelNames, modelIndex),
                    unchecked((int)obj.UniqueId),
                    position, rotation,
                    obj.Scale > 0f ? obj.Scale : 1f));
            }

            if (unchecked((int)obj.UniqueId) < 0)
                report.ObjectsWithNegativeUniqueId++;

            report.ObjectsPlaced++;
        }
    }

    private static LkMcnkData EmptyChunk(int cx, int cy, int tileX, int tileY) => new()
    {
        IndexX = cx,
        IndexY = cy,
        Heights = new float[VerticesPerChunk],
        Normals = FlatNormals(),
        PosX = -((ChunkSubSize * cy) + ChunkSize * tileY - ChunkSize * 32f),
        PosY = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32f),
        PosZ = 0f,
    };

    private static byte[] FlatNormals()
    {
        var n = new byte[McnrByteCount];
        for (int i = 0; i < VerticesPerChunk; i++)
            n[(i * 3) + 2] = 127; // +Z up
        return n;
    }

    /// <summary>
    /// LK MCNR stores (X, Z, Y) of a Z-up normal as signed bytes with 127 = 1.0, matching
    /// <see cref="AlphaToLkConverter"/>. <see cref="AdtAhdrTileSlicer.SliceStoredNormals"/> already returns the
    /// renderer's Z-up frame, so only the component order changes.
    /// </summary>
    private static byte[] EncodeNormals(Vector3[]? normals)
    {
        if (normals is null || normals.Length != VerticesPerChunk)
            return FlatNormals();

        var result = new byte[McnrByteCount];
        for (int i = 0; i < VerticesPerChunk; i++)
        {
            Vector3 n = normals[i];
            result[i * 3] = EncodeComponent(n.X);
            result[(i * 3) + 1] = EncodeComponent(n.Z);
            result[(i * 3) + 2] = EncodeComponent(n.Y);
        }

        return result;
    }

    private static byte EncodeComponent(float value)
        => unchecked((byte)(sbyte)Math.Clamp(MathF.Round(value * 127f), -128, 127));

    private static int GetOrAdd(string name, List<string> names, Dictionary<string, int> index)
    {
        if (index.TryGetValue(name, out int existing))
            return existing;

        int added = names.Count;
        names.Add(name);
        index[name] = added;
        return added;
    }

    private static float MinNonZero(float[] values)
    {
        float min = float.MaxValue;
        foreach (float v in values)
        {
            if (v != 0f && v < min)
                min = v;
        }

        return min == float.MaxValue ? 0f : min;
    }

    private static bool AnyNonZero(byte[] data)
    {
        foreach (byte b in data)
        {
            if (b != 0)
                return true;
        }

        return false;
    }

    private static int ReadHeaderInt32(byte[] header, int offset)
        => header.Length >= offset + 4 ? BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(offset)) : 0;

    private static int ReadHeaderUInt16(byte[] header, int offset)
        => header.Length >= offset + 2 ? BinaryPrimitives.ReadUInt16LittleEndian(header.AsSpan(offset)) : 0;
}
