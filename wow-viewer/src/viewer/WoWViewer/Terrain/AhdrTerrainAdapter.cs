using System.Collections.Concurrent;
using System.Numerics;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WoWViewer.Terrain;

/// <summary>
/// Spec 237: terrain adapter over a folder of AHDR-family terrain files (DAT v26). The files carry
/// no usable names and no WDT exists, so every file is content-sniffed and placed by its ALOC tile
/// coordinates.
/// <para>
/// Axis mapping: v26 grid rows (and ACNK IndexY) run along ALOC tile Y, while the renderer's chunk
/// rows run along its tile X (see <see cref="TerrainMeshBuilder"/>). The adapter therefore uses
/// renderer TileX = ALOC tile Y and renderer TileY = ALOC tile X, which keeps every tile untransposed
/// and seam-exact. Whether the result is mirrored relative to the game world is not yet verified.
/// </para>
/// <para>
/// Objects come from ACDO (frame measured, see <see cref="AdtAhdrObjectDefinition"/>): M2 names become
/// <see cref="MddfPlacements"/> and WMO names <see cref="ModfPlacements"/>, positioned on the same grid as the
/// terrain. ADST rows (uniqueId → model FileDataID) have no position and are not placed.
/// Rotation follows the MDDF axis order; whether yaw is mirrored with the terrain is not yet verified.
/// </para>
/// <para>
/// Normals come from ANRM at the ÷36 display scale (component order measured) and vertex colours from ACVT
/// (MCCV-like, red/blue order unverified). ASHD is not rendered: it is all zero in the DAT v26 corpus.
/// </para>
/// </summary>
public sealed class AhdrTerrainAdapter : ITerrainAdapter
{
    private readonly Dictionary<int, string> _pathByTileKey = [];
    private readonly List<int> _existingTiles = [];
    private readonly List<PhaseLayerSettings> _phaseLayers = [];

    public AhdrTerrainAdapter(string folder, float heightDivisor = 1f)
    {
        Folder = folder;
        HeightDivisor = heightDivisor > 0f ? heightDivisor : 1f;
        foreach (string path in Directory.EnumerateFiles(folder))
        {
            byte[] head = ReadHead(path, 256);
            if (!AdtAhdrReader.IsAhdrFamily(head))
                continue;

            // v22/v23 DAT files (e.g. the Wrath "area_*" corpus) carry the same AHDR-family
            // vocabulary as v26 but no ALOC chunk, so the tile location must come from the file
            // name. v26 keeps its ALOC and is unaffected.
            if (!AdtAhdrReader.TryReadTileLocation(head, out int alocX, out int alocY)
                && !AdtAhdrReader.TryParseTileLocationFromName(path, out alocX, out alocY))
            {
                SkippedFiles.Add($"{Path.GetFileName(path)}: no ALOC and no tile coordinates in the file name");
                continue;
            }

            if (alocX is < 0 or >= 64 || alocY is < 0 or >= 64)
            {
                SkippedFiles.Add($"{Path.GetFileName(path)}: tile ({alocX}, {alocY}) outside the 64x64 grid");
                continue;
            }

            int key = TileKey(alocY, alocX);
            if (!_pathByTileKey.TryAdd(key, path))
            {
                SkippedFiles.Add($"{Path.GetFileName(path)}: duplicate of tile ALOC ({alocX}, {alocY})");
                continue;
            }

            _existingTiles.Add(key);
        }

        _existingTiles.Sort();
    }

    public string Folder { get; }

    private const float InchesPerYard = 36f;

    /// <summary>
    /// Display divisor applied to DAT v26 heights, which are in inches (÷36 = yards): the corpus's
    /// 5th-percentile height −18559.47 ÷ 36 = −515.54, matching the shipped Azeroth ADT ocean floor
    /// (−515.19..−516.07 yd). Horizontally a chunk is 1200 inches (33.33 yd), the same span as an ADT chunk,
    /// which ACDO object positions confirm.
    /// </summary>
    public float HeightDivisor { get; }

    /// <summary>Files that were AHDR-family but not placed, with the reason.</summary>
    public List<string> SkippedFiles { get; } = [];

    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    private readonly object _placementLock = new();
    private readonly List<string> _mdxNames = [];
    private readonly List<string> _wmoNames = [];
    private readonly Dictionary<string, int> _mdxNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _placedUniqueIds = [];

    public IReadOnlyList<string> MdxModelNames => _mdxNames;

    public IReadOnlyList<string> WmoModelNames => _wmoNames;

    /// <summary>ADST rows seen in loaded tiles (not placed: they carry no position).</summary>
    public int ModelFileReferenceCount { get; private set; }

    public List<MddfPlacement> MddfPlacements { get; } = [];

    public List<ModfPlacement> ModfPlacements { get; } = [];

    public bool IsWmoBased => false;

    public List<Vector3> LastLoadedChunkPositions { get; } = [];

    public IList<PhaseLayerSettings> PhaseLayers => _phaseLayers;

    public string? OverlayMapName
    {
        get => null;
        set { }
    }

    public bool TileExists(int tileX, int tileY) => _pathByTileKey.ContainsKey(TileKey(tileX, tileY));

    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        if (!_pathByTileKey.TryGetValue(TileKey(tileX, tileY), out string? path))
            return new TileLoadResult();

        AdtAhdrTile tile = AdtAhdrReader.Read(File.ReadAllBytes(path), path);
        foreach (string diagnostic in tile.Diagnostics)
            ViewerLog.Trace($"[AhdrTerrainAdapter] {Path.GetFileName(path)}: {diagnostic}");

        TileTextures[(tileX, tileY)] = tile.TextureNames.ToList();

        const float tileSpan = WoWConstants.ChunkSize;
        float chunkSpan = tileSpan / 16f;
        float vertexSpacing = chunkSpan / 8f;
        var chunks = new List<TerrainChunkData>(tile.Chunks.Count);
        for (int i = 0; i < tile.Chunks.Count; i++)
        {
            AdtAhdrChunk source = tile.Chunks[i];
            int gridColumn = source.IndexX; // along ALOC tile X  -> renderer ChunkX (world Y axis)
            int gridRow = source.IndexY;    // along ALOC tile Y  -> renderer ChunkY (world X axis)

            float worldX = WoWConstants.MapOrigin - tileX * tileSpan - gridRow * chunkSpan;
            float worldY = WoWConstants.MapOrigin - tileY * tileSpan - gridColumn * chunkSpan;
            LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));

            var layers = new TerrainLayer[source.Layers.Count];
            var alphaMaps = new Dictionary<int, byte[]>();
            for (int layerIndex = 0; layerIndex < source.Layers.Count; layerIndex++)
            {
                AdtAhdrLayer layer = source.Layers[layerIndex];
                layers[layerIndex] = new TerrainLayer { TextureIndex = layer.TextureIndex, Flags = layer.Flags };
            }

            // AMAP holds per-layer blend weights (sum 255 per pixel, layer 0 included); the renderer blends ADT-style
            // sequential alpha, so convert (see AdtAhdrAlpha).
            if (source.Layers.Count > 1 && source.Layers.All(static l => l.AlphaMap is { Length: AdtAhdrAlpha.Pixels }))
            {
                byte[][] sequential = AdtAhdrAlpha.WeightsToSequentialAlpha(source.Layers.Select(static l => l.AlphaMap!).ToArray());
                for (int layerIndex = 1; layerIndex < source.Layers.Count; layerIndex++)
                    alphaMaps[layerIndex] = sequential[layerIndex - 1];
            }

            chunks.Add(new TerrainChunkData
            {
                McinIndex = gridRow * 16 + gridColumn,
                TileX = tileX,
                TileY = tileY,
                ChunkX = gridColumn,
                ChunkY = gridRow,
                Heights = ScaleHeights(AdtAhdrTileSlicer.SliceHeights(tile, gridColumn, gridRow)),
                // Stored ANRM normals describe the true (inch) surface, so they apply at the ÷36 scale. Other
                // display scales flatten slopes by D, so normals are recomputed with the spacing widened by D.
                Normals = (HeightDivisor == InchesPerYard ? AdtAhdrTileSlicer.SliceStoredNormals(tile, gridColumn, gridRow) : null)
                    ?? AdtAhdrTileSlicer.ComputeNormals(tile, gridColumn, gridRow, vertexSpacing * HeightDivisor),
                MccvColors = AdtAhdrTileSlicer.SliceVertexColors(tile, gridColumn, gridRow),
                Layers = layers,
                AlphaMaps = alphaMaps,
                WorldPosition = new Vector3(worldX, worldY, 0f),
            });
        }

        var result = new TileLoadResult { Chunks = chunks };
        AddObjectPlacements(tile, tileX, tileY, vertexSpacing, result);
        return result;
    }

    private void AddObjectPlacements(AdtAhdrTile tile, int tileX, int tileY, float vertexSpacing, TileLoadResult result)
    {
        const float tileSpan = WoWConstants.ChunkSize;
        int skipped = 0;
        lock (_placementLock)
        {
            ModelFileReferenceCount += tile.ModelFileReferences.Count;
            foreach (AdtAhdrChunk chunk in tile.Chunks)
            {
                foreach (AdtAhdrObjectDefinition obj in chunk.Objects)
                {
                    if ((uint)obj.ModelIndex >= (uint)tile.ModelNames.Count || string.IsNullOrWhiteSpace(tile.ModelNames[obj.ModelIndex]))
                    {
                        skipped++;
                        continue;
                    }

                    (float column, float row, float height) = AdtAhdrTileSlicer.ResolveObjectGridPosition(tile, chunk, obj);

                    // Same mapping as the terrain vertices: renderer TileX = ALOC Y (grid rows), TileY = ALOC X (grid columns).
                    var position = new Vector3(
                        WoWConstants.MapOrigin - tileX * tileSpan - row * vertexSpacing,
                        WoWConstants.MapOrigin - tileY * tileSpan - column * vertexSpacing,
                        height / HeightDivisor);

                    // ACDO rotation is in position order (column axis, vertical, row axis), like MDDF's (X, Z, Y).
                    var rotation = new Vector3(obj.RotationDegrees.X, obj.RotationDegrees.Z, obj.RotationDegrees.Y);
                    string modelName = tile.ModelNames[obj.ModelIndex];
                    bool firstSighting = _placedUniqueIds.Add(obj.UniqueId);

                    if (modelName.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    {
                        var placement = new ModfPlacement
                        {
                            NameIndex = GetOrAddName(modelName, _wmoNames, _wmoNameIndex),
                            UniqueId = unchecked((int)obj.UniqueId),
                            Position = position,
                            Rotation = rotation,
                            BoundsMin = position,
                            BoundsMax = position,
                        };
                        if (firstSighting)
                            ModfPlacements.Add(placement);
                        result.ModfPlacements.Add(placement);
                    }
                    else
                    {
                        var placement = new MddfPlacement
                        {
                            NameIndex = GetOrAddName(modelName, _mdxNames, _mdxNameIndex),
                            UniqueId = unchecked((int)obj.UniqueId),
                            Position = position,
                            Rotation = rotation,
                            Scale = obj.Scale > 0f ? obj.Scale : 1f,
                        };
                        if (firstSighting)
                            MddfPlacements.Add(placement);
                        result.MddfPlacements.Add(placement);
                    }
                }
            }
        }

        int placed = result.MddfPlacements.Count + result.ModfPlacements.Count;
        if (placed > 0 || skipped > 0 || tile.ModelFileReferences.Count > 0)
        {
            ViewerLog.Trace($"[AhdrTerrainAdapter] tile ({tileX},{tileY}): {result.MddfPlacements.Count} M2 + {result.ModfPlacements.Count} WMO placements, " +
                $"{skipped} with an invalid model index, {tile.ModelFileReferences.Count} ADST rows (not placed)");
        }
    }

    private static int GetOrAddName(string name, List<string> names, Dictionary<string, int> index)
    {
        if (!index.TryGetValue(name, out int value))
        {
            value = names.Count;
            names.Add(name);
            index[name] = value;
        }

        return value;
    }

    public bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = [];
        return false;
    }

    public bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    public IReadOnlyList<(int TileX, int TileY)> GetOccupiedTiles(string mapName) =>
        _existingTiles.Select(static key => (key / 64, key % 64)).ToList();

    public bool TryResolveMap(string mapName) => true;

    public bool IsMapWmoBased(string mapName) => false;

    private float[] ScaleHeights(float[] heights)
    {
        if (HeightDivisor != 1f)
        {
            for (int i = 0; i < heights.Length; i++)
                heights[i] /= HeightDivisor;
        }

        return heights;
    }

    private static int TileKey(int tileX, int tileY) => tileX * 64 + tileY;

    private static byte[] ReadHead(string path, int count)
    {
        using FileStream stream = File.OpenRead(path);
        var buffer = new byte[Math.Min(count, stream.Length)];
        stream.ReadExactly(buffer);
        return buffer;
    }
}
