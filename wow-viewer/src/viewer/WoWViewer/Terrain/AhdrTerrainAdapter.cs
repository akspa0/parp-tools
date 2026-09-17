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
/// Not rendered yet: object placements (ACDO frame unverified), vertex shading (ACVT channel order
/// unverified), shadows (ASHD bit layout unverified).
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
            if (!AdtAhdrReader.IsAhdrFamily(head) || !AdtAhdrReader.TryReadTileLocation(head, out int alocX, out int alocY))
                continue;

            if (alocX is < 0 or >= 64 || alocY is < 0 or >= 64)
            {
                SkippedFiles.Add($"{Path.GetFileName(path)}: ALOC tile ({alocX}, {alocY}) outside the 64x64 grid");
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

    /// <summary>
    /// Display divisor applied to DAT v26 heights. DAT heights appear to be in inches: the corpus's
    /// 5th-percentile height −18559.47 ÷ 36 = −515.54, matching the shipped Azeroth ADT ocean floor
    /// (−515.19..−516.07 yd). The horizontal scale is not yet established.
    /// </summary>
    public float HeightDivisor { get; }

    /// <summary>Files that were AHDR-family but not placed, with the reason.</summary>
    public List<string> SkippedFiles { get; } = [];

    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    public IReadOnlyList<string> MdxModelNames => [];

    public IReadOnlyList<string> WmoModelNames => [];

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
                if (layerIndex > 0 && layer.AlphaMap is { Length: 64 * 64 })
                    alphaMaps[layerIndex] = layer.AlphaMap;
            }

            chunks.Add(new TerrainChunkData
            {
                McinIndex = gridRow * 16 + gridColumn,
                TileX = tileX,
                TileY = tileY,
                ChunkX = gridColumn,
                ChunkY = gridRow,
                Heights = ScaleHeights(AdtAhdrTileSlicer.SliceHeights(tile, gridColumn, gridRow)),
                // Dividing heights by D flattens slopes by D; equivalently widen the spacing by D.
                Normals = AdtAhdrTileSlicer.ComputeNormals(tile, gridColumn, gridRow, vertexSpacing * HeightDivisor),
                Layers = layers,
                AlphaMaps = alphaMaps,
                WorldPosition = new Vector3(worldX, worldY, 0f),
            });
        }

        return new TileLoadResult { Chunks = chunks };
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
