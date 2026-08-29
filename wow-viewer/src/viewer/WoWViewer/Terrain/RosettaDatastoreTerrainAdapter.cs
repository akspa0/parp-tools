using System.Collections.Concurrent;
using System.Numerics;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Terrain adapter that directly streams and renders multi-version Rosetta maps
/// from the unified Zarr v3 Datastore without requiring loose ADT/WDT files on disk.
/// </summary>
public sealed class RosettaDatastoreTerrainAdapter : ITerrainAdapter
{
    private readonly RosettaObjectLibrary _library;
    private readonly string _buildId;
    private readonly string _mapName;
    private readonly IDataSource? _dataSource;
    private readonly List<int> _existingTiles;
    private readonly HashSet<int> _existingTileSet;
    private readonly Dictionary<(int X, int Y), List<RosettaPlacementRecord>> _placementsByTile = new();
    private readonly Dictionary<(int X, int Y), int> _tileOrderIndex = new();
    private readonly List<string> _mdxNames = new();
    private readonly List<string> _wmoNames = new();
    private readonly Dictionary<string, int> _mdxNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly string _groundTexture;
    private readonly string _checkersTexture;
    private readonly string _inkTexture;

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();
    public IReadOnlyList<string> MdxModelNames => _mdxNames;
    public IReadOnlyList<string> WmoModelNames => _wmoNames;
    public List<MddfPlacement> MddfPlacements { get; } = new();
    public List<ModfPlacement> ModfPlacements { get; } = new();
    public bool IsWmoBased => false;
    public List<Vector3> LastLoadedChunkPositions { get; } = new();
    public IReadOnlyList<int> ExistingTiles => _existingTiles;
    public string? OverlayMapName { get; set; }

    public RosettaDatastoreTerrainAdapter(
        RosettaObjectLibrary library,
        string buildId,
        string mapName,
        IDataSource? dataSource,
        string groundTexture = "tileset\\desert\\desertdirt01.blp",
        string checkersTexture = "tileset\\generic\\checkers.blp",
        string inkTexture = "tileset\\generic\\black.blp")
    {
        _library = library ?? throw new ArgumentNullException(nameof(library));
        _buildId = buildId;
        _mapName = mapName;
        _dataSource = dataSource;
        _groundTexture = groundTexture;
        _checkersTexture = checkersTexture;
        _inkTexture = inkTexture;

        var allPlacements = _library.GetPlacements(buildId, mapName);
        var tileCoords = new List<(int X, int Y)>();

        foreach (var p in allPlacements)
        {
            var key = (p.TileX, p.TileY);
            if (!_placementsByTile.TryGetValue(key, out var list))
            {
                list = new List<RosettaPlacementRecord>();
                _placementsByTile[key] = list;
                tileCoords.Add(key);
            }
            list.Add(p);

            // Index models and WMOs
            if (p.Asset.Kind == RosettaAssetKind.WorldModel)
            {
                if (!_wmoNameIndex.ContainsKey(p.Asset.AssetPath))
                {
                    _wmoNameIndex[p.Asset.AssetPath] = _wmoNames.Count;
                    _wmoNames.Add(p.Asset.AssetPath);
                }
            }
            else
            {
                if (!_mdxNameIndex.ContainsKey(p.Asset.AssetPath))
                {
                    _mdxNameIndex[p.Asset.AssetPath] = _mdxNames.Count;
                    _mdxNames.Add(p.Asset.AssetPath);
                }
            }
        }

        // Deterministic tile order
        tileCoords.Sort((a, b) => a.Y != b.Y ? a.Y.CompareTo(b.Y) : a.X.CompareTo(b.X));
        _existingTiles = new List<int>(tileCoords.Count);
        for (int i = 0; i < tileCoords.Count; i++)
        {
            var (tx, ty) = tileCoords[i];
            _existingTiles.Add(tx * 64 + ty);
            _tileOrderIndex[(tx, ty)] = i;
        }

        _existingTileSet = new HashSet<int>(_existingTiles);
    }

    public bool TileExists(int tileX, int tileY) => _existingTileSet.Contains(tileX * 64 + tileY);

    public bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();
        return false;
    }

    public bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        int tileIdx = tileX * 64 + tileY;
        if (!_existingTileSet.Contains(tileIdx))
            return new TileLoadResult();

        var key = (tileX, tileY);
        int orderIdx = _tileOrderIndex.TryGetValue(key, out int oi) ? oi : 0;
        var placements = _placementsByTile.TryGetValue(key, out var plist) ? plist : new List<RosettaPlacementRecord>();

        // Set tile textures: Layer 0 Ground, Layer 1 Checkers, Layer 2 Ink
        var textureList = new List<string> { _groundTexture, _checkersTexture, _inkTexture };
        TileTextures.TryAdd((tileX, tileY), textureList);

        var tileMddfs = new List<MddfPlacement>();
        var tileModfs = new List<ModfPlacement>();

        foreach (var p in placements)
        {
            if (p.Asset.Kind == RosettaAssetKind.WorldModel)
            {
                int nameIdx = _wmoNameIndex.TryGetValue(p.Asset.AssetPath, out int wi) ? wi : 0;
                var modf = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = p.UniqueId,
                    Position = p.RendererPosition,
                    Rotation = Vector3.Zero,
                    BoundsMin = p.Asset.BoundsMin,
                    BoundsMax = p.Asset.BoundsMax,
                    Flags = 0
                };
                tileModfs.Add(modf);
                ModfPlacements.Add(modf);
            }
            else
            {
                int nameIdx = _mdxNameIndex.TryGetValue(p.Asset.AssetPath, out int mi) ? mi : 0;
                var mddf = new MddfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = p.UniqueId,
                    Position = p.RendererPosition,
                    Rotation = Vector3.Zero,
                    Scale = 1024
                };
                tileMddfs.Add(mddf);
                MddfPlacements.Add(mddf);
            }
        }

        // Get or build alpha and checkers canvases
        byte[]? alphaCanvas = _library.GetTileAlpha(_buildId, _mapName, orderIdx);
        byte[]? checkersCanvas = _library.GetTileCheckers(_buildId, _mapName, orderIdx);

        // Build pedestals for heightfield calculation
        var pedestals = new List<RosettaPedestal>();
        foreach (var p in placements)
        {
            pedestals.Add(new RosettaPedestal(
                p.CellU,
                p.CellV,
                p.CellU + p.CellSize,
                p.CellV + p.ObjectBandSize,
                -10f)); // 10m museum sunken plateau
        }

        if (alphaCanvas == null)
        {
            var dummyPlan = new RosettaTilePlan(tileX, tileY, placements, Array.Empty<RosettaMccvRect>(), Array.Empty<RosettaLabel>(), null, pedestals, null);
            alphaCanvas = RosettaTilesetGenerator.BuildTileAlphaCanvas(dummyPlan, paintCellBorders: true);
        }

        if (checkersCanvas == null)
        {
            var dummyPlan = new RosettaTilePlan(tileX, tileY, placements, Array.Empty<RosettaMccvRect>(), Array.Empty<RosettaLabel>(), null, pedestals, null);
            checkersCanvas = RosettaTilesetGenerator.BuildTileCheckersCanvas(dummyPlan);
        }

        var chunks = new List<TerrainChunkData>(256);
        float chunkSmall = WoWConstants.ChunkSize / 16f;

        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int cx = chunkIndex % 16;
            int cy = chunkIndex / 16;

            float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - cy * chunkSmall;
            float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - cx * chunkSmall;

            LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));

            // Chunk heights from pedestals
            float[] heights = CreateChunkHeights(cx, cy, pedestals);
            var normals = new Vector3[145];
            Array.Fill(normals, Vector3.UnitZ);

            // Slice 64x64 alpha maps for Layer 1 (checkers) and Layer 2 (ink text)
            byte[] chunkCheckersAlpha = SliceCanvas64x64(checkersCanvas, cx, cy);
            byte[] chunkInkAlpha = SliceCanvas64x64(alphaCanvas, cx, cy);

            var layers = new TerrainLayer[]
            {
                new() { TextureIndex = 0, Flags = 0 },
                new() { TextureIndex = 1, Flags = 0x200 },
                new() { TextureIndex = 2, Flags = 0x200 }
            };

            var alphaMaps = new Dictionary<int, byte[]>
            {
                [1] = chunkCheckersAlpha,
                [2] = chunkInkAlpha
            };

            var chunkData = new TerrainChunkData
            {
                TileX = tileX,
                TileY = tileY,
                ChunkX = cx,
                ChunkY = cy,
                Heights = heights,
                Normals = normals,
                HoleMask = 0,
                Layers = layers,
                AlphaMaps = alphaMaps,
                ShadowMap = null,
                Liquid = null,
                WorldPosition = new Vector3(worldX, worldY, 0f),
                AreaId = 0,
                McnkFlags = 0
            };

            chunks.Add(chunkData);
        }

        return new TileLoadResult
        {
            Chunks = chunks,
            MddfPlacements = tileMddfs,
            ModfPlacements = tileModfs
        };
    }

    private static byte[] SliceCanvas64x64(byte[] canvas1024, int chunkX, int chunkY)
    {
        var slice = new byte[64 * 64];
        if (canvas1024 == null || canvas1024.Length < 1024 * 1024)
            return slice;

        int startX = chunkX * 64;
        int startY = chunkY * 64;

        for (int y = 0; y < 64; y++)
        {
            int srcOffset = ((startY + y) * 1024) + startX;
            int dstOffset = y * 64;
            Buffer.BlockCopy(canvas1024, srcOffset, slice, dstOffset, 64);
        }

        return slice;
    }

    private static float[] CreateChunkHeights(int cx, int cy, IReadOnlyList<RosettaPedestal> pedestals)
    {
        float chunkSize = WoWConstants.ChunkSize / 16f; // 33.33333m
        var heights = new float[145];
        if (pedestals == null || pedestals.Count == 0)
            return heights;

        const float bevelMeters = 12.5f;
        int idx = 0;
        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            int r = row / 2;
            for (int col = 0; col < cols; col++)
            {
                float u = (cx * chunkSize) + ((isInner ? col + 0.5f : col) * (chunkSize / 8f));
                float v = (cy * chunkSize) + ((isInner ? r + 0.5f : r) * (chunkSize / 8f));

                float h = 0f;
                for (int pIdx = 0; pIdx < pedestals.Count; pIdx++)
                {
                    RosettaPedestal p = pedestals[pIdx];
                    if (u < p.U0 || u > p.U1 || v < p.V0 || v > p.V1)
                        continue;

                    float distLeft = u - p.U0;
                    float distRight = p.U1 - u;
                    float distTop = v - p.V0;
                    float distBottom = p.V1 - v;
                    float edgeDist = MathF.Min(MathF.Min(distLeft, distRight), MathF.Min(distTop, distBottom));

                    float pedH = (bevelMeters > 0f && edgeDist < bevelMeters)
                        ? p.Height * (edgeDist / bevelMeters)
                        : p.Height;

                    if (p.Height > 0f && pedH > h)
                        h = pedH;
                    else if (p.Height < 0f && pedH < h)
                        h = pedH;
                }

                heights[idx++] = h;
            }
        }

        return heights;
    }
}
