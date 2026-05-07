using System.Collections.Concurrent;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public sealed class AlphaTerrainAdapter : ITerrainAdapter
{
    private const int TilesPerAxis = 64;
    private const int ChunksPerTile = 16;
    private const int TileHeightmapSize = 257;
    private const int ChunkAlphaSize = 64;
    private const float MapOrigin = 17066f;
    private const float ChunkSize = 533.33333f;
    private const float ChunkSmall = ChunkSize / ChunksPerTile;

    private readonly byte[] _wdtData;
    private readonly WdtSummary _summary;
    private readonly bool[] _tileExists;
    private readonly string _mapName;
    private readonly List<string> _mdxModelNames;
    private readonly List<string> _wmoModelNames;

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    public IReadOnlyList<string> MdxModelNames => _mdxModelNames;
    public IReadOnlyList<string> WmoModelNames => _wmoModelNames;
    public List<MddfPlacement> MddfPlacements { get; } = [];
    public List<ModfPlacement> ModfPlacements { get; } = [];
    public bool IsWmoBased => _summary.IsWmoBased;
    public List<Vector3> LastLoadedChunkPositions { get; } = [];

    public IReadOnlyList<int> ExistingTiles => _existingTiles;
    private readonly List<int> _existingTiles;

    public AlphaTerrainAdapter(byte[] wdtData, string mapName)
    {
        _wdtData = wdtData;
        _mapName = mapName;

        using var ms = new MemoryStream(wdtData, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, mapName);
        _summary = WdtSummaryReader.Read(ms, fileSummary);

        _tileExists = new bool[TilesPerAxis * TilesPerAxis];
        _existingTiles = [];

        byte[] mainData = ReadMainPayload(wdtData);
        int cellSize = _summary.MainCellSizeBytes;
        if (cellSize < sizeof(uint))
            cellSize = sizeof(uint);

        for (int i = 0; i < TilesPerAxis * TilesPerAxis; i++)
        {
            int offset = i * cellSize;
            if (offset + sizeof(uint) <= mainData.Length)
            {
                uint val = BitConverter.ToUInt32(mainData, offset);
                if (val != 0)
                {
                    _tileExists[i] = true;
                    _existingTiles.Add(i);
                }
            }
        }

        _mdxModelNames = ReadNameTable(wdtData, MapChunkIds.Mdnm, MapChunkIds.Mmdx);
        _wmoModelNames = ReadNameTable(wdtData, MapChunkIds.Monm, MapChunkIds.Mwmo);
    }

    public bool TileExists(int tileX, int tileY)
    {
        if ((uint)tileX >= TilesPerAxis || (uint)tileY >= TilesPerAxis)
            return false;
        return _tileExists[tileX * TilesPerAxis + tileY];
    }

    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        if (!AlphaWdtReader.TryReadTile(_wdtData, tileX, tileY, out AlphaTileData? tileData) || tileData == null)
            return new TileLoadResult { Chunks = [], MddfPlacements = [], ModfPlacements = [] };

        TileTextures[(tileX, tileY)] = tileData.TextureNames.ToList();

        var chunks = new List<TerrainChunkData>(256);

        float tileWorldX = MapOrigin - tileX * ChunkSize;
        float tileWorldY = MapOrigin - tileY * ChunkSize;

        for (int cy = 0; cy < ChunksPerTile; cy++)
        {
            for (int cx = 0; cx < ChunksPerTile; cx++)
            {
                var chunk = BuildChunkData(tileData, tileX, tileY, cx, cy, tileWorldX, tileWorldY);
                if (chunk != null)
                    chunks.Add(chunk);
            }
        }

        var mddfPlacements = tileData.ModelPlacements.Select(p => new MddfPlacement(
            p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale)).ToList();

        var modfPlacements = tileData.WorldModelPlacements.Select(p => new ModfPlacement(
            p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation,
            p.BoundsMin, p.BoundsMax, p.Flags)).ToList();

        lock (_placementLock)
        {
            MddfPlacements.AddRange(mddfPlacements);
            ModfPlacements.AddRange(modfPlacements);
        }

        return new TileLoadResult
        {
            Chunks = chunks,
            MddfPlacements = mddfPlacements,
            ModfPlacements = modfPlacements
        };
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

    private TerrainChunkData? BuildChunkData(
        AlphaTileData tileData, int tileX, int tileY,
        int cx, int cy, float tileWorldX, float tileWorldY)
    {
        float[] heights = ExtractChunkHeights(tileData.Heightmap, cx, cy);
        if (heights == null || heights.All(h => h == 0f))
            return null;

        Vector3[] normals = tileData.McnrNormalXyz != null
            ? ExtractChunkNormals(tileData.McnrNormalXyz, cx, cy)
            : [];
        byte[]? shadowMap = tileData.McshShadowMask1024 != null
            ? ExtractChunkShadow(tileData.McshShadowMask1024, cx, cy)
            : null;

        bool[,] holes = tileData.HoleMask;

        int chunkIdx = cy * ChunksPerTile + cx;

        var layers = new List<TerrainLayer>();
        var alphaMaps = new Dictionary<int, byte[]>();

        for (int l = 0; l < 4; l++)
        {
            if (!tileData.MclyLayerMask[cx, cy, l])
                break;

            layers.Add(new TerrainLayer
            {
                TextureIndex = tileData.MclyTextureIds[cx, cy, l],
                Flags = 0,
                AlphaOffset = 0,
                EffectId = 0
            });

            if (l > 0 && tileData.McalAlphaPack != null)
            {
                var alpha = ExtractChunkAlpha(tileData.McalAlphaPack, cx, cy, l);
                if (alpha != null)
                    alphaMaps[l] = alpha;
            }
        }

        var liquid = FindLiquidChunk(tileData, cx, cy);

        float chunkWorldX = tileWorldX - cy * ChunkSmall;
        float chunkWorldY = tileWorldY - cx * ChunkSmall;

        LastLoadedChunkPositions.Add(new Vector3(chunkWorldX, chunkWorldY, 0f));

        return new TerrainChunkData
        {
            TileX = tileX,
            TileY = tileY,
            ChunkX = cx,
            ChunkY = cy,
            Heights = heights,
            Normals = normals,
            ShadowMap = shadowMap,
            HoleMask = holes[cx, cy] ? 1 : 0,
            Layers = layers.ToArray(),
            AlphaMaps = alphaMaps,
            Liquid = liquid,
            WorldPosition = new Vector3(chunkWorldX, chunkWorldY, 0f),
            AreaId = 0,
            McnkFlags = liquid != null ? 0x3C : 0
        };
    }

    private static float[] ExtractChunkHeights(float[,] heightmap, int cx, int cy)
    {
        var heights = new float[145];
        int baseX = cy * 16;
        int baseY = cx * 16;
        int idx = 0;

        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            for (int col = 0; col < cols; col++)
            {
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;

                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                    heights[idx] = heightmap[py, px];

                idx++;
            }
        }

        return heights;
    }

    private static byte[] ExtractChunkAlpha(float[,,] alphaPack, int cx, int cy, int layer)
    {
        const int size = ChunkAlphaSize;
        var alpha = new byte[size * size];

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int srcY = cx * size + y;
                int srcX = cy * size + x;
                if (srcY < alphaPack.GetLength(0) && srcX < alphaPack.GetLength(1))
                {
                    float f = alphaPack[srcY, srcX, layer];
                    alpha[y * size + x] = (byte)Math.Clamp((int)(f * 255f), 0, 255);
                }
            }
        }

        return alpha;
    }

    private static LiquidChunkData? FindLiquidChunk(AlphaTileData tileData, int cx, int cy)
    {
        if (tileData.LiquidChunks == null)
            return null;

        foreach (var liquid in tileData.LiquidChunks)
        {
            if (liquid.IndexX == cx && liquid.IndexY == cy)
            {
                return new LiquidChunkData
                {
                    LiquidType = ClassifyLiquidType(liquid.McnkFlags),
                    MinHeight = liquid.MinHeight,
                    MaxHeight = liquid.MaxHeight,
                    TileFlags = liquid.TileFlags
                };
            }
        }

        return null;
    }

    private static int ClassifyLiquidType(uint mcnkFlags)
    {
        if ((mcnkFlags & 0x08) != 0) return 1;
        int bits = (int)((mcnkFlags >> 4) & 3);
        return bits switch
        {
            1 => 1,
            2 => 2,
            3 => 3,
            _ => 0
        };
    }

    private static byte[] ReadMainPayload(byte[] wdtData)
    {
        using var ms = new MemoryStream(wdtData, writable: false);
        var fileSummary = MapFileSummaryReader.Read(ms, "<alpha-main>");

        foreach (var chunk in fileSummary.Chunks)
        {
            if (chunk.Id == MapChunkIds.Main)
            {
                ms.Position = chunk.DataOffset;
                byte[] payload = new byte[chunk.Size];
                ms.ReadExactly(payload, 0, payload.Length);
                return payload;
            }
        }

        return [];
    }

    private static List<string> ReadNameTable(byte[] wdtData, FourCC primaryChunkId, FourCC fallbackChunkId)
    {
        using var ms = new MemoryStream(wdtData, writable: false);
        var fileSummary = MapFileSummaryReader.Read(ms, "<alpha-names>");

        foreach (var id in new[] { primaryChunkId, fallbackChunkId })
        {
            foreach (var chunk in fileSummary.Chunks)
            {
                if (chunk.Id == id)
                {
                    ms.Position = chunk.DataOffset;
                    byte[] payload = new byte[chunk.Size];
                    ms.ReadExactly(payload, 0, payload.Length);
                    return ReadStringEntries(payload);
                }
            }
        }

        return [];
    }

    private static Vector3[] ExtractChunkNormals(float[,,] normalXyz, int cx, int cy)
    {
        const int tileSize = 257;
        var normals = new Vector3[145];
        int baseX = cy * 16;
        int baseY = cx * 16;
        int idx = 0;

        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            for (int col = 0; col < cols; col++)
            {
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;

                if ((uint)px < tileSize && (uint)py < tileSize)
                    normals[idx] = new Vector3(normalXyz[py, px, 0], normalXyz[py, px, 1], normalXyz[py, px, 2]);
                idx++;
            }
        }

        return normals;
    }

    private static byte[]? ExtractChunkShadow(float[,] shadowMask1024, int cx, int cy)
    {
        const int srcSize = 1024;
        const int chunkSize = 64;
        var shadow = new byte[chunkSize * chunkSize];
        int baseX = cy * chunkSize;
        int baseY = cx * chunkSize;

        for (int y = 0; y < chunkSize; y++)
        {
            for (int x = 0; x < chunkSize; x++)
            {
                int sy = baseY + y;
                int sx = baseX + x;
                if (sy < srcSize && sx < srcSize)
                    shadow[y * chunkSize + x] = (byte)(shadowMask1024[sy, sx] * 255f);
            }
        }

        return shadow;
    }

    private static List<string> ReadStringEntries(byte[]? payload)
    {
        if (payload == null || payload.Length == 0) return [];
        var entries = new List<string>();
        int start = 0;
        for (int i = 0; i < payload.Length; i++)
        {
            if (payload[i] != 0) continue;
            if (i > start) entries.Add(System.Text.Encoding.UTF8.GetString(payload, start, i - start));
            start = i + 1;
        }
        if (start < payload.Length)
            entries.Add(System.Text.Encoding.UTF8.GetString(payload, start, payload.Length - start));
        return entries;
    }

    private readonly object _placementLock = new();
}