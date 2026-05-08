using System.Numerics;

namespace WowViewer.Core.Maps;

public sealed record AlphaModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public sealed record AlphaWorldModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags);

public sealed record AlphaLiquidChunk(
    int ChunkIndex,
    int IndexX,
    int IndexY,
    float MinHeight,
    float MaxHeight,
    byte[]? TileFlags,
    uint McnkFlags,
    float[]? Heights);

public sealed record AlphaTileDiagnostics(
    bool HasResidualData,
    bool HasSparseChunks,
    int ResidualDataBytes,
    int ActiveChunkCount,
    bool McshSunOrientationUpperRight,
    int McshDataSize);

public sealed class AlphaTileData
{
    public AlphaTileData(
        string sourcePath,
        float[,] heightmap,
        float[,,]? mcalAlphaPack,
        int[,,] mclyTextureIds,
        bool[,,] mclyLayerMask,
        bool[,] holeMask,
        IReadOnlyList<string> textureNames,
        IReadOnlyList<AlphaModelPlacement> modelPlacements,
        IReadOnlyList<AlphaWorldModelPlacement> worldModelPlacements,
        IReadOnlyList<AlphaLiquidChunk> liquidChunks,
        AlphaTileDiagnostics? diagnostics = null,
        float[,,]? mcnrNormalXyz = null,
        float[,]? mcshShadowMask256 = null,
        float[,]? mclqSurfaceHeight = null,
        int[,]? mclqTypeMask = null,
        float[,]? mcshShadowMask1024 = null)
    {
        SourcePath = sourcePath;
        Heightmap = heightmap;
        McalAlphaPack = mcalAlphaPack;
        MclyTextureIds = mclyTextureIds;
        MclyLayerMask = mclyLayerMask;
        HoleMask = holeMask;
        TextureNames = textureNames;
        ModelPlacements = modelPlacements;
        WorldModelPlacements = worldModelPlacements;
        LiquidChunks = liquidChunks;
        Diagnostics = diagnostics;
        McnrNormalXyz = mcnrNormalXyz;
        McshShadowMask256 = mcshShadowMask256;
        MclqSurfaceHeight = mclqSurfaceHeight;
        MclqTypeMask = mclqTypeMask;
        McshShadowMask1024 = mcshShadowMask1024;
    }

    public string SourcePath { get; }
    public float[,] Heightmap { get; }
    public float[,,]? McalAlphaPack { get; }
    public int[,,] MclyTextureIds { get; }
    public bool[,,] MclyLayerMask { get; }
    public bool[,] HoleMask { get; }
    public IReadOnlyList<string> TextureNames { get; }
    public IReadOnlyList<AlphaModelPlacement> ModelPlacements { get; }
    public IReadOnlyList<AlphaWorldModelPlacement> WorldModelPlacements { get; }
    public IReadOnlyList<AlphaLiquidChunk> LiquidChunks { get; }
    public AlphaTileDiagnostics? Diagnostics { get; }
    public float[,,]? McnrNormalXyz { get; }
    public float[,]? McshShadowMask256 { get; }
    public float[,]? MclqSurfaceHeight { get; }
    public int[,]? MclqTypeMask { get; }
    public float[,]? McshShadowMask1024 { get; }

    public AdtPlacementCatalog ToPlacementCatalog()
    {
        return new AdtPlacementCatalog(
            SourcePath,
            MapFileKind.Adt,
            ModelPlacements.Select(static p => p.ModelPath).Distinct().OrderBy(static p => p).ToList(),
            WorldModelPlacements.Select(static p => p.ModelPath).Distinct().OrderBy(static p => p).ToList(),
            ModelPlacements.Select(static p => new AdtModelPlacement(
                p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale)).ToList(),
            WorldModelPlacements.Select(static p => new AdtWorldModelPlacement(
                p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation,
                p.BoundsMin, p.BoundsMax, p.Flags)).ToList());
    }

    public TileLoadResult ToTileLoadResult(int tileX, int tileY)
    {
        const int chunksPerTile = 16;
        const int tileSize = 257;
        const int alphaSize = 64;
        const float mapOrigin = 17066f;
        const float chunkSize = 533.33333f;
        const float chunkSmall = chunkSize / chunksPerTile;

        float tileWorldX = mapOrigin - tileX * chunkSize;
        float tileWorldY = mapOrigin - tileY * chunkSize;

        var chunks = new List<TerrainChunkData>(256);

        for (int cy = 0; cy < chunksPerTile; cy++)
        {
            for (int cx = 0; cx < chunksPerTile; cx++)
            {
var heights = SliceChunkHeights(Heightmap, cx, cy, tileSize);
                var normals = McnrNormalXyz != null ? SliceChunkNormals(McnrNormalXyz, cx, cy) : [];
                byte[] shadow = McshShadowMask1024 != null ? SliceChunkShadow1024(McshShadowMask1024, cx, cy) : [];
                var liquid = FindLiquid(cx, cy);

                var layers = new List<TerrainLayer>();
                var alphaMaps = new Dictionary<int, byte[]>();

                for (int l = 0; l < 4; l++)
                {
                    if (!MclyLayerMask[cx, cy, l])
                        break;

                    layers.Add(new TerrainLayer
                    {
                        TextureIndex = MclyTextureIds[cx, cy, l],
                        Flags = 0,
                        AlphaOffset = 0,
                        EffectId = 0
                    });

                    if (l > 0 && McalAlphaPack != null)
                    {
                        var alpha = SliceChunkAlpha(McalAlphaPack, cx, cy, l, alphaSize);
                        if (alpha != null)
                            alphaMaps[l] = alpha;
                    }
                }

                float chunkWorldX = tileWorldX - cy * chunkSmall;
                float chunkWorldY = tileWorldY - cx * chunkSmall;

                chunks.Add(new TerrainChunkData
                {
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = cx,
                    ChunkY = cy,
                    Heights = heights,
                    Normals = normals,
                    ShadowMap = shadow,
                    HoleMask = (cx < HoleMask.GetLength(0) && cy < HoleMask.GetLength(1) && HoleMask[cx, cy]) ? 1 : 0,
                    Layers = layers.ToArray(),
                    AlphaMaps = alphaMaps,
                    Liquid = liquid,
                    WorldPosition = new Vector3(chunkWorldX, chunkWorldY, 0f),
                    AreaId = 0,
                    McnkFlags = liquid != null ? 0x3C : 0
                });
            }
        }

        var mddfPlacements = ModelPlacements.Select(p => new MddfPlacement(
            p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale)).ToList();

        var modfPlacements = WorldModelPlacements.Select(p => new ModfPlacement(
            p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation,
            p.BoundsMin, p.BoundsMax, p.Flags)).ToList();

        return new TileLoadResult
        {
            Chunks = chunks,
            MddfPlacements = mddfPlacements,
            ModfPlacements = modfPlacements
        };
    }

    private LiquidChunkData? FindLiquid(int cx, int cy)
    {
        foreach (var lc in LiquidChunks)
        {
            if (lc.IndexX == cx && lc.IndexY == cy)
            {
                return new LiquidChunkData
                {
                    LiquidType = ClassifyLiquid(lc.McnkFlags),
                    MinHeight = lc.MinHeight,
                    MaxHeight = lc.MaxHeight,
                    TileFlags = lc.TileFlags
                };
            }
        }
        return null;
    }

    private static int ClassifyLiquid(uint mcnkFlags)
    {
        if ((mcnkFlags & 0x04) != 0) return 1;
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

    private static float[] SliceChunkHeights(float[,] heightmap, int cx, int cy, int tileSize)
    {
        var heights = new float[145];
        int baseX = cx * 16;
        int baseY = cy * 16;
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
                    heights[idx] = heightmap[py, px];

                idx++;
            }
        }

        return heights;
    }

    private static byte[] SliceChunkAlpha(float[,,] alphaPack, int cx, int cy, int layer, int alphaSize)
    {
        var alpha = new byte[alphaSize * alphaSize];

        for (int y = 0; y < alphaSize; y++)
        {
            for (int x = 0; x < alphaSize; x++)
            {
                int srcY = cy * alphaSize + y;
                int srcX = cx * alphaSize + x;
                if (srcY < alphaPack.GetLength(0) && srcX < alphaPack.GetLength(1))
                {
                    float f = alphaPack[srcY, srcX, layer];
                    alpha[y * alphaSize + x] = (byte)Math.Clamp((int)(f * 255f), 0, 255);
                }
            }
        }

        return alpha;
    }

    private static Vector3[] SliceChunkNormals(float[,,] normalXyz, int cx, int cy)
    {
        var normals = new Vector3[145];
        int baseX = cx * 16;
        int baseY = cy * 16;
        const int tileSize = 257;
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
                {
                    normals[idx] = new Vector3(
                        normalXyz[py, px, 0],
                        normalXyz[py, px, 1],
                        normalXyz[py, px, 2]);
                }
                idx++;
            }
        }

        return normals;
    }

    private static byte[] SliceChunkShadow1024(float[,] shadowMask, int cx, int cy)
    {
        const int srcSize = 1024;
        const int chunkSize = 64;
        var shadow = new byte[chunkSize * chunkSize];
        int baseX = cx * chunkSize;
        int baseY = cy * chunkSize;

        for (int y = 0; y < chunkSize; y++)
        {
            for (int x = 0; x < chunkSize; x++)
            {
                int sy = baseY + y;
                int sx = baseX + x;
                if (sy < srcSize && sx < srcSize)
                    shadow[y * chunkSize + x] = (byte)(shadowMask[sy, sx] * 255f);
            }
        }

        return shadow;
    }
}
