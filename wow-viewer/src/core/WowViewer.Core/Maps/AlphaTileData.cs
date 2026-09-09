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
        float[,]? mcshShadowMask1024 = null,
        float[,,]? mcalAlphaPackFull = null,
        IReadOnlyList<TerrainRawChunkBlob>? rawChunks = null,
        int[,]? areaIds = null,
        int[,,]? mfboFlightBounds = null,
        float[,,]? mccvRgb = null,
        byte[,,]? mclvLightingBytes = null,
        ushort[,]? holeFullMasks = null,
        IReadOnlyList<int>[]? mcrfDoodadRefsByChunk = null,
        IReadOnlyList<int>[]? mcrfWorldModelRefsByChunk = null,
        IReadOnlyList<int>[]? mcrfDoodadUniqueIdsByChunk = null,
        IReadOnlyList<int>[]? mcrfWorldModelUniqueIdsByChunk = null,
        int[,]? mcnkFlags16 = null)
    {
        SourcePath = sourcePath;
        Heightmap = heightmap;
        McalAlphaPack = mcalAlphaPack;
        MclyTextureIds = mclyTextureIds;
        MclyLayerMask = mclyLayerMask;
        HoleMask = holeMask;
        HoleFullMasks = holeFullMasks;
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
        McalAlphaPackFull = mcalAlphaPackFull;
        RawChunks = rawChunks ?? Array.Empty<TerrainRawChunkBlob>();
        AreaIds = areaIds;
        MfboFlightBounds = mfboFlightBounds;
        MccvRgb = mccvRgb;
        MclvLightingBytes = mclvLightingBytes;
        McrfDoodadRefsByChunk = mcrfDoodadRefsByChunk;
        McrfWorldModelRefsByChunk = mcrfWorldModelRefsByChunk;
        McrfDoodadUniqueIdsByChunk = mcrfDoodadUniqueIdsByChunk;
        McrfWorldModelUniqueIdsByChunk = mcrfWorldModelUniqueIdsByChunk;
        McnkFlags16 = mcnkFlags16;
    }

    public string SourcePath { get; }
    public float[,] Heightmap { get; }

    /// <summary>
    /// Downsampled (4x4-block averaged) alpha pack, <c>256x256x4</c>. Kept at this resolution for
    /// the dataset/tensor-pack consumers whose <c>mcal_alpha_pack</c> contract is 256².
    /// </summary>
    public float[,,]? McalAlphaPack { get; }

    /// <summary>
    /// Full-resolution alpha pack, <c>1024x1024x4</c> (64 px per MCNK). <see cref="ToTileLoadResult"/>
    /// slices per-chunk 64x64 alpha maps from this plane; the 256² <see cref="McalAlphaPack"/> cannot
    /// satisfy a 64-px-per-chunk stride and produced silent zero alpha on real tiles (Spec 232 T015d
    /// MCLY regression). Null when the tile carries no MCAL data.
    /// </summary>
    public float[,,]? McalAlphaPackFull { get; }
    public int[,,] MclyTextureIds { get; }
    public bool[,,] MclyLayerMask { get; }
    public bool[,] HoleMask { get; }
    public ushort[,]? HoleFullMasks { get; }
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
    public IReadOnlyList<TerrainRawChunkBlob> RawChunks { get; }
    public int[,]? AreaIds { get; }
    public int[,,]? MfboFlightBounds { get; }
    public float[,,]? MccvRgb { get; }
    public byte[,,]? MclvLightingBytes { get; }
    public IReadOnlyList<int>[]? McrfDoodadRefsByChunk { get; }
    public IReadOnlyList<int>[]? McrfWorldModelRefsByChunk { get; }
    public IReadOnlyList<int>[]? McrfDoodadUniqueIdsByChunk { get; }
    public IReadOnlyList<int>[]? McrfWorldModelUniqueIdsByChunk { get; }

    /// <summary>
    /// Raw MCNK header flags per chunk, [chunkY, chunkX] (row-major, matching the LK path's
    /// <c>ReadMcnkFlags</c> convention). Spec 112 T005: the Alpha reader always parsed these but
    /// previously kept them only for liquid chunks, leaving the dataset signal zero-filled.
    /// </summary>
    public int[,]? McnkFlags16 { get; }

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

    /// <summary>
    /// Spec 232 T015a: returns a full-tile, exact-grid transform of this tile's terrain data.
    /// The 257x257 source lattices move as one unit before <see cref="ToTileLoadResult"/> slices
    /// them into MCNKs. That preserves the shared vertices on MCNK edges, unlike rotating each
    /// already-sliced chunk independently.
    /// </summary>
    /// <remarks>
    /// Positive quarter turns use the same clockwise convention as
    /// <see cref="TileContentTransform.TransformChunkSlot"/>. Mirrors apply after rotation,
    /// matching <see cref="PhaseCompositionPolicy.ComposeTileTransforms"/>. Parsed source data
    /// is never mutated; absent or non-square optional planes stay absent or unchanged.
    /// </remarks>
    public AlphaTileData RotateQuarterTurn(int quarterTurns, bool mirrorH, bool mirrorV)
    {
        int normalizedTurns = ((quarterTurns % 4) + 4) % 4;
        var kinds = new List<TileTransformKind>(3);
        if (normalizedTurns == 1)
            kinds.Add(TileTransformKind.Rotate90CW);
        else if (normalizedTurns == 2)
            kinds.Add(TileTransformKind.Rotate180);
        else if (normalizedTurns == 3)
            kinds.Add(TileTransformKind.Rotate90CCW);

        if (mirrorH)
            kinds.Add(TileTransformKind.MirrorH);
        if (mirrorV)
            kinds.Add(TileTransformKind.MirrorV);

        if (kinds.Count == 0)
            return this;

        return new AlphaTileData(
            SourcePath,
            TransformYX(Heightmap, kinds) ?? Heightmap,
            TransformYX(McalAlphaPack, kinds),
            TransformXY(MclyTextureIds, kinds),
            TransformXY(MclyLayerMask, kinds),
            TransformXY(HoleMask, kinds) ?? HoleMask,
            TextureNames,
            ModelPlacements,
            WorldModelPlacements,
            TransformLiquids(LiquidChunks, kinds),
            diagnostics: Diagnostics,
            mcnrNormalXyz: TransformNormalsYX(McnrNormalXyz, kinds),
            mcshShadowMask256: TransformYX(McshShadowMask256, kinds),
            mclqSurfaceHeight: TransformYX(MclqSurfaceHeight, kinds),
            mclqTypeMask: TransformYX(MclqTypeMask, kinds),
            mcshShadowMask1024: TransformYX(McshShadowMask1024, kinds),
            rawChunks: RawChunks,
            areaIds: TransformXY(AreaIds, kinds),
            mfboFlightBounds: MfboFlightBounds,
            mccvRgb: TransformYX(MccvRgb, kinds),
            mclvLightingBytes: TransformYX(MclvLightingBytes, kinds),
            holeFullMasks: TransformXY(HoleFullMasks, kinds),
            mcrfDoodadRefsByChunk: TransformChunkReferences(McrfDoodadRefsByChunk, kinds),
            mcrfWorldModelRefsByChunk: TransformChunkReferences(McrfWorldModelRefsByChunk, kinds),
            mcrfDoodadUniqueIdsByChunk: TransformChunkReferences(McrfDoodadUniqueIdsByChunk, kinds),
            mcrfWorldModelUniqueIdsByChunk: TransformChunkReferences(McrfWorldModelUniqueIdsByChunk, kinds),
            mcnkFlags16: TransformYX(McnkFlags16, kinds),
            mcalAlphaPackFull: TransformYX(McalAlphaPackFull, kinds));
    }

    public TileLoadResult ToTileLoadResult(int tileX, int tileY)
    {
        const int chunksPerTile = 16;
        const int tileSize = 257;
        const int alphaSize = 64;
        const float mapOrigin = 17066.66666f;
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

                    if (l > 0)
                    {
                        byte[]? alpha = SliceChunkAlphaForChunk(cx, cy, l, alphaSize);
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
                    AreaId = AreaIds != null && cx < AreaIds.GetLength(0) && cy < AreaIds.GetLength(1) ? AreaIds[cx, cy] : 0,
                    McnkFlags = McnkFlags16 != null && cy < McnkFlags16.GetLength(0) && cx < McnkFlags16.GetLength(1)
                        ? McnkFlags16[cy, cx]
                        : liquid != null ? 0x3C : 0
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
        return (int)McnkFlagDecoder.Decode(mcnkFlags);
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

    /// <summary>
    /// Extracts one chunk's 64x64 layer alpha. Prefers the full-resolution 1024x1024 pack
    /// (64 px per chunk, the native MCAL sampling). Falls back to the 256x256 downsampled pack
    /// (16 px per chunk, nearest-neighbor 4x upsample) so tiles built without the full plane
    /// still receive correctly-located alpha instead of silent zeros. Returns null when the
    /// tile carries no alpha data at all.
    /// </summary>
    private byte[]? SliceChunkAlphaForChunk(int cx, int cy, int layer, int alphaSize)
    {
        if (McalAlphaPackFull != null)
            return SliceChunkAlpha(McalAlphaPackFull, cx, cy, layer, alphaSize);

        if (McalAlphaPack != null)
            return UpsampleChunkAlphaFromPacked(McalAlphaPack, cx, cy, layer, alphaSize);

        return null;
    }

    private static byte[] UpsampleChunkAlphaFromPacked(float[,,] alphaPack, int cx, int cy, int layer, int alphaSize)
    {
        int packedEdge = alphaPack.GetLength(0) / 16;
        int sampleStride = alphaSize / packedEdge;
        var alpha = new byte[alphaSize * alphaSize];

        for (int y = 0; y < alphaSize; y++)
        {
            int srcY = cy * packedEdge + (y / sampleStride);
            for (int x = 0; x < alphaSize; x++)
            {
                int srcX = cx * packedEdge + (x / sampleStride);
                float f = alphaPack[srcY, srcX, layer];
                alpha[y * alphaSize + x] = (byte)Math.Clamp((int)(f * 255f), 0, 255);
            }
        }

        return alpha;
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

    // Full-tile arrays use two orientations. Height/normal/alpha/shadow planes are [y, x] because
    // their consumers index a row before a column. The per-MCNK metadata arrays inherited the
    // Alpha parser's [x, y] convention. Both map a source coordinate to the exact destination
    // coordinate named by TileContentTransform, so the two conventions cannot silently invert.
    private static T[,]? TransformYX<T>(T[,]? source, IReadOnlyList<TileTransformKind> kinds)
    {
        if (source == null)
            return null;

        T[,] transformed = source;
        foreach (TileTransformKind kind in kinds)
            transformed = TransformYXOne(transformed, kind);
        return transformed;
    }

    private static T[,] TransformYXOne<T>(T[,] source, TileTransformKind kind)
    {
        int height = source.GetLength(0);
        int width = source.GetLength(1);
        if (height != width)
            return source;

        var result = new T[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                (int tx, int ty) = TransformIndex(x, y, width - 1, kind);
                result[ty, tx] = source[y, x];
            }
        }
        return result;
    }

    private static T[,]? TransformXY<T>(T[,]? source, IReadOnlyList<TileTransformKind> kinds)
    {
        if (source == null)
            return null;

        T[,] transformed = source;
        foreach (TileTransformKind kind in kinds)
            transformed = TransformXYOne(transformed, kind);
        return transformed;
    }

    private static T[,] TransformXYOne<T>(T[,] source, TileTransformKind kind)
    {
        int width = source.GetLength(0);
        int height = source.GetLength(1);
        if (width != height)
            return source;

        var result = new T[width, height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                (int tx, int ty) = TransformIndex(x, y, width - 1, kind);
                result[tx, ty] = source[x, y];
            }
        }
        return result;
    }

    private static T[,,]? TransformYX<T>(T[,,]? source, IReadOnlyList<TileTransformKind> kinds)
    {
        if (source == null)
            return null;

        T[,,] transformed = source;
        foreach (TileTransformKind kind in kinds)
            transformed = TransformYXOne(transformed, kind);
        return transformed;
    }

    private static T[,,] TransformYXOne<T>(T[,,] source, TileTransformKind kind)
    {
        int height = source.GetLength(0);
        int width = source.GetLength(1);
        int depth = source.GetLength(2);
        if (height != width)
            return source;

        var result = new T[height, width, depth];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                (int tx, int ty) = TransformIndex(x, y, width - 1, kind);
                for (int channel = 0; channel < depth; channel++)
                    result[ty, tx, channel] = source[y, x, channel];
            }
        }
        return result;
    }

    private static T[,,] TransformXY<T>(T[,,] source, IReadOnlyList<TileTransformKind> kinds)
    {
        T[,,] transformed = source;
        foreach (TileTransformKind kind in kinds)
        {
            int width = transformed.GetLength(0);
            int height = transformed.GetLength(1);
            int depth = transformed.GetLength(2);
            if (width != height)
                return transformed;

            var result = new T[width, height, depth];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    (int tx, int ty) = TransformIndex(x, y, width - 1, kind);
                    for (int channel = 0; channel < depth; channel++)
                        result[tx, ty, channel] = transformed[x, y, channel];
                }
            }
            transformed = result;
        }
        return transformed;
    }

    private static float[,,]? TransformNormalsYX(float[,,]? source, IReadOnlyList<TileTransformKind> kinds)
    {
        if (source == null)
            return null;

        float[,,] transformed = source;
        foreach (TileTransformKind kind in kinds)
        {
            int height = transformed.GetLength(0);
            int width = transformed.GetLength(1);
            if (height != width || transformed.GetLength(2) < 3)
                return transformed;

            var result = new float[height, width, transformed.GetLength(2)];
            Func<Vector3, Vector3> transformNormal = TileContentTransform.GetNormalTransform(kind);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    (int tx, int ty) = TransformIndex(x, y, width - 1, kind);
                    Vector3 normal = transformNormal(new Vector3(
                        transformed[y, x, 0], transformed[y, x, 1], transformed[y, x, 2]));
                    result[ty, tx, 0] = normal.X;
                    result[ty, tx, 1] = normal.Y;
                    result[ty, tx, 2] = normal.Z;
                    for (int channel = 3; channel < transformed.GetLength(2); channel++)
                        result[ty, tx, channel] = transformed[y, x, channel];
                }
            }
            transformed = result;
        }
        return transformed;
    }

    private static IReadOnlyList<AlphaLiquidChunk> TransformLiquids(
        IReadOnlyList<AlphaLiquidChunk> source,
        IReadOnlyList<TileTransformKind> kinds)
    {
        if (source.Count == 0)
            return source;

        var result = new List<AlphaLiquidChunk>(source.Count);
        foreach (AlphaLiquidChunk liquid in source)
        {
            int chunkX = liquid.IndexX;
            int chunkY = liquid.IndexY;
            byte[]? tileFlags = liquid.TileFlags;
            float[]? heights = liquid.Heights;
            foreach (TileTransformKind kind in kinds)
            {
                (chunkX, chunkY) = TileContentTransform.TransformChunkSlot(chunkX, chunkY, kind);
                tileFlags = TransformSquare(tileFlags, 8, kind);
                heights = TransformSquare(heights, 9, kind);
            }

            result.Add(new AlphaLiquidChunk(
                (chunkY * 16) + chunkX,
                chunkX,
                chunkY,
                liquid.MinHeight,
                liquid.MaxHeight,
                tileFlags,
                liquid.McnkFlags,
                heights));
        }
        return result;
    }

    private static T[]? TransformSquare<T>(T[]? source, int edge, TileTransformKind kind)
    {
        if (source == null || source.Length != edge * edge)
            return source;

        var result = new T[source.Length];
        for (int y = 0; y < edge; y++)
        {
            for (int x = 0; x < edge; x++)
            {
                (int tx, int ty) = TransformIndex(x, y, edge - 1, kind);
                result[(ty * edge) + tx] = source[(y * edge) + x];
            }
        }
        return result;
    }

    private static IReadOnlyList<int>[]? TransformChunkReferences(
        IReadOnlyList<int>[]? source,
        IReadOnlyList<TileTransformKind> kinds)
    {
        if (source == null || source.Length != 256)
            return source;

        var result = new IReadOnlyList<int>[source.Length];
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                int targetX = chunkX;
                int targetY = chunkY;
                foreach (TileTransformKind kind in kinds)
                    (targetX, targetY) = TileContentTransform.TransformChunkSlot(targetX, targetY, kind);
                result[(targetY * 16) + targetX] = source[(chunkY * 16) + chunkX];
            }
        }
        return result;
    }

    private static (int X, int Y) TransformIndex(int x, int y, int max, TileTransformKind kind)
        => kind switch
        {
            TileTransformKind.Rotate90CW => (max - y, x),
            TileTransformKind.Rotate90CCW => (y, max - x),
            TileTransformKind.Rotate180 => (max - x, max - y),
            TileTransformKind.MirrorH => (max - x, y),
            TileTransformKind.MirrorV => (x, max - y),
            _ => (x, y),
        };
}
