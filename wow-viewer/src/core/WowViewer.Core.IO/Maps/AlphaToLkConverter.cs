using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public sealed class AlphaToLkConversionResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public string? MapName { get; init; }
    public string? OutputDirectory { get; init; }
    public int TilesConverted { get; init; }
    public int TotalTiles { get; init; }
    public long ElapsedMs { get; init; }
    public List<string> Warnings { get; init; } = [];
}

public sealed class AlphaToLkOptions
{
    public string? AreaCrosswalkPath { get; init; }
    public bool Verbose { get; init; }
}

public static class AlphaToLkConverter
{
    private const float MapOrigin = 17066.666f;
    private const float ChunkSize = 533.33333f;
    private const float ChunkSubSize = ChunkSize / 16f;
    private const int TileHeightmapSize = 257;
    private const int ChunksPerTile = 16;

    public static AlphaToLkConversionResult ConvertWdt(
        string wdtPath,
        string outputDir,
        AlphaToLkOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(wdtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDir);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var warnings = new List<string>();

        try
        {
            if (!File.Exists(wdtPath))
                return new AlphaToLkConversionResult { Success = false, Error = $"WDT not found: {wdtPath}" };

            Directory.CreateDirectory(outputDir);
            string mapName = Path.GetFileNameWithoutExtension(wdtPath);

            byte[] wdtData = File.ReadAllBytes(wdtPath);
            if (!AlphaWdtReader.IsAlphaWdt(wdtData))
                return new AlphaToLkConversionResult { Success = false, Error = "Not an Alpha WDT file." };

            var existingTiles = AlphaWdtReader.ReadExistingTiles(wdtData);
            if (existingTiles.Count == 0)
                return new AlphaToLkConversionResult { Success = false, Error = "No existing tiles found in WDT." };

            byte[] wdtLk = LkWdtWriter.Build(existingTiles);
            File.WriteAllBytes(Path.Combine(outputDir, $"{mapName}.wdt"), wdtLk);

            var wdlTiles = new List<WdlHeightTile>();
            int converted = 0;

            foreach (var (tileX, tileY) in existingTiles)
            {
                if (!AlphaWdtReader.TryReadTile(wdtData, tileX, tileY, out AlphaTileData? tileData) || tileData == null)
                {
                    warnings.Add($"Tile ({tileX},{tileY}): failed to read");
                    continue;
                }

                LkAdtData adtData = ConvertTile(tileData, tileX, tileY);
                byte[] adtBytes = LkAdtWriter.Build(adtData);
                File.WriteAllBytes(Path.Combine(outputDir, $"{mapName}_{tileX}_{tileY}.adt"), adtBytes);

                wdlTiles.Add(WdlWriter.ExtractTileHeightsFromAlpha(tileData.Heightmap, tileX, tileY));
                converted++;
            }

            byte[] wdlBytes = WdlWriter.Build(wdlTiles);
            File.WriteAllBytes(Path.Combine(outputDir, $"{mapName}.wdl"), wdlBytes);

            sw.Stop();
            return new AlphaToLkConversionResult
            {
                Success = true,
                MapName = mapName,
                OutputDirectory = outputDir,
                TilesConverted = converted,
                TotalTiles = existingTiles.Count,
                ElapsedMs = sw.ElapsedMilliseconds,
                Warnings = warnings
            };
        }
        catch (Exception ex)
        {
            return new AlphaToLkConversionResult
            {
                Success = false,
                Error = ex.Message,
                ElapsedMs = sw.ElapsedMilliseconds
            };
        }
    }

    public static LkAdtData ConvertTile(AlphaTileData tile, int tileX, int tileY)
    {
        ArgumentNullException.ThrowIfNull(tile);

        var modelPlacements = tile.ModelPlacements.Select(p => new LkMddfEntry(
            p.NameId,
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.Scale)).ToList();

        var worldModelPlacements = tile.WorldModelPlacements.Select(p => new LkModfEntry(
            p.NameId,
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.BoundsMin,
            p.BoundsMax,
            p.Flags,
            DoodadSet: 0,
            NameSet: 0,
            Scale: 1.0f)).ToList();

        var textureNames = tile.TextureNames.ToList();

        var chunks = new List<LkMcnkData>(256);
        for (int cy = 0; cy < ChunksPerTile; cy++)
        {
            for (int cx = 0; cx < ChunksPerTile; cx++)
            {
                chunks.Add(BuildChunkData(tile, cx, cy, tileX, tileY, textureNames, modelPlacements, worldModelPlacements));
            }
        }

        return new LkAdtData
        {
            MapName = "",
            TileX = tileX,
            TileY = tileY,
            TextureNames = textureNames,
            ModelNames = tile.ModelPlacements.Select(p => p.ModelPath).Distinct().ToList(),
            WorldModelNames = tile.WorldModelPlacements.Select(p => p.ModelPath).Distinct().ToList(),
            ModelPlacements = RemapPlacements(tile.ModelPlacements, tile.ModelPlacements.Select(p => p.ModelPath).Distinct().ToList()),
            WorldModelPlacements = RemapWorldPlacements(tile.WorldModelPlacements, tile.WorldModelPlacements.Select(p => p.ModelPath).Distinct().ToList()),
            Chunks = chunks,
            MhdrFlags = ComputeMhdrFlags(tile)
        };
    }

    private static LkMcnkData BuildChunkData(
        AlphaTileData tile, int cx, int cy, int tileX, int tileY,
        List<string> textureNames,
        List<LkMddfEntry> modelPlacements,
        List<LkModfEntry> worldModelPlacements)
    {
        const int ChunkAltSize = 64;

        bool hasData = false;
        float baseHeight = 0f;
        float[] heights = ExtractChunkHeights(tile.Heightmap, cx, cy);
        foreach (float h in heights) { if (h != 0f) { hasData = true; } }
        if (!hasData) return CreateEmptyChunk(cx, cy, tileX, tileY);

        baseHeight = ComputeBaseHeight(heights);

        for (int i = 0; i < heights.Length; i++)
            heights[i] -= baseHeight;

        byte[] normals = EncodeChunkNormals(tile.McnrNormalXyz, cx, cy);

        byte[]? shadowMap = tile.McshShadowMask1024 != null
            ? SliceChunkShadow1024(tile.McshShadowMask1024, cx, cy)
            : null;

        uint mcnkFlags = 0;
        int liquid = FindLiquidType(tile, cx, cy);
        if (liquid > 0)
            mcnkFlags |= (liquid == 1) ? 0x08u : (liquid == 2) ? 0x10u : 0x18u;

        if (shadowMap != null)
            mcnkFlags |= 0x01u;

        int areaId = 0;

        int nLayers = 0;
        for (int l = 0; l < 4; l++)
        {
            if (cx < tile.MclyLayerMask.GetLength(0) && cy < tile.MclyLayerMask.GetLength(1) && tile.MclyLayerMask[cx, cy, l])
                nLayers = l + 1;
        }

        var layers = new List<LkMclyEntry>();
        byte[]? alphaMapData = null;
        int alphaOffset = 0;

        for (int l = 0; l < nLayers; l++)
        {
            int texIdx = (cx < tile.MclyTextureIds.GetLength(0) && cy < tile.MclyTextureIds.GetLength(1))
                ? tile.MclyTextureIds[cx, cy, l] : 0;

            uint flags = 0u;
            if (l > 0 && tile.McalAlphaPack != null)
            {
                flags |= 0x100u;
            }

            layers.Add(new LkMclyEntry((uint)texIdx, flags, (uint)alphaOffset, 0));

            if (l > 0 && tile.McalAlphaPack != null)
            {
                var chunkAlpha = SliceChunkAlphaBytes(tile.McalAlphaPack, cx, cy, l, ChunkAltSize);
                alphaOffset += chunkAlpha.Length;
                alphaMapData = alphaMapData != null
                    ? [.. alphaMapData, .. chunkAlpha]
                    : chunkAlpha;
            }
        }

        int holeMask = (cx < tile.HoleMask.GetLength(0) && cy < tile.HoleMask.GetLength(1))
            ? (tile.HoleMask[cx, cy] ? 1 : 0) : 0;

        float posX = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32f);
        float posY = -((ChunkSubSize * cy) + ChunkSize * tileY - ChunkSize * 32f);

        float worldZ = 0f;
        if (heights.Length >= 1)
            worldZ = heights[0] + baseHeight;

        var doodadRefs = FindDoodadRefs(modelPlacements, cx, cy, tileX, tileY);
        var worldModelRefs = FindWorldModelRefs(worldModelPlacements, cx, cy, tileX, tileY);

        return new LkMcnkData
        {
            IndexX = cx,
            IndexY = cy,
            Flags = (int)mcnkFlags,
            AreaId = areaId,
            NLayers = nLayers,
            HoleMask = holeMask,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = normals,
            ShadowMap = shadowMap,
            AlphaMapData = alphaMapData,
            AlphaMapSize = alphaMapData?.Length ?? 0,
            Layers = layers,
            DoodadRefs = doodadRefs,
            WorldModelRefs = worldModelRefs,
            PosX = posX,
            PosY = posY,
            PosZ = worldZ
        };
    }

    private static LkMcnkData CreateEmptyChunk(int cx, int cy, int tileX, int tileY)
    {
        float posX = -((ChunkSubSize * cx) + ChunkSize * tileX - ChunkSize * 32f);
        float posY = -((ChunkSubSize * cy) + ChunkSize * tileY - ChunkSize * 32f);

        return new LkMcnkData
        {
            IndexX = cx,
            IndexY = cy,
            Flags = 0,
            AreaId = 0,
            NLayers = 0,
            HoleMask = 0,
            BaseHeight = 0f,
            Heights = new float[McvtFloatCount],
            Normals = new byte[McnrByteCount],
            PosX = posX,
            PosY = posY,
            PosZ = 0f
        };
    }

    private static float[] ExtractChunkHeights(float[,] heightmap, int cx, int cy)
    {
        float[] heights = new float[145];
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

                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                    heights[idx] = heightmap[py, px];

                idx++;
            }
        }

        return heights;
    }

    private static float ComputeBaseHeight(float[] heights)
    {
        float min = float.MaxValue;
        for (int i = 0; i < heights.Length; i++)
        {
            if (heights[i] != 0f && heights[i] < min)
                min = heights[i];
        }
        return min == float.MaxValue ? 0f : min;
    }

    private static byte[] EncodeChunkNormals(float[,,] normalXyz, int cx, int cy)
    {
        byte[] result = new byte[448];
        if (normalXyz == null) return result;

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

                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize && idx + 2 < 435)
                {
                    result[idx] = EncodeNormal(normalXyz[py, px, 0]);
                    result[idx + 1] = EncodeNormal(normalXyz[py, px, 2]);
                    result[idx + 2] = EncodeNormal(normalXyz[py, px, 1]);
                }
                idx += 3;
            }
        }

        return result;
    }

    private static byte EncodeNormal(float value)
    {
        return unchecked((byte)(sbyte)Math.Clamp(MathF.Round(value * 127f), -128, 127));
    }

    private static byte[] SliceChunkShadow1024(float[,] shadowMask, int cx, int cy)
    {
        const int ChunkSize = 64;
        var shadow = new byte[ChunkSize * ChunkSize];
        int baseX = cx * ChunkSize;
        int baseY = cy * ChunkSize;

        for (int y = 0; y < ChunkSize; y++)
        {
            for (int x = 0; x < ChunkSize; x++)
            {
                int sy = baseY + y;
                int sx = baseX + x;
                if (sx < shadowMask.GetLength(1) && sy < shadowMask.GetLength(0))
                    shadow[y * ChunkSize + x] = shadowMask[sy, sx] > 0.5f ? (byte)0xFF : (byte)0;
            }
        }

        return shadow;
    }

    private static byte[] SliceChunkAlphaBytes(float[,,] alphaPack, int cx, int cy, int layer, int alphaSize)
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

    private static int FindLiquidType(AlphaTileData tile, int cx, int cy)
    {
        if (tile.LiquidChunks == null) return 0;
        foreach (var lc in tile.LiquidChunks)
        {
            if (lc.IndexX == cx && lc.IndexY == cy)
            {
                if ((lc.McnkFlags & 0x08) != 0) return 1;
                int bits = (int)((lc.McnkFlags >> 4) & 3);
                return bits switch { 1 => 1, 2 => 2, 3 => 3, _ => 0 };
            }
        }
        return 0;
    }

    private static List<int> FindDoodadRefs(List<LkMddfEntry> placements, int cx, int cy, int tileX, int tileY)
    {
        var refs = new List<int>();
        float chunkWorldX = MapOrigin - tileX * ChunkSize - cy * ChunkSubSize;
        float chunkWorldY = MapOrigin - tileY * ChunkSize - cx * ChunkSubSize;

        for (int i = 0; i < placements.Count; i++)
        {
            float dx = MathF.Abs(placements[i].Position.X - chunkWorldX);
            float dy = MathF.Abs(placements[i].Position.Y - chunkWorldY);
            if (dx < ChunkSize && dy < ChunkSize)
                refs.Add(i);
        }

        return refs;
    }

    private static List<int> FindWorldModelRefs(List<LkModfEntry> placements, int cx, int cy, int tileX, int tileY)
    {
        var refs = new List<int>();
        float chunkWorldX = MapOrigin - tileX * ChunkSize - cy * ChunkSubSize;
        float chunkWorldY = MapOrigin - tileY * ChunkSize - cx * ChunkSubSize;

        for (int i = 0; i < placements.Count; i++)
        {
            float dx = MathF.Abs(placements[i].Position.X - chunkWorldX);
            float dy = MathF.Abs(placements[i].Position.Y - chunkWorldY);
            if (dx < ChunkSize && dy < ChunkSize)
                refs.Add(i);
        }

        return refs;
    }

    private static IReadOnlyList<LkMddfEntry> RemapPlacements(
        IReadOnlyList<AlphaModelPlacement> alphaPlacements,
        IReadOnlyList<string> uniqueNames)
    {
        var nameIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < uniqueNames.Count; i++)
            nameIndex[uniqueNames[i]] = i;

        return alphaPlacements.Select(p => new LkMddfEntry(
            nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0,
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.Scale)).ToList();
    }

    private static IReadOnlyList<LkModfEntry> RemapWorldPlacements(
        IReadOnlyList<AlphaWorldModelPlacement> alphaPlacements,
        IReadOnlyList<string> uniqueNames)
    {
        var nameIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < uniqueNames.Count; i++)
            nameIndex[uniqueNames[i]] = i;

        return alphaPlacements.Select(p => new LkModfEntry(
            nameIndex.TryGetValue(p.ModelPath, out int idx) ? idx : 0,
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.BoundsMin,
            p.BoundsMax,
            p.Flags,
            DoodadSet: 0,
            NameSet: 0,
            Scale: 1.0f)).ToList();
    }

    private static uint ComputeMhdrFlags(AlphaTileData tile)
    {
        bool hasShadow = tile.McshShadowMask256 != null || tile.McshShadowMask1024 != null;
        return hasShadow ? 0x01u : 0x00u;
    }

    private const int McvtFloatCount = 145;
    private const int McnrByteCount = 448;
}