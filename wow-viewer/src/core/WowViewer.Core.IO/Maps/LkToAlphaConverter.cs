using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public sealed class LkToAlphaConversionResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public string? MapName { get; init; }
    public string? OutputPath { get; init; }
    public int TilesConverted { get; init; }
    public int TotalTiles { get; init; }
    public long ElapsedMs { get; init; }
    public List<string> Warnings { get; init; } = [];
}

public sealed class LkToAlphaOptions
{
    public bool Verbose { get; init; }
}

public static class LkToAlphaConverter
{
    private const float MapOrigin = 17066.666f;
    private const float ChunkSize = 533.33333f;
    private const float ChunkSubSize = ChunkSize / 16f;
    private const int TileHeightmapSize = 257;
    private const int ChunksPerTile = 16;

    public static AlphaTileData ConvertTile(LkAdtData adt, int tileX, int tileY, AreaIdMapper? areaIdMapper = null, string? sourceMapDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(adt);

        float[,] heightmap = new float[TileHeightmapSize, TileHeightmapSize];
        float[,,] normalXyz = new float[TileHeightmapSize, TileHeightmapSize, 3];
        float[,,] alphaPack = new float[1024, 1024, 4];
        float[,] shadowMask1024 = new float[1024, 1024];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        bool[,] holes = new bool[16, 16];
        int[,] areaIds = new int[16, 16];
        ushort[,] holeFullMasks = new ushort[16, 16];
        float[,,]? mccvRgb = new float[257, 257, 3];
        byte[,,]? mclvLighting = new byte[257, 257, 4];
        IReadOnlyList<int>[] mcrfDoodadRefsByChunk = new IReadOnlyList<int>[ChunksPerTile * ChunksPerTile];
        IReadOnlyList<int>[] mcrfWorldModelRefsByChunk = new IReadOnlyList<int>[ChunksPerTile * ChunksPerTile];
        IReadOnlyList<int>[] mcrfDoodadUniqueIdsByChunk = new IReadOnlyList<int>[ChunksPerTile * ChunksPerTile];
        IReadOnlyList<int>[] mcrfWorldModelUniqueIdsByChunk = new IReadOnlyList<int>[ChunksPerTile * ChunksPerTile];
        bool hasMccv = false;
        bool hasMclv = false;
        List<AlphaLiquidChunk> liquidChunks = [];

        float tileBaseHeight = ComputeTileBaseHeight(adt);

        for (int cy = 0; cy < ChunksPerTile; cy++)
        {
            for (int cx = 0; cx < ChunksPerTile; cx++)
            {
                int chunkIdx = cy * ChunksPerTile + cx;
                if (chunkIdx >= adt.Chunks.Count) continue;

                LkMcnkData chunk = adt.Chunks[chunkIdx];
                mcrfDoodadRefsByChunk[chunkIdx] = chunk.DoodadRefs.Count > 0 ? [.. chunk.DoodadRefs] : Array.Empty<int>();
                mcrfWorldModelRefsByChunk[chunkIdx] = chunk.WorldModelRefs.Count > 0 ? [.. chunk.WorldModelRefs] : Array.Empty<int>();
                mcrfDoodadUniqueIdsByChunk[chunkIdx] = MapDoodadRefsToUniqueIds(chunk.DoodadRefs, adt.ModelPlacements);
                mcrfWorldModelUniqueIdsByChunk[chunkIdx] = MapWorldModelRefsToUniqueIds(chunk.WorldModelRefs, adt.WorldModelPlacements);
                InjectChunkHeights(heightmap, chunk, tileBaseHeight, cx, cy);
                InjectChunkNormals(normalXyz, chunk, cx, cy);
                InjectChunkAlpha(alphaPack, chunk, adt.TextureNames, texIds, layerMask, cx, cy);
                InjectChunkShadow(shadowMask1024, chunk, cx, cy);
                holes[cx, cy] = chunk.HoleMask != 0;
                holeFullMasks[cx, cy] = (ushort)(chunk.HoleMask & 0xFFFF);
                areaIds[cx, cy] = areaIdMapper is null
                    ? chunk.AreaId
                    : areaIdMapper.MapAreaIdToAlpha(chunk.AreaId, sourceMapDirectory);

                if (chunk.MccvColors is { Length: >= 580 })
                {
                    hasMccv = true;
                    InjectChunkMccv(mccvRgb!, chunk.MccvColors, cx, cy);
                }

                if (chunk.MclvLighting is { Length: >= 580 })
                {
                    hasMclv = true;
                    InjectChunkMclv(mclvLighting!, chunk.MclvLighting, cx, cy);
                }

                if (chunk.LiquidData is { Layers.Count: > 0 })
                {
                    AlphaLiquidChunk? alphaLiquidChunk = BuildAlphaLiquidChunk(chunk, cx, cy);
                    if (alphaLiquidChunk is not null)
                        liquidChunks.Add(alphaLiquidChunk);
                }
                else if ((chunk.Flags & 0x3C) != 0)
                {
                    liquidChunks.Add(new AlphaLiquidChunk(
                        cy * ChunksPerTile + cx, cx, cy,
                        chunk.BaseHeight, chunk.BaseHeight,
                        null, (uint)chunk.Flags, null));
                }
            }
        }

        var modelPlacements = adt.ModelPlacements.Select(p => new AlphaModelPlacement(
            p.NameId,
            adt.ModelNames.Count > p.NameId ? adt.ModelNames[p.NameId] : $"unknown_{p.NameId}",
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.Scale)).ToList();

        var worldModelPlacements = adt.WorldModelPlacements.Select(p => new AlphaWorldModelPlacement(
            p.NameId,
            adt.WorldModelNames.Count > p.NameId ? adt.WorldModelNames[p.NameId] : $"unknown_{p.NameId}",
            p.UniqueId,
            p.Position,
            p.Rotation,
            p.BoundsMin,
            p.BoundsMax,
            p.Flags)).ToList();

        bool hasAlpha = false;
        bool hasShadow = false;
        for (int cy = 0; cy < ChunksPerTile; cy++)
        {
            for (int cx = 0; cx < ChunksPerTile; cx++)
            {
                for (int l = 1; l < 4; l++)
                {
                    if (layerMask[cx, cy, l]) hasAlpha = true;
                }
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        if (shadowMask1024[cy * 64 + y, cx * 64 + x] > 0.5f) hasShadow = true;
                    }
                }
            }
        }

        float[,,]? alphaPack256 = hasAlpha ? DownsampleAlphaPack(alphaPack) : null;
        float[,]? shadowMask256 = hasShadow ? DownsampleShadowMask(shadowMask1024) : null;

        FillHeightmapGaps(heightmap);

        return new AlphaTileData(
            $"lk-to-alpha({tileX},{tileY})",
            heightmap,
            alphaPack,
            texIds,
            layerMask,
            holes,
            adt.TextureNames.ToList(),
            modelPlacements,
            worldModelPlacements,
            liquidChunks,
            mcnrNormalXyz: normalXyz,
            mcshShadowMask256: shadowMask256,
            mcshShadowMask1024: hasShadow ? shadowMask1024 : null,
            areaIds: areaIds,
            mfboFlightBounds: adt.MfboFlightBounds,
            mccvRgb: hasMccv ? mccvRgb : null,
            mclvLightingBytes: hasMclv ? mclvLighting : null,
            holeFullMasks: holeFullMasks,
            mcrfDoodadRefsByChunk: mcrfDoodadRefsByChunk,
            mcrfWorldModelRefsByChunk: mcrfWorldModelRefsByChunk,
            mcrfDoodadUniqueIdsByChunk: mcrfDoodadUniqueIdsByChunk,
            mcrfWorldModelUniqueIdsByChunk: mcrfWorldModelUniqueIdsByChunk);
    }

    private static IReadOnlyList<int> MapDoodadRefsToUniqueIds(IReadOnlyList<int> refs, IReadOnlyList<LkMddfEntry> placements)
    {
        if (refs.Count == 0 || placements.Count == 0)
            return Array.Empty<int>();

        List<int> uniqueIds = [];
        foreach (int refIndex in refs)
        {
            if ((uint)refIndex < (uint)placements.Count)
                uniqueIds.Add(placements[refIndex].UniqueId);
        }

        return uniqueIds.Count > 0 ? uniqueIds : Array.Empty<int>();
    }

    private static IReadOnlyList<int> MapWorldModelRefsToUniqueIds(IReadOnlyList<int> refs, IReadOnlyList<LkModfEntry> placements)
    {
        if (refs.Count == 0 || placements.Count == 0)
            return Array.Empty<int>();

        List<int> uniqueIds = [];
        foreach (int refIndex in refs)
        {
            if ((uint)refIndex < (uint)placements.Count)
                uniqueIds.Add(placements[refIndex].UniqueId);
        }

        return uniqueIds.Count > 0 ? uniqueIds : Array.Empty<int>();
    }

    // Fill heightmap gaps for sparse tiles where some chunks had no height data.
    // Known limitation: uses 0.0f as sentinel, which could overwrite a legitimate
    // height of exactly 0.0f (sea level). In practice alpha terrain heights are
    // rarely exactly 0.0f at grid vertices, so this is acceptable.
    private static void FillHeightmapGaps(float[,] hm)
    {
        for (int y = 0; y < TileHeightmapSize; y++)
        {
            for (int x = 0; x < TileHeightmapSize; x++)
            {
                if (hm[y, x] != 0f) continue;
                if (x > 0 && hm[y, x - 1] != 0f) hm[y, x] = hm[y, x - 1];
                else if (y > 0 && hm[y - 1, x] != 0f) hm[y, x] = hm[y - 1, x];
                else if (x < TileHeightmapSize - 1 && hm[y, x + 1] != 0f) hm[y, x] = hm[y, x + 1];
                else if (y < TileHeightmapSize - 1 && hm[y + 1, x] != 0f) hm[y, x] = hm[y + 1, x];
            }
        }
    }

    private static float ComputeTileBaseHeight(LkAdtData adt)
    {
        float min = float.MaxValue;
        foreach (var chunk in adt.Chunks)
        {
            if (chunk.Heights != null)
            {
                foreach (float h in chunk.Heights)
                {
                    float abs = h + chunk.BaseHeight;
                    if (abs < min) min = abs;
                }
            }
        }
        return min == float.MaxValue ? 0f : min;
    }

    private static void InjectChunkHeights(float[,] heightmap, LkMcnkData chunk, float tileBaseHeight, int cx, int cy)
    {
        if (chunk.Heights == null || chunk.Heights.Length == 0) return;

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

                if (px < TileHeightmapSize && py < TileHeightmapSize && idx < chunk.Heights.Length)
                    heightmap[py, px] = chunk.Heights[idx] + chunk.BaseHeight;

                idx++;
            }
        }
    }

    private static void InjectChunkNormals(float[,,] normalXyz, LkMcnkData chunk, int cx, int cy)
    {
        if (chunk.Normals == null || chunk.Normals.Length == 0) return;

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

                if (px < TileHeightmapSize && py < TileHeightmapSize && idx + 2 < chunk.Normals.Length)
                {
                    float nx = DecodeNormal(chunk.Normals[idx]);
                    float ny = DecodeNormal(chunk.Normals[idx + 2]);
                    float nz = DecodeNormal(chunk.Normals[idx + 1]);
                    float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                    if (len > 0.001f) { nx /= len; ny /= len; nz /= len; }
                    else { nx = 0f; ny = 1f; nz = 0f; }

                    normalXyz[py, px, 0] = nx;
                    normalXyz[py, px, 1] = ny;
                    normalXyz[py, px, 2] = nz;
                }
                idx += 3;
            }
        }
    }

    private static void InjectChunkAlpha(float[,,] alphaPack, LkMcnkData chunk,
        IReadOnlyList<string> textureNames, int[,,] texIds, bool[,,] layerMask, int cx, int cy)
    {
        if (chunk.Layers == null || chunk.Layers.Count == 0) return;

        for (int l = 0; l < chunk.Layers.Count && l < 4; l++)
        {
            var layer = chunk.Layers[l];
            uint texId = layer.TextureId;
            texIds[cx, cy, l] = (int)texId;
            layerMask[cx, cy, l] = true;
        }

        if (chunk.AlphaMapData != null && chunk.AlphaMapData.Length > 0)
        {
            for (int l = 1; l < chunk.Layers.Count && l < 4; l++)
            {
                uint alphaOff = chunk.Layers[l].AlphaOffset;
                // 4.0.0 uses the ALPHA-era flag convention:
                //   0x200 = RLE compressed (not big alpha!)
                //   0x10000 = big alpha (8-bit uncompressed, 4096 bytes)
                // Ref: gillijimproject-csharp Mcal.cs MclyFlags.CompressedAlpha = 0x200
                // Ref: WoW_400_ADT_Analysis.md: "MCLY 0x200 is still the per-layer compressed-alpha flag"
                bool compressed = (chunk.Layers[l].Flags & 0x200) != 0;
                bool bigAlpha = (chunk.Layers[l].Flags & 0x10000) != 0;

                int remaining = chunk.AlphaMapData.Length - (int)alphaOff;

                if (compressed)
                {
                    byte[]? decoded = DecodeCompressedAlpha(chunk.AlphaMapData, (int)alphaOff, remaining);
                    if (decoded != null)
                        InjectAlphaLayer(alphaPack, decoded, cx, cy, l, 64);
                    continue;
                }

                if (remaining <= 0)
                    continue;

                if (bigAlpha || remaining >= 4096)
                {
                    int take = Math.Min(4096, remaining);
                    for (int y = 0; y < 64; y++)
                        for (int x = 0; x < 64; x++)
                        {
                            int src = (int)alphaOff + y * 64 + x;
                            if (src < chunk.AlphaMapData.Length)
                                alphaPack[cy * 64 + y, cx * 64 + x, l] = chunk.AlphaMapData[src] / 255f;
                        }
                }
                else if (remaining >= 2048)
                {
                    int take = Math.Min(2048, remaining);
                    for (int i = 0; i < take; i++)
                    {
                        byte b = chunk.AlphaMapData[(int)alphaOff + i];
                        int ax = (i * 2) % 64;
                        int ay = (i * 2) / 64;
                        float lo = ((b & 0x0F) * 17) / 255f;
                        float hi = (((b >> 4) & 0x0F) * 17) / 255f;
                        alphaPack[cy * 64 + ay, cx * 64 + ax, l] = lo;
                        if (ax + 1 < 64)
                            alphaPack[cy * 64 + ay, cx * 64 + ax + 1, l] = hi;
                    }
                }
            }
        }

        // Synthesize residual alpha for overlay layers without direct data
        // In 4.0.0, the last active overlay layer gets residual coverage:
        // alpha_last = 1.0 - sum(alpha_prev_overlays)
        int lastActiveLayer = -1;
        for (int l = 3; l >= 1; l--)
        {
            if (layerMask[cx, cy, l]) { lastActiveLayer = l; break; }
        }
        if (lastActiveLayer > 0)
        {
            bool hasAlpha = false;
            for (int y = 0; y < 64 && !hasAlpha; y++)
                for (int x = 0; x < 64 && !hasAlpha; x++)
                    if (alphaPack[cy * 64 + y, cx * 64 + x, lastActiveLayer] > 0.01f)
                        hasAlpha = true;

            if (!hasAlpha)
            {
                for (int y = 0; y < 64; y++)
                    for (int x = 0; x < 64; x++)
                    {
                        float sum = 0f;
                        for (int prev = 1; prev < lastActiveLayer; prev++)
                            sum += alphaPack[cy * 64 + y, cx * 64 + x, prev];
                        alphaPack[cy * 64 + y, cx * 64 + x, lastActiveLayer] = Math.Clamp(1f - sum, 0f, 1f);
                    }
            }
        }
    }

    private static void InjectAlphaLayer(float[,,] alphaPack, byte[] decoded, int cx, int cy, int layer, int size)
    {
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int src = y * size + x;
                int dstY = cy * 64 + y;
                int dstX = cx * 64 + x;
                if (dstY < 1024 && dstX < 1024 && src < decoded.Length)
                    alphaPack[dstY, dstX, layer] = decoded[src] / 255f;
            }
        }
    }

    private static byte[]? DecodeCompressedAlpha(byte[] data, int offset, int available)
    {
        var result = new byte[4096];
        int produced = 0;
        int p = offset;
        int limit = Math.Min(data.Length, offset + available);

        for (int row = 0; row < 64 && produced < 4096; row++)
        {
            int rowPos = 0;
            while (rowPos < 64 && produced < 4096)
            {
                if (p >= limit) return null;
                byte control = data[p++];
                bool fill = (control & 0x80) != 0;
                int count = control & 0x7F;
                if (count == 0) count = 64;
                int room = 64 - rowPos;
                int take = Math.Min(count, room);
                if (fill)
                {
                    if (p >= limit) return null;
                    byte v = data[p++];
                    for (int i = 0; i < take; i++) result[produced + i] = v;
                }
                else
                {
                    if (p + take > limit) return null;
                    Buffer.BlockCopy(data, p, result, produced, take);
                    p += take;
                }
                produced += take;
                rowPos += take;
            }
        }

        return produced == 4096 ? result : null;
    }

    private static void InjectChunkShadow(float[,] shadowMask, LkMcnkData chunk, int cx, int cy)
    {
        if (chunk.ShadowMap == null || chunk.ShadowMap.Length == 0) return;

        const int chunkSize = 64;
        int shadowSize = (int)Math.Sqrt(chunk.ShadowMap.Length);
        if (shadowSize != 64) return;

        for (int y = 0; y < chunkSize; y++)
        {
            for (int x = 0; x < chunkSize; x++)
            {
                int src = y * chunkSize + x;
                if (src < chunk.ShadowMap.Length)
                {
                    int dstY = cy * chunkSize + y;
                    int dstX = cx * chunkSize + x;
                    if (dstY < 1024 && dstX < 1024)
                        shadowMask[dstY, dstX] = chunk.ShadowMap[src] > 0x7F ? 1.0f : 0.0f;
                }
            }
        }
    }

    private static float DecodeNormal(byte b) => (sbyte)b / 127f;

    private static void InjectChunkMccv(float[,,] mccvRgb, byte[] chunkColors, int cx, int cy)
    {
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

                if (px < 257 && py < 257 && idx + 3 < chunkColors.Length)
                {
                    mccvRgb[py, px, 0] = chunkColors[idx] / 255f;
                    mccvRgb[py, px, 1] = chunkColors[idx + 1] / 255f;
                    mccvRgb[py, px, 2] = chunkColors[idx + 2] / 255f;
                }
                idx += 4;
            }
        }
    }

    private static void InjectChunkMclv(byte[,,] mclvLighting, byte[] chunkLighting, int cx, int cy)
    {
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

                if (px < 257 && py < 257 && idx + 3 < chunkLighting.Length)
                {
                    mclvLighting[py, px, 0] = chunkLighting[idx];
                    mclvLighting[py, px, 1] = chunkLighting[idx + 1];
                    mclvLighting[py, px, 2] = chunkLighting[idx + 2];
                    mclvLighting[py, px, 3] = chunkLighting[idx + 3];
                }
                idx += 4;
            }
        }
    }

    private static byte ClassifyLkLiquid(int flags)
    {
        if ((flags & 0x04) != 0) return 1;
        if ((flags & 0x08) != 0) return 1;
        int bits = (flags >> 4) & 3;
        return bits switch { 1 => 1, 2 => 2, 3 => 3, _ => 0 };
    }

    private static AlphaLiquidChunk? BuildAlphaLiquidChunk(LkMcnkData chunk, int cx, int cy)
    {
        AdtLiquidLayer? layer = chunk.LiquidData?.Layers.FirstOrDefault();
        if (layer is null)
            return null;

        uint mcnkFlags = AlphaLiquidTypeCodec.GetWriterChunkFlags(layer.BasicType);

        byte[]? tileFlags = BuildAlphaTileFlags(layer);
        float[] heights = BuildAlphaLiquidHeights(layer);

        return new AlphaLiquidChunk(
            cy * ChunksPerTile + cx,
            cx,
            cy,
            heights.Min(),
            heights.Max(),
            tileFlags,
            mcnkFlags,
            heights);
    }

    private static byte[]? BuildAlphaTileFlags(AdtLiquidLayer layer)
    {
        if (layer.Width <= 0 || layer.Height <= 0)
            return null;

        byte[] tileFlags = new byte[64];
        Array.Fill(tileFlags, (byte)0x0F);
        byte visibleTileFlag = AlphaLiquidTypeCodec.GetWriterTileTypeNibble(layer.BasicType);

        for (int y = 0; y < layer.Height; y++)
        {
            for (int x = 0; x < layer.Width; x++)
            {
                int globalX = layer.XOffset + x;
                int globalY = layer.YOffset + y;
                if ((uint)globalX >= 8 || (uint)globalY >= 8)
                    continue;

                if (layer.TileExists(x, y))
                    tileFlags[(globalY * 8) + globalX] = visibleTileFlag;
            }
        }

        return tileFlags;
    }

    private static float[] BuildAlphaLiquidHeights(AdtLiquidLayer layer)
    {
        float[] heights = new float[81];
        float fallbackHeight = layer.Heights is { Length: > 0 }
            ? layer.Heights[0]
            : (layer.MinHeight + layer.MaxHeight) * 0.5f;
        Array.Fill(heights, fallbackHeight);

        if (layer.Heights is not { Length: > 0 })
            return heights;

        int srcWidth = layer.Width + 1;
        int srcHeight = layer.Height + 1;
        for (int y = 0; y < srcHeight; y++)
        {
            for (int x = 0; x < srcWidth; x++)
            {
                int globalX = layer.XOffset + x;
                int globalY = layer.YOffset + y;
                if ((uint)globalX >= 9 || (uint)globalY >= 9)
                    continue;

                int srcIndex = (y * srcWidth) + x;
                if ((uint)srcIndex >= (uint)layer.Heights.Length)
                    continue;

                heights[(globalY * 9) + globalX] = layer.Heights[srcIndex];
            }
        }

        return heights;
    }

    private static float[,,] DownsampleAlphaPack(float[,,] src)
    {
        const int srcSize = 1024;
        const int dstSize = 256;
        const int ratio = srcSize / dstSize;
        const int samples = ratio * ratio;
        var dst = new float[dstSize, dstSize, 4];

        for (int y = 0; y < dstSize; y++)
        {
            for (int x = 0; x < dstSize; x++)
            {
                for (int l = 0; l < 4; l++)
                {
                    float sum = 0f;
                    for (int dy = 0; dy < ratio; dy++)
                    {
                        for (int dx = 0; dx < ratio; dx++)
                        {
                            sum += src[y * ratio + dy, x * ratio + dx, l];
                        }
                    }
                    dst[y, x, l] = sum / samples;
                }
            }
        }
        return dst;
    }

    private static float[,] DownsampleShadowMask(float[,] src)
    {
        const int srcSize = 1024;
        const int dstSize = 256;
        const int ratio = srcSize / dstSize;
        const int samples = ratio * ratio;
        var dst = new float[dstSize, dstSize];

        for (int y = 0; y < dstSize; y++)
        {
            for (int x = 0; x < dstSize; x++)
            {
                float sum = 0f;
                for (int dy = 0; dy < ratio; dy++)
                {
                    for (int dx = 0; dx < ratio; dx++)
                    {
                        sum += src[y * ratio + dy, x * ratio + dx];
                    }
                }
                dst[y, x] = sum / samples;
            }
        }
        return dst;
    }
}