using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AlphaTensorPackBuilder
{
    private const int TileHeightmapSize = 257;
    private const int TileChunks = 16;
    private const int VerticesPerChunk = 17;
    private const int TileLiquidSize = TileChunks * VerticesPerChunk;

    public static TerrainTileTensorPack Build(AlphaTileData tileData)
    {
        HashSet<string> signals = [];

        string sourcePath = tileData.SourcePath;
        string tileName = ExtractTileName(sourcePath);
        string mapName = ExtractMapNameFromTilePath(sourcePath);

        float[,]? height257 = tileData.Heightmap;
        if (height257 is not null)
            signals.Add("height_257");

        if (tileData.McnrNormalXyz is not null)
            signals.Add("mcnr_normal_xyz");

        float[,]? height65 = DownsampleHeightmap(height257, 65);
        float[,]? height17 = DownsampleHeightmap(height257, 17);

        float[,,]? mcalAlphaPack256 = tileData.McalAlphaPack;

        float[,]? mclqSurfaceHeight257 = tileData.MclqSurfaceHeight;
        int[,]? mclqTypeMask257 = tileData.MclqTypeMask;
        if (mclqSurfaceHeight257 != null)
        {
            signals.Add("mclq_surface_height");
            signals.Add("mclq_type_mask");
        }
        else
        {
            BuildAlphaLiquid(tileData, ref mclqSurfaceHeight257, ref mclqTypeMask257, signals);
        }

        bool[,]? holeMask16 = tileData.HoleMask;
        if (holeMask16 is not null)
            signals.Add("hole_mask_16");

        if (tileData.MclyTextureIds is not null)
            signals.Add("mcly_texture_ids");
        if (tileData.MclyLayerMask is not null)
            signals.Add("mcly_layer_mask");
        if (tileData.McalAlphaPack is not null)
            signals.Add("mcal_alpha_pack_256");
        if (tileData.McshShadowMask256 is not null)
            signals.Add("mcsh_shadow_mask_256");

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = mapName,
            BuildKey = "alpha",
            SourceAdtPath = sourcePath,
            Height257 = height257,
            Height65 = height65,
            Height17 = height17,
            MclyTextureIds = tileData.MclyTextureIds,
            MclyTextureNames = tileData.TextureNames,
            MclyLayerMask = tileData.MclyLayerMask,
            McalAlphaPack256 = mcalAlphaPack256,
            McnrNormalXyz = tileData.McnrNormalXyz,
            McshShadowMask256 = tileData.McshShadowMask256,
            MccvRgb = null,
            Mh2oSurfaceHeight = null,
            Mh2oDepth = null,
            Mh2oTypeMask = null,
            MclqSurfaceHeight = mclqSurfaceHeight257,
            MclqTypeMask = mclqTypeMask257,
            HoleMask16 = holeMask16,
            AvailableSignals = signals,
        };
    }

    private static string ExtractTileName(string sourcePath)
    {
        int hash = sourcePath.IndexOf('#');
        string wdtName = Path.GetFileNameWithoutExtension(sourcePath);
        if (hash >= 0)
        {
            string tilePart = sourcePath[(hash + 1)..];
            return $"{wdtName}_{tilePart}";
        }
        return wdtName;
    }

    private static string ExtractMapNameFromTilePath(string sourcePath)
    {
        string fileName = Path.GetFileNameWithoutExtension(sourcePath);
        int hash = fileName.IndexOf('#');
        if (hash >= 0)
            fileName = fileName[..hash];
        return fileName;
    }

    private static void BuildAlphaLiquid(
        AlphaTileData tileData,
        ref float[,]? mclqSurfaceHeight,
        ref int[,]? mclqTypeMask,
        HashSet<string> signals)
    {
        if (tileData.LiquidChunks.Count == 0)
            return;

        mclqSurfaceHeight = new float[TileLiquidSize, TileLiquidSize];
        mclqTypeMask = new int[TileLiquidSize, TileLiquidSize];

        for (int i = 0; i < TileLiquidSize; i++)
            for (int j = 0; j < TileLiquidSize; j++)
                mclqTypeMask[i, j] = -1;

        foreach (AlphaLiquidChunk lc in tileData.LiquidChunks)
        {
            int chunkIndex = lc.ChunkIndex;
            int chunkX = lc.IndexX;
            int chunkY = lc.IndexY;

            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            int baseX = chunkX * VerticesPerChunk;
            int baseY = chunkY * VerticesPerChunk;

            float avgHeight = (lc.MinHeight + lc.MaxHeight) * 0.5f;
            int liquidType = McnkFlagsToLiquidType(lc.MinHeight, lc.MaxHeight, lc.McnkFlags);

            for (int vy = 0; vy < VerticesPerChunk; vy++)
            {
                for (int vx = 0; vx < VerticesPerChunk; vx++)
                {
                    int gx = baseX + vx;
                    int gy = baseY + vy;
                    if ((uint)gx < TileLiquidSize && (uint)gy < TileLiquidSize)
                    {
                        mclqSurfaceHeight[gy, gx] = avgHeight;
                        mclqTypeMask[gy, gx] = liquidType;
                    }
                }
            }
        }

        signals.Add("mclq_surface_height");
        signals.Add("mclq_type_mask");
    }

    private static int McnkFlagsToLiquidType(float minH, float maxH, uint mcnkFlags)
    {
        if ((mcnkFlags & 0x08u) != 0)
            return 1;
        int liquidBits = (int)((mcnkFlags >> 4) & 3u);
        return liquidBits;
    }

    private static float[,]? DownsampleHeightmap(float[,]? source, int targetSize)
    {
        if (source is null)
            return null;

        float[,] result = new float[targetSize, targetSize];
        int sourceSize = source.GetLength(0);
        float scale = (float)(sourceSize - 1) / (targetSize - 1);

        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                float sourceX = x * scale;
                float sourceY = y * scale;
                int ix = Math.Clamp((int)sourceX, 0, sourceSize - 2);
                int iy = Math.Clamp((int)sourceY, 0, sourceSize - 2);
                float fx = sourceX - ix;
                float fy = sourceY - iy;

                float v00 = source[iy, ix];
                float v10 = source[iy, ix + 1];
                float v01 = source[iy + 1, ix];
                float v11 = source[iy + 1, ix + 1];

                result[y, x] = BilinearInterpolate(v00, v10, v01, v11, fx, fy);
            }
        }

        return result;
    }

    private static float BilinearInterpolate(float v00, float v10, float v01, float v11, float fx, float fy)
    {
        float top = v00 + (v10 - v00) * fx;
        float bottom = v01 + (v11 - v01) * fx;
        return top + (bottom - top) * fy;
    }
}
