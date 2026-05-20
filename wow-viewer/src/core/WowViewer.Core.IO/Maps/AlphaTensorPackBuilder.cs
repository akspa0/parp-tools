using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AlphaTensorPackBuilder
{
    private const int TileHeightmapSize = 257;
    private const int TileChunks = 16;
    private const int VerticesPerChunk = 17;
    private const int TileLiquidSize = TileChunks * VerticesPerChunk;
    private const float ObjectTileSize = 533.33333f;
    private const float ObjectMapOrigin = 17066.666f;
    private const float ObjectMaskMarginTiles = 0.25f;

    public static TerrainTileTensorPack Build(AlphaTileData tileData, int tileX, int tileY)
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

        float[,,]? mcalAlphaPack = tileData.McalAlphaPack;

        float[,]? mclqSurfaceHeight257 = tileData.MclqSurfaceHeight;
        int[,]? mclqTypeMask257 = tileData.MclqTypeMask;
        bool[,]? mclqPresenceMask257 = null;
        if (mclqSurfaceHeight257 != null)
        {
            signals.Add("mclq_surface_height");
            signals.Add("mclq_type_mask");
            if (mclqTypeMask257 is not null)
                mclqPresenceMask257 = BuildPresenceMaskFromTypeMask(mclqTypeMask257);
        }
        else
        {
            BuildAlphaLiquid(tileData, ref mclqSurfaceHeight257, ref mclqTypeMask257, ref mclqPresenceMask257, signals);
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
        if (tileData.RawChunks.Count > 0)
            signals.Add("raw_alpha_chunks");

        // Generate object footprint masks from MDDF/MODF placements
        float[,]? objectMask257 = null;
        float[,]? objectPreciseMask257 = null;
        int[,]? objectInstanceMask257 = null;
        BuildObjectMasks(tileData, tileX, tileY, ref objectMask257, ref objectPreciseMask257, ref objectInstanceMask257, signals);

        // Shadow residual: MCSH shadow not explained by objects
        float[,]? shadowResidual256 = null;
        if (tileData.McshShadowMask256 is not null && objectPreciseMask257 is not null)
        {
            shadowResidual256 = BuildShadowResidual(tileData.McshShadowMask256, objectPreciseMask257);
            if (shadowResidual256 is not null)
                signals.Add("shadow_residual_mask_256");
        }

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = mapName,
            BuildKey = "alpha",
            SourceAdtPath = sourcePath,
            TileX = tileX,
            TileY = tileY,
            Height257 = height257,
            Height65 = height65,
            Height17 = height17,
            MclyTextureIds = tileData.MclyTextureIds,
            MclyTextureNames = tileData.TextureNames,
            MclyLayerMask = tileData.MclyLayerMask,
            McalAlphaPack = mcalAlphaPack,
            McalAlphaPack256 = DownsampleAlpha256(mcalAlphaPack),
            McnrNormalXyz = tileData.McnrNormalXyz,
            McshShadowMask256 = tileData.McshShadowMask256,
            MccvRgb = null,
            Mh2oSurfaceHeight = null,
            Mh2oDepth = null,
            Mh2oTypeMask = null,
            Mh2oPresenceMask = null,
            MclqSurfaceHeight = mclqSurfaceHeight257,
            MclqTypeMask = mclqTypeMask257,
            MclqPresenceMask = mclqPresenceMask257,
            HoleMask16 = holeMask16,
            ObjectMask257 = objectMask257,
            ObjectPreciseMask257 = objectPreciseMask257,
            ObjectInstanceMask257 = objectInstanceMask257,
            ShadowResidualMask256 = shadowResidual256,
            PlacementMddfCount = tileData.ModelPlacements.Count,
            PlacementModfCount = tileData.WorldModelPlacements.Count,
            PlacementMddfData = BuildPlacementMddfData(tileData.ModelPlacements),
            PlacementModfData = BuildPlacementModfData(tileData.WorldModelPlacements),
            PlacementMddfNames = BuildIndexedModelNames(tileData.ModelPlacements),
            PlacementModfNames = BuildIndexedWorldModelNames(tileData.WorldModelPlacements),
            RawChunks = tileData.RawChunks,
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
        ref bool[,]? mclqPresenceMask,
        HashSet<string> signals)
    {
        if (tileData.LiquidChunks.Count == 0)
            return;

        mclqSurfaceHeight = new float[TileLiquidSize, TileLiquidSize];
        mclqTypeMask = new int[TileLiquidSize, TileLiquidSize];
        mclqPresenceMask = new bool[TileLiquidSize, TileLiquidSize];

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
                        mclqPresenceMask[gy, gx] = true;
                    }
                }
            }
        }

        signals.Add("mclq_surface_height");
        signals.Add("mclq_type_mask");
    }

    private static bool[,] BuildPresenceMaskFromTypeMask(int[,] typeMask)
    {
        int h = typeMask.GetLength(0);
        int w = typeMask.GetLength(1);
        bool[,] presence = new bool[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
                presence[y, x] = typeMask[y, x] >= 0;
        }

        return presence;
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

    private static float[,,]? DownsampleAlpha256(float[,,]? alpha)
    {
        if (alpha is null)
            return null;

        int srcSize = alpha.GetLength(0);
        int channels = alpha.GetLength(2);
        const int DstSize = 256;

        float scale = (float)srcSize / DstSize;
        float[,,] result = new float[DstSize, DstSize, channels];

        for (int y = 0; y < DstSize; y++)
        {
            for (int x = 0; x < DstSize; x++)
            {
                int srcX0 = (int)(x * scale);
                int srcY0 = (int)(y * scale);
                int srcX1 = Math.Min(srcX0 + 1, srcSize - 1);
                int srcY1 = Math.Min(srcY0 + 1, srcSize - 1);
                float fx = (x * scale) - srcX0;
                float fy = (y * scale) - srcY0;

                for (int c = 0; c < channels; c++)
                {
                    float v00 = alpha[srcY0, srcX0, c];
                    float v10 = alpha[srcY0, srcX1, c];
                    float v01 = alpha[srcY1, srcX0, c];
                    float v11 = alpha[srcY1, srcX1, c];
                    result[y, x, c] = v00 + (v10 - v00) * fx + (v01 - v00) * fy + (v00 - v10 - v01 + v11) * fx * fy;
                }
            }
        }

        return result;
    }

    private static void BuildObjectMasks(
        AlphaTileData tileData, int tileX, int tileY,
        ref float[,]? objectMask,
        ref float[,]? objectPreciseMask,
        ref int[,]? objectInstanceMask,
        HashSet<string> signals)
    {
        if (tileData.ModelPlacements.Count == 0 && tileData.WorldModelPlacements.Count == 0)
            return;

        objectMask = new float[TileHeightmapSize, TileHeightmapSize];
        objectPreciseMask = new float[TileHeightmapSize, TileHeightmapSize];
        objectInstanceMask = new int[TileHeightmapSize, TileHeightmapSize];
        int instanceId = 1;

        foreach (var p in tileData.ModelPlacements)
        {
            if (!TryProjectPlacementToTilePixel(p.Position, tileX, tileY, out int px, out int py))
                continue;

            float r = MathF.Max(1.5f, p.Scale * 2f);
            PaintCircle(objectMask, px, py, 2f, 1.0f);
            PaintSoftCircle(objectPreciseMask, px, py, r);
            PaintIntCircle(objectInstanceMask, px, py, 2f, instanceId);
            instanceId++;
        }

        foreach (var p in tileData.WorldModelPlacements)
        {
            Vector3 min = p.BoundsMin, max = p.BoundsMax;
            if (min.X < max.X && min.Y < max.Y && !float.IsNaN(min.X) && !float.IsNaN(max.X))
            {
                ProjectBoundsToTilePixels(min, max, tileX, tileY, out int minPx, out int minPy, out int maxPx, out int maxPy);
                if (minPx > maxPx || minPy > maxPy)
                    continue;

                PaintRect(objectMask, minPx, minPy, maxPx, maxPy, 1.0f);
                PaintSoftRect(objectPreciseMask, minPx, minPy, maxPx, maxPy);
                PaintIntRect(objectInstanceMask, minPx, minPy, maxPx, maxPy, instanceId);
                instanceId++;
            }
            else
            {
                if (!TryProjectPlacementToTilePixel(p.Position, tileX, tileY, out int px, out int py))
                    continue;

                PaintCircle(objectMask, px, py, 3f, 1.0f);
                PaintSoftCircle(objectPreciseMask, px, py, 3f);
                PaintIntCircle(objectInstanceMask, px, py, 3f, instanceId);
                instanceId++;
            }
        }

        signals.Add("object_mask_257");
        signals.Add("object_precise_mask_257");
        signals.Add("object_instance_mask_257");
    }

    private static bool TryProjectPlacementToTilePixel(Vector3 position, int tileX, int tileY, out int pixelX, out int pixelY)
    {
        pixelX = 0;
        pixelY = 0;

        (float U, float V)[] candidates =
        [
            ((position.X / ObjectTileSize) - tileX, (position.Z / ObjectTileSize) - tileY),
            (((ObjectMapOrigin - position.Z) / ObjectTileSize) - tileX, ((ObjectMapOrigin - position.X) / ObjectTileSize) - tileY),
            ((position.X / ObjectTileSize) - tileX, (position.Y / ObjectTileSize) - tileY),
            (((ObjectMapOrigin - position.Y) / ObjectTileSize) - tileX, ((ObjectMapOrigin - position.X) / ObjectTileSize) - tileY),
        ];

        float bestOverflow = float.PositiveInfinity;
        (float U, float V) best = default;
        bool found = false;

        foreach ((float U, float V) candidate in candidates)
        {
            float overflow =
                MathF.Max(0f, -candidate.U) + MathF.Max(0f, candidate.U - 1f) +
                MathF.Max(0f, -candidate.V) + MathF.Max(0f, candidate.V - 1f);

            if (overflow < bestOverflow)
            {
                bestOverflow = overflow;
                best = candidate;
                found = true;
                if (overflow <= 0.000001f)
                    break;
            }
        }

        if (!found ||
            best.U < -ObjectMaskMarginTiles || best.U > 1f + ObjectMaskMarginTiles ||
            best.V < -ObjectMaskMarginTiles || best.V > 1f + ObjectMaskMarginTiles)
        {
            return false;
        }

        pixelX = Math.Clamp((int)MathF.Round(Math.Clamp(best.U, 0f, 1f) * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
        pixelY = Math.Clamp((int)MathF.Round(Math.Clamp(best.V, 0f, 1f) * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
        return true;
    }

    private static void ProjectBoundsToTilePixels(
        Vector3 min,
        Vector3 max,
        int tileX,
        int tileY,
        out int minPx,
        out int minPy,
        out int maxPx,
        out int maxPy)
    {
        minPx = int.MaxValue;
        minPy = int.MaxValue;
        maxPx = int.MinValue;
        maxPy = int.MinValue;

        ReadOnlySpan<Vector3> corners =
        [
            new(min.X, min.Y, min.Z),
            new(min.X, max.Y, min.Z),
            new(max.X, min.Y, min.Z),
            new(max.X, max.Y, min.Z),
            new(min.X, min.Y, max.Z),
            new(min.X, max.Y, max.Z),
            new(max.X, min.Y, max.Z),
            new(max.X, max.Y, max.Z),
        ];

        foreach (Vector3 corner in corners)
        {
            if (!TryProjectPlacementToTilePixel(corner, tileX, tileY, out int px, out int py))
                continue;

            minPx = Math.Min(minPx, px);
            minPy = Math.Min(minPy, py);
            maxPx = Math.Max(maxPx, px);
            maxPy = Math.Max(maxPy, py);
        }

        if (minPx == int.MaxValue)
        {
            minPx = 1;
            minPy = 1;
            maxPx = 0;
            maxPy = 0;
        }
    }

    private static void PaintIntCircle(int[,] buffer, int cx, int cy, float radius, int value)
    {
        int r = (int)MathF.Ceiling(radius);
        int rSq = r * r;
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                if ((dx * dx) + (dy * dy) > rSq)
                    continue;
                int x = cx + dx;
                int y = cy + dy;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;
                buffer[y, x] = value;
            }
        }
    }

    private static void PaintIntRect(int[,] buffer, int minX, int minY, int maxX, int maxY, int value)
    {
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;
                buffer[y, x] = value;
            }
        }
    }

    private static float[,]? BuildShadowResidual(float[,] shadowMask256, float[,] objectPreciseMask257)
    {
        const int size = 256;
        var result = new float[size, size];
        bool any = false;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float shadow = shadowMask256[y, x];
                // Use center-weighted sample from 257 grid
                int oy = Math.Clamp(y * 257 / 256, 0, 256);
                int ox = Math.Clamp(x * 257 / 256, 0, 256);
                float obj = objectPreciseMask257[oy, ox];
                float residual = MathF.Max(0f, shadow - Math.Clamp(obj, 0f, 1f));
                result[y, x] = residual;
                if (residual > 0f) any = true;
            }
        }
        return any ? result : null;
    }

    private static float[,]? BuildPlacementMddfData(IReadOnlyList<AlphaModelPlacement> placements)
    {
        if (placements.Count == 0) return null;
        var data = new float[placements.Count, 9];
        for (int i = 0; i < placements.Count; i++)
        {
            var p = placements[i];
            data[i, 0] = p.NameId;
            data[i, 1] = p.UniqueId;
            data[i, 2] = p.Position.X;
            data[i, 3] = p.Position.Y;
            data[i, 4] = p.Position.Z;
            data[i, 5] = p.Rotation.X;
            data[i, 6] = p.Rotation.Y;
            data[i, 7] = p.Rotation.Z;
            data[i, 8] = p.Scale;
        }
        return data;
    }

    private static IReadOnlyList<string> BuildIndexedModelNames(IReadOnlyList<AlphaModelPlacement> placements)
    {
        int maxNameId = placements.Count == 0 ? -1 : placements.Max(static p => p.NameId);
        if (maxNameId < 0) return Array.Empty<string>();

        string[] names = new string[maxNameId + 1];
        Array.Fill(names, string.Empty);
        foreach (var placement in placements)
        {
            if (placement.NameId >= 0 && placement.NameId < names.Length && names[placement.NameId].Length == 0)
                names[placement.NameId] = placement.ModelPath;
        }

        return names;
    }

    private static float[,]? BuildPlacementModfData(IReadOnlyList<AlphaWorldModelPlacement> placements)
    {
        if (placements.Count == 0) return null;
        var data = new float[placements.Count, 14];
        for (int i = 0; i < placements.Count; i++)
        {
            var p = placements[i];
            data[i, 0] = p.NameId;
            data[i, 1] = p.UniqueId;
            data[i, 2] = p.Position.X;
            data[i, 3] = p.Position.Y;
            data[i, 4] = p.Position.Z;
            data[i, 5] = p.Rotation.X;
            data[i, 6] = p.Rotation.Y;
            data[i, 7] = p.Rotation.Z;
            data[i, 8] = p.BoundsMin.X;
            data[i, 9] = p.BoundsMin.Y;
            data[i, 10] = p.BoundsMin.Z;
            data[i, 11] = p.BoundsMax.X;
            data[i, 12] = p.BoundsMax.Y;
            data[i, 13] = p.BoundsMax.Z;
        }
        return data;
    }

    private static IReadOnlyList<string> BuildIndexedWorldModelNames(IReadOnlyList<AlphaWorldModelPlacement> placements)
    {
        int maxNameId = placements.Count == 0 ? -1 : placements.Max(static p => p.NameId);
        if (maxNameId < 0) return Array.Empty<string>();

        string[] names = new string[maxNameId + 1];
        Array.Fill(names, string.Empty);
        foreach (var placement in placements)
        {
            if (placement.NameId >= 0 && placement.NameId < names.Length && names[placement.NameId].Length == 0)
                names[placement.NameId] = placement.ModelPath;
        }

        return names;
    }

    private const float ObjectWorldTileSize = 533.33333f;

    private static void PaintCircle(float[,] buf, int cx, int cy, float radius, float value)
    {
        int r = (int)MathF.Ceiling(radius);
        int r2 = r * r;
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                if (dx * dx + dy * dy > r2) continue;
                int px = cx + dx, py = cy + dy;
                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                    buf[py, px] = value;
            }
        }
    }

    private static void PaintSoftCircle(float[,] buf, int cx, int cy, float radius)
    {
        int r = (int)MathF.Ceiling(radius * 1.5f);
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                if (dist > radius * 1.5f) continue;
                float a = 1f - MathF.Min(1f, dist / radius);
                if (a <= 0f) continue;
                int px = cx + dx, py = cy + dy;
                if ((uint)px < TileHeightmapSize && (uint)py < TileHeightmapSize)
                    buf[py, px] = Math.Max(buf[py, px], a);
            }
        }
    }

    private static void PaintRect(float[,] buf, int x0, int y0, int x1, int y1, float value)
    {
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                if ((uint)x < TileHeightmapSize && (uint)y < TileHeightmapSize)
                    buf[y, x] = value;
            }
        }
    }

    private static void PaintSoftRect(float[,] buf, int x0, int y0, int x1, int y1)
    {
        int pad = 2;
        for (int y = y0 - pad; y <= y1 + pad; y++)
        {
            for (int x = x0 - pad; x <= x1 + pad; x++)
            {
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize) continue;
                float dx = (x < x0) ? x0 - x : (x > x1) ? x - x1 : 0f;
                float dy = (y < y0) ? y0 - y : (y > y1) ? y - y1 : 0f;
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                float a = 1f - MathF.Min(1f, dist / pad);
                buf[y, x] = Math.Max(buf[y, x], a);
            }
        }
    }
}
