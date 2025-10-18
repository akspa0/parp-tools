using System;
using System.Collections.Generic;
using System.Linq;

namespace WoWRollback.LkToAlphaModule.Liquids;

/// <summary>
/// Bidirectional converter between LK MH2O chunks and Alpha-era MCLQ payloads.
/// Ported from Next.Core implementation to maintain parity in rollback tooling.
/// </summary>
public static class LiquidsConverter
{
    public static MclqData Mh2oToMclq(Mh2oChunk src, LiquidsOptions opts)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (opts is null) throw new ArgumentNullException(nameof(opts));

        var heights = new float[MclqData.VertexGrid * MclqData.VertexGrid];
        var depth = new byte[MclqData.VertexGrid * MclqData.VertexGrid];
        var types = new MclqLiquidType[MclqData.TileGrid, MclqData.TileGrid];
        var flags = new MclqTileFlags[MclqData.TileGrid, MclqData.TileGrid];

        if (!opts.EnableLiquids || src.IsEmpty)
            return new MclqData(heights, depth, types, flags);

        var chosen = new Mh2oInstance[MclqData.TileGrid, MclqData.TileGrid];
        for (int ty = 0; ty < MclqData.TileGrid; ty++)
        {
            for (int tx = 0; tx < MclqData.TileGrid; tx++)
            {
                int bestRank = int.MaxValue;
                Mh2oInstance? bestInst = null;
                MclqLiquidType bestType = MclqLiquidType.None;

                foreach (var inst in src.Instances)
                {
                    if (!InstanceCoversTile(inst, tx, ty))
                        continue;

                    ushort liquidTypeId = inst.LiquidTypeId;
                    if (opts.GreenLava && inst.LiquidTypeId == opts.Mapping.ToLiquidTypeId(MclqLiquidType.Magma))
                    {
                        liquidTypeId = opts.Mapping.ToLiquidTypeId(MclqLiquidType.Magma);
                    }

                    var mType = opts.Mapping.ToMclqType(liquidTypeId);
                    if (mType == MclqLiquidType.None)
                        continue;

                    int rank = GetPrecedenceRank(opts, mType);
                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestInst = inst;
                        bestType = mType;
                    }
                }

                types[tx, ty] = bestType;
                if (bestInst is not null)
                    chosen[tx, ty] = bestInst;
            }
        }

        if (src.Attributes is not null)
        {
            for (int ty = 0; ty < MclqData.TileGrid; ty++)
            {
                for (int tx = 0; tx < MclqData.TileGrid; tx++)
                {
                    if (IsBitSet(src.Attributes.FishableMask, tx, ty))
                        flags[tx, ty] |= MclqTileFlags.None;
                    if (IsBitSet(src.Attributes.DeepMask, tx, ty))
                        flags[tx, ty] |= MclqTileFlags.Fatigue;
                }
            }
        }

        for (int precedenceIndex = opts.Precedence.Count - 1; precedenceIndex >= 0; precedenceIndex--)
        {
            var type = opts.Precedence[precedenceIndex];
            for (int ty = 0; ty < MclqData.TileGrid; ty++)
            {
                for (int tx = 0; tx < MclqData.TileGrid; tx++)
                {
                    if (types[tx, ty] != type)
                        continue;

                    var inst = chosen[tx, ty];
                    if (inst is null)
                        continue;

                    bool hasH = HasHeightData(inst);
                    bool hasD = HasDepthData(inst);

                    for (int oy = 0; oy <= 1; oy++)
                    {
                        for (int ox = 0; ox <= 1; ox++)
                        {
                            int vx = tx + ox;
                            int vy = ty + oy;
                            int vertexIndex = VertexIndex(vx, vy);

                            int localVx = vx - inst.XOffset;
                            int localVy = vy - inst.YOffset;
                            int cols = inst.Width + 1;
                            int instIndex = localVy * cols + localVx;

                            if (hasH && inst.HeightMap is not null && instIndex < inst.HeightMap.Length)
                                heights[vertexIndex] = inst.HeightMap[instIndex];
                            if (hasD && inst.DepthMap is not null && instIndex < inst.DepthMap.Length)
                                depth[vertexIndex] = inst.DepthMap[instIndex];
                        }
                    }
                }
            }
        }

        return new MclqData(heights, depth, types, flags);
    }

    public static Mh2oChunk MclqToMh2o(MclqData src, LiquidsOptions opts)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (opts is null) throw new ArgumentNullException(nameof(opts));

        ulong fishableMask = 0;
        ulong deepMask = 0;

        for (int ty = 0; ty < MclqData.TileGrid; ty++)
        {
            for (int tx = 0; tx < MclqData.TileGrid; tx++)
            {
                int bit = ty * 8 + tx;
                if (src.Types[tx, ty] != MclqLiquidType.None)
                    fishableMask |= (1UL << bit);
                if ((src.Flags[tx, ty] & (MclqTileFlags.Fatigue | MclqTileFlags.ForcedSwim)) != 0)
                    deepMask |= (1UL << bit);
            }
        }

        var chunk = new Mh2oChunk
        {
            Attributes = fishableMask == 0 && deepMask == 0 ? null : new Mh2oAttributes(fishableMask, deepMask)
        };

        var visited = new bool[MclqData.TileGrid, MclqData.TileGrid];
        for (int ty = 0; ty < MclqData.TileGrid; ty++)
        {
            for (int tx = 0; tx < MclqData.TileGrid; tx++)
            {
                if (visited[tx, ty])
                    continue;

                var type = src.Types[tx, ty];
                if (type == MclqLiquidType.None)
                {
                    visited[tx, ty] = true;
                    continue;
                }

                int width = 0;
                while (tx + width < MclqData.TileGrid && !visited[tx + width, ty] && src.Types[tx + width, ty] == type)
                    width++;

                int height = 1;
                while (ty + height < MclqData.TileGrid)
                {
                    bool fullRow = true;
                    for (int dx = 0; dx < width; dx++)
                    {
                        if (visited[tx + dx, ty + height] || src.Types[tx + dx, ty + height] != type)
                        {
                            fullRow = false;
                            break;
                        }
                    }
                    if (!fullRow)
                        break;
                    height++;
                }

                for (int dy = 0; dy < height; dy++)
                {
                    for (int dx = 0; dx < width; dx++)
                    {
                        visited[tx + dx, ty + dy] = true;
                    }
                }

                int vCols = width + 1;
                int vRows = height + 1;
                int vCount = vCols * vRows;
                var subHeights = new float[vCount];
                var subDepths = new byte[vCount];
                bool hasHeights = false;
                float minHeight = float.PositiveInfinity;
                float maxHeight = float.NegativeInfinity;

                for (int vy = 0; vy < vRows; vy++)
                {
                    for (int vx = 0; vx < vCols; vx++)
                    {
                        int globalVx = tx + vx;
                        int globalVy = ty + vy;
                        int srcIndex = VertexIndex(globalVx, globalVy);
                        int dstIndex = vy * vCols + vx;
                        float h = src.Heights[srcIndex];
                        subHeights[dstIndex] = h;
                        subDepths[dstIndex] = src.Depth[srcIndex];
                        if (Math.Abs(h) > float.Epsilon)
                        {
                            hasHeights = true;
                            if (h < minHeight) minHeight = h;
                            if (h > maxHeight) maxHeight = h;
                        }
                    }
                }

                var instance = new Mh2oInstance
                {
                    LiquidTypeId = opts.Mapping.ToLiquidTypeId(type),
                    Lvf = hasHeights ? LiquidVertexFormat.HeightDepth : LiquidVertexFormat.DepthOnly,
                    MinHeightLevel = hasHeights ? minHeight : 0.0f,
                    MaxHeightLevel = hasHeights ? maxHeight : 0.0f,
                    XOffset = (byte)tx,
                    YOffset = (byte)ty,
                    Width = (byte)width,
                    Height = (byte)height,
                    ExistsBitmap = null,
                    HeightMap = hasHeights ? subHeights : null,
                    DepthMap = subDepths,
                };

                chunk.Add(instance);
            }
        }

        return chunk;
    }

    private static int GetPrecedenceRank(LiquidsOptions opts, MclqLiquidType type)
    {
        for (int i = 0; i < opts.Precedence.Count; i++)
        {
            if (opts.Precedence[i] == type)
                return i;
        }
        return int.MaxValue;
    }

    private static bool InstanceCoversTile(Mh2oInstance inst, int tileX, int tileY)
    {
        if (tileX < inst.XOffset || tileX >= inst.XOffset + inst.Width)
            return false;
        if (tileY < inst.YOffset || tileY >= inst.YOffset + inst.Height)
            return false;
        return InstanceTileExists(inst, tileX, tileY);
    }

    private static bool InstanceTileExists(Mh2oInstance inst, int tileX, int tileY)
    {
        if (inst.ExistsBitmap is null)
            return true;

        int localX = tileX - inst.XOffset;
        int localY = tileY - inst.YOffset;
        int bitIndex = localY * inst.Width + localX;
        int byteIndex = bitIndex / 8;
        int bit = bitIndex % 8;
        return (inst.ExistsBitmap[byteIndex] & (1 << bit)) != 0;
    }

    private static bool HasHeightData(Mh2oInstance inst)
        => inst.HeightMap is not null && inst.Lvf != LiquidVertexFormat.DepthOnly;

    private static bool HasDepthData(Mh2oInstance inst)
        => inst.DepthMap is not null;

    private static int VertexIndex(int vx, int vy)
        => vy * MclqData.VertexGrid + vx;

    private static bool IsBitSet(ulong mask, int x, int y)
        => ((mask >> (y * 8 + x)) & 1UL) != 0UL;
}
