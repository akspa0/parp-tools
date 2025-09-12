using System;
using System.Collections.Generic;
using GillijimProject.Next.Core.Domain.Liquids;

namespace GillijimProject.Next.Core.Transform.Liquids;

/// <summary>
/// Bidirectional converter for MH2O ↔ MCLQ liquid data.
/// This file provides compile-ready stubs; implementations will follow.
/// </summary>
public static class LiquidsConverter
{
    /// <summary>
    /// Converts an MH2O chunk (LK) to an MCLQ payload (Alpha) for a single MCNK.
    /// </summary>
    /// <exception cref="NotImplementedException">Implementation pending.</exception>
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

        // Choose per-tile instance based on precedence
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
                    if (!InstanceCoversTile(inst, tx, ty)) continue;
                    var mType = opts.Mapping.ToMclqType(inst.LiquidTypeId);
                    if (mType == MclqLiquidType.None) continue;
                    int rank = GetPrecedenceRank(opts, mType);
                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestInst = inst;
                        bestType = mType;
                    }
                }

                types[tx, ty] = bestType;
                if (bestInst is not null) chosen[tx, ty] = bestInst;
            }
        }

        // Map attributes → flags
        if (src.Attributes is not null)
        {
            for (int ty = 0; ty < MclqData.TileGrid; ty++)
            {
                for (int tx = 0; tx < MclqData.TileGrid; tx++)
                {
                    if (IsBitSet(src.Attributes.DeepMask, tx, ty))
                    {
                        flags[tx, ty] |= MclqTileFlags.Fatigue; // map deep → fatigue
                    }
                }
            }
        }

        // Fill vertex grids: write lower precedence first, then higher precedence overrides
        for (int p = opts.Precedence.Count - 1; p >= 0; p--)
        {
            var t = opts.Precedence[p];
            for (int ty = 0; ty < MclqData.TileGrid; ty++)
            {
                for (int tx = 0; tx < MclqData.TileGrid; tx++)
                {
                    if (types[tx, ty] != t) continue;
                    var inst = chosen[tx, ty];
                    if (inst is null) continue;

                    bool hasH = HasHeightData(inst);
                    bool hasD = HasDepthData(inst);

                    // Four vertices of tile (tx,ty)
                    for (int oy = 0; oy <= 1; oy++)
                    {
                        for (int ox = 0; ox <= 1; ox++)
                        {
                            int vx = tx + ox;
                            int vy = ty + oy;
                            int vIndex = VertexIndex(vx, vy);

                            int localVx = vx - inst.XOffset;
                            int localVy = vy - inst.YOffset;
                            int instCols = inst.Width + 1;
                            int instIndex = localVy * instCols + localVx;

                            if (hasH)
                            {
                                // Only overwrite if this instance provides height data
                                heights[vIndex] = inst.HeightMap![instIndex];
                            }
                            if (hasD)
                            {
                                depth[vIndex] = inst.DepthMap![instIndex];
                            }
                        }
                    }
                }
            }
        }

        return new MclqData(heights, depth, types, flags);
    }

    /// <summary>
    /// Converts an MCLQ payload (Alpha) to an MH2O chunk (LK) for a single MCNK.
    /// </summary>
    /// <exception cref="NotImplementedException">Implementation pending.</exception>
    public static Mh2oChunk MclqToMh2o(MclqData src, LiquidsOptions opts)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (opts is null) throw new ArgumentNullException(nameof(opts));

        // Derive attributes first
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

        var chunk = new Mh2oChunk { Attributes = new Mh2oAttributes(fishableMask, deepMask) };

        // Group tiles into rectangles by type
        var visited = new bool[MclqData.TileGrid, MclqData.TileGrid];
        for (int ty = 0; ty < MclqData.TileGrid; ty++)
        {
            for (int tx = 0; tx < MclqData.TileGrid; tx++)
            {
                if (visited[tx, ty]) continue;
                var type = src.Types[tx, ty];
                if (type == MclqLiquidType.None) { visited[tx, ty] = true; continue; }

                // Determine maximal rectangle starting at (tx,ty)
                int w = 0;
                while (tx + w < MclqData.TileGrid && !visited[tx + w, ty] && src.Types[tx + w, ty] == type) w++;

                int h = 1;
                while (ty + h < MclqData.TileGrid)
                {
                    bool fullRow = true;
                    for (int dx = 0; dx < w; dx++)
                    {
                        if (visited[tx + dx, ty + h] || src.Types[tx + dx, ty + h] != type)
                        {
                            fullRow = false; break;
                        }
                    }
                    if (!fullRow) break;
                    h++;
                }

                // Mark visited
                for (int dy = 0; dy < h; dy++)
                    for (int dx = 0; dx < w; dx++)
                        visited[tx + dx, ty + dy] = true;

                // Slice vertex data
                int vCols = w + 1;
                int vRows = h + 1;
                int vCount = vCols * vRows;
                var subH = new float[vCount];
                var subD = new byte[vCount];
                bool hasHeights = false;
                float minH = float.PositiveInfinity, maxH = float.NegativeInfinity;

                for (int vy = 0; vy < vRows; vy++)
                {
                    for (int vx = 0; vx < vCols; vx++)
                    {
                        int globalVx = tx + vx;
                        int globalVy = ty + vy;
                        int srcIdx = VertexIndex(globalVx, globalVy);
                        int dstIdx = vy * vCols + vx;
                        float hv = src.Heights[srcIdx];
                        subH[dstIdx] = hv;
                        subD[dstIdx] = src.Depth[srcIdx];
                        if (hv != 0.0f) { hasHeights = true; if (hv < minH) minH = hv; if (hv > maxH) maxH = hv; }
                    }
                }

                var inst = new Mh2oInstance
                {
                    LiquidTypeId = opts.Mapping.ToLiquidTypeId(type),
                    Lvf = hasHeights ? LiquidVertexFormat.HeightDepth : LiquidVertexFormat.DepthOnly,
                    MinHeightLevel = hasHeights ? minH : 0.0f,
                    MaxHeightLevel = hasHeights ? maxH : 0.0f,
                    XOffset = (byte)tx,
                    YOffset = (byte)ty,
                    Width = (byte)w,
                    Height = (byte)h,
                    ExistsBitmap = null,
                    HeightMap = hasHeights ? subH : null,
                    DepthMap = subD,
                };

                chunk.Add(inst);
            }
        }

        return chunk;
    }

    // Helpers
    private static int GetPrecedenceRank(LiquidsOptions opts, MclqLiquidType t)
    {
        for (int i = 0; i < opts.Precedence.Count; i++)
            if (opts.Precedence[i] == t) return i;
        return int.MaxValue; // lowest precedence if not found
    }

    private static bool InstanceCoversTile(Mh2oInstance inst, int tileX, int tileY)
    {
        if (tileX < inst.XOffset || tileX >= inst.XOffset + inst.Width) return false;
        if (tileY < inst.YOffset || tileY >= inst.YOffset + inst.Height) return false;
        return InstanceTileExists(inst, tileX, tileY);
    }

    private static bool InstanceTileExists(Mh2oInstance inst, int tileX, int tileY)
    {
        if (inst.ExistsBitmap is null) return true;
        int localX = tileX - inst.XOffset;
        int localY = tileY - inst.YOffset;
        int bitIndex = localY * inst.Width + localX;
        int byteIndex = bitIndex / 8;
        int bit = bitIndex % 8;
        return (inst.ExistsBitmap[byteIndex] & (1 << bit)) != 0;
    }

    private static bool HasHeightData(Mh2oInstance inst) => inst.HeightMap is not null && inst.Lvf != LiquidVertexFormat.DepthOnly;
    private static bool HasDepthData(Mh2oInstance inst) => inst.DepthMap is not null;

    private static int VertexIndex(int vx, int vy) => vy * MclqData.VertexGrid + vx;

    private static bool IsBitSet(ulong mask, int x, int y)
        => ((mask >> (y * 8 + x)) & 1UL) != 0UL;
}
