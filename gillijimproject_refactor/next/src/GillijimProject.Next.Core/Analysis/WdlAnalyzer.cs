using System;
using System.Collections.Generic;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Analysis;

/// <summary>
/// Provides whiteplate detection and analysis for WDLs.
/// A "whiteplate" tile is defined heuristically as:
///  - All-zero heights (both grids) AND hole-mask all zeros; OR
///  - Matching the dominant tile signature (by frequency) across the map.
/// Callers may tune the baseline share threshold.
/// </summary>
public static class WdlAnalyzer
{
    public sealed record WhiteplateAnalysis(
        int PresentTiles,
        int ZeroTiles,
        ulong BaselineHash,
        int BaselineCount,
        bool BaselineIsZero,
        double BaselineShare
    );

    /// <summary>
    /// Analyze a WDL for whiteplate patterns and return statistics.
    /// </summary>
    public static WhiteplateAnalysis Analyze(Wdl wdl, double baselineMinShare = 0.66)
    {
        if (wdl is null) throw new ArgumentNullException(nameof(wdl));
        if (baselineMinShare < 0) baselineMinShare = 0; else if (baselineMinShare > 1) baselineMinShare = 1;

        int present = 0;
        int zeroTiles = 0;
        var freq = new Dictionary<ulong, int>();
        ulong zeroSig = 0;
        bool zeroSigSet = false;

        // Compute signatures for present tiles
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var t = wdl.Tiles[y, x];
                if (t is null) continue;
                present++;

                bool isZero = IsAllZero(t.Height17) && IsAllZero(t.Height16) && IsAllZero(t.HoleMask16);
                if (isZero)
                {
                    zeroTiles++;
                }

                ulong sig = ComputeSignature(t);
                if (isZero && !zeroSigSet)
                {
                    zeroSig = sig;
                    zeroSigSet = true;
                }

                if (!freq.TryGetValue(sig, out var c)) c = 0;
                freq[sig] = c + 1;
            }
        }

        if (present == 0)
        {
            return new WhiteplateAnalysis(0, 0, 0, 0, false, 0);
        }

        // Determine dominant baseline signature
        ulong baselineHash = 0;
        int baselineCount = 0;
        foreach (var kv in freq)
        {
            if (kv.Value > baselineCount)
            {
                baselineCount = kv.Value;
                baselineHash = kv.Key;
            }
        }
        double share = present > 0 ? (double)baselineCount / present : 0;
        bool baselineIsZero = zeroSigSet && baselineHash == zeroSig;

        // If the dominant share is below threshold, we still report it but callers may ignore it.
        return new WhiteplateAnalysis(
            PresentTiles: present,
            ZeroTiles: zeroTiles,
            BaselineHash: baselineHash,
            BaselineCount: baselineCount,
            BaselineIsZero: baselineIsZero,
            BaselineShare: share
        );
    }

    /// <summary>
    /// Compute a robust FNV-1a 64-bit signature over the tile's height grids and holes mask.
    /// Consistent across runs and platforms.
    /// </summary>
    public static ulong ComputeSignature(WdlTile tile)
    {
        const ulong FnvOffset = 14695981039346656037UL;
        const ulong FnvPrime = 1099511628211UL;
        ulong h = FnvOffset;

        // 17x17 int16
        for (int j = 0; j < WdlTile.OuterGrid; j++)
        {
            for (int i = 0; i < WdlTile.OuterGrid; i++)
            {
                unchecked
                {
                    ushort v = (ushort)tile.Height17[j, i];
                    byte lo = (byte)(v & 0xFF);
                    byte hi = (byte)(v >> 8);
                    h = (h ^ lo) * FnvPrime;
                    h = (h ^ hi) * FnvPrime;
                }
            }
        }

        // 16x16 int16
        for (int j = 0; j < WdlTile.InnerGrid; j++)
        {
            for (int i = 0; i < WdlTile.InnerGrid; i++)
            {
                unchecked
                {
                    ushort v = (ushort)tile.Height16[j, i];
                    byte lo = (byte)(v & 0xFF);
                    byte hi = (byte)(v >> 8);
                    h = (h ^ lo) * FnvPrime;
                    h = (h ^ hi) * FnvPrime;
                }
            }
        }

        // Holes mask rows (u16)
        for (int r = 0; r < WdlTile.InnerGrid; r++)
        {
            unchecked
            {
                ushort v = tile.HoleMask16[r];
                byte lo = (byte)(v & 0xFF);
                byte hi = (byte)(v >> 8);
                h = (h ^ lo) * FnvPrime;
                h = (h ^ hi) * FnvPrime;
            }
        }

        return h;
    }

    private static bool IsAllZero(short[,] grid)
    {
        int h = grid.GetLength(0), w = grid.GetLength(1);
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                if (grid[j, i] != 0) return false;
            }
        }
        return true;
    }

    private static bool IsAllZero(ushort[] rows)
    {
        for (int i = 0; i < rows.Length; i++) if (rows[i] != 0) return false;
        return true;
    }
}
