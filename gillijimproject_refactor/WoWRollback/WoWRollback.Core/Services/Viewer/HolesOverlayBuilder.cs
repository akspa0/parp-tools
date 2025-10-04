using System;
using System.Collections.Generic;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds holes overlay (terrain holes with 4×4 grid)
/// </summary>
public static class HolesOverlayBuilder
{
    public static object Build(List<McnkTerrainEntry> chunks, string version)
    {
        var holes = chunks
            .Where(c => c.HasHoles)
            .Select(c => new
            {
                row = c.ChunkRow,
                col = c.ChunkCol,
                type = c.HoleType,
                holes = DecodeHoleBitmap(c.HoleBitmapHex, c.HoleType)
            })
            .ToList();

        return new
        {
            version,
            holes
        };
    }

    /// <summary>
    /// Decode hole bitmap hex string into array of hole indices
    /// Alpha uses 4×4 grid (16 bits)
    /// </summary>
    private static List<int> DecodeHoleBitmap(string holeBitmapHex, string holeType)
    {
        if (string.IsNullOrEmpty(holeBitmapHex) || holeBitmapHex == "0x0000")
            return new List<int>();

        // Parse hex value
        var hex = holeBitmapHex.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? holeBitmapHex.Substring(2)
            : holeBitmapHex;

        if (!ushort.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out ushort bitmap))
            return new List<int>();

        var holeIndices = new List<int>();

        // For low-res holes, check 16 bits (4×4 grid)
        if (holeType == "low_res")
        {
            for (int i = 0; i < 16; i++)
            {
                if ((bitmap & (1 << i)) != 0)
                {
                    holeIndices.Add(i);
                }
            }
        }

        return holeIndices;
    }
}
