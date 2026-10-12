using System;
using System.Buffers.Binary;

namespace WowViewer.Core.Maps;

/// <summary>
/// Bitmath and coordinate translation for terrain holes across all WoW client eras:
/// 16-bit legacy/low-res hole masks (4×4 groups of 2×2 cells) and 64-bit modern high_res_holes (8×8 cells).
/// </summary>
public static class TerrainHoleMath
{
    public const int CellsPerAxis = 8;
    public const int HoleGroupsPerAxis = 4;
    public const uint HighResHolesFlag = 0x10000u;

    /// <summary>
    /// Test if cell (cellX, cellY) is holed in a 64-bit hole mask.
    /// Bit is 1 for holed, 0 for intact. Row = cellY (0..7), Column = cellX (0..7).
    /// </summary>
    public static bool IsCellHoled64(ulong holeMask64, int cellX, int cellY)
    {
        if (holeMask64 == 0UL || (uint)cellX >= CellsPerAxis || (uint)cellY >= CellsPerAxis)
            return false;
        return ((holeMask64 >> (cellY * CellsPerAxis + cellX)) & 1UL) != 0UL;
    }

    /// <summary>
    /// Test if cell (cellX, cellY) is holed in a 16-bit legacy low-res hole mask (4×4 groups of 2×2 cells).
    /// </summary>
    public static bool IsCellHoled16(ushort holeMask16, int cellX, int cellY)
    {
        if (holeMask16 == 0 || (uint)cellX >= CellsPerAxis || (uint)cellY >= CellsPerAxis)
            return false;
        int holeGroupX = cellX / 2;
        int holeGroupY = cellY / 2;
        int bit = 1 << (holeGroupY * HoleGroupsPerAxis + holeGroupX);
        return (holeMask16 & bit) != 0;
    }

    /// <summary>
    /// Upsample a 16-bit low-res mask (4×4 2×2 cell groups) to a full 64-bit mask (8×8 1×1 cells).
    /// </summary>
    public static ulong UpsampleLowResToHighRes(ushort lowRes)
    {
        if (lowRes == 0) return 0UL;
        ulong highRes = 0UL;
        for (int hy = 0; hy < HoleGroupsPerAxis; hy++)
        {
            for (int hx = 0; hx < HoleGroupsPerAxis; hx++)
            {
                if ((lowRes & (1 << (hy * HoleGroupsPerAxis + hx))) != 0)
                {
                    int cy = hy * 2;
                    int cx = hx * 2;
                    highRes |= 1UL << (cy * CellsPerAxis + cx);
                    highRes |= 1UL << (cy * CellsPerAxis + cx + 1);
                    highRes |= 1UL << ((cy + 1) * CellsPerAxis + cx);
                    highRes |= 1UL << ((cy + 1) * CellsPerAxis + cx + 1);
                }
            }
        }
        return highRes;
    }

    /// <summary>
    /// Downsample a 64-bit mask (8×8 cells) to a 16-bit mask (4×4 groups).
    /// A 2×2 group is marked holed if ANY of its 4 subcells is holed.
    /// </summary>
    public static ushort DownsampleHighResToLowRes(ulong highRes)
    {
        if (highRes == 0UL) return 0;
        ushort lowRes = 0;
        for (int hy = 0; hy < HoleGroupsPerAxis; hy++)
        {
            for (int hx = 0; hx < HoleGroupsPerAxis; hx++)
            {
                int cy = hy * 2;
                int cx = hx * 2;
                if (((highRes >> (cy * CellsPerAxis + cx)) & 1UL) != 0UL ||
                    ((highRes >> (cy * CellsPerAxis + cx + 1)) & 1UL) != 0UL ||
                    ((highRes >> ((cy + 1) * CellsPerAxis + cx)) & 1UL) != 0UL ||
                    ((highRes >> ((cy + 1) * CellsPerAxis + cx + 1)) & 1UL) != 0UL)
                {
                    lowRes |= (ushort)(1 << (hy * HoleGroupsPerAxis + hx));
                }
            }
        }
        return lowRes;
    }

    /// <summary>
    /// Read the hole masks from an MCNK header and payload.
    /// Supports modern 5.3+ high_res_holes (flag 0x10000, 8 bytes at offset 0x14) and legacy 16-bit holes (offset 0x3C).
    /// Returns both full 64-bit mask and 16-bit compatible representation.
    /// </summary>
    public static (ulong HoleMask64, ushort HoleMask16) ReadHoleMasks(ReadOnlySpan<byte> mcnkHeaderOrPayload, uint mcnkFlags)
    {
        if (mcnkHeaderOrPayload.Length < 0x3E)
            return (0UL, 0);

        bool hasHighRes = (mcnkFlags & HighResHolesFlag) != 0;
        if (hasHighRes && mcnkHeaderOrPayload.Length >= 0x1C)
        {
            ulong mask64 = BinaryPrimitives.ReadUInt64LittleEndian(mcnkHeaderOrPayload.Slice(0x14, 8));
            ushort mask16 = DownsampleHighResToLowRes(mask64);
            return (mask64, mask16);
        }

        ushort lowRes = BinaryPrimitives.ReadUInt16LittleEndian(mcnkHeaderOrPayload.Slice(0x3C, 2));
        ulong highRes = UpsampleLowResToHighRes(lowRes);
        return (highRes, lowRes);
    }
}
