// docs/AlphaWDTReader/snippets/mask_packing.cs
// Purpose: Pack/unpack MH2O layer masks (row-major, 1 bit per cell) and crop to bbox.

using System;

namespace Snippets
{
    public static class MaskPacking
    {
        // Packs a boolean grid within [minX..maxX], [minY..maxY] into a compact bitmask
        public static byte[] Pack(bool[,] occ, int minX, int minY, int maxX, int maxY)
        {
            int w = maxX - minX + 1;
            int h = maxY - minY + 1;
            int bits = w * h;
            int bytes = (bits + 7) >> 3;
            var dst = new byte[bytes];
            int bitIndex = 0;
            for (int y=minY; y<=maxY; y++)
            for (int x=minX; x<=maxX; x++)
            {
                if (occ[x,y]) dst[bitIndex >> 3] |= (byte)(1 << (bitIndex & 7));
                bitIndex++;
            }
            return dst;
        }

        public static bool[,] Unpack(ReadOnlySpan<byte> src, int w, int h)
        {
            var grid = new bool[w,h];
            int bits = w * h;
            for (int i=0; i<bits; i++)
            {
                bool v = (src[i >> 3] & (1 << (i & 7))) != 0;
                int x = i % w; int y = i / w;
                grid[x,y] = v;
            }
            return grid;
        }
    }
}
