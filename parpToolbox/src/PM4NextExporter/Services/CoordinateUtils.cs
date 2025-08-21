using System;

namespace PM4NextExporter.Services
{
    /// <summary>
    /// Coordinate utilities to correlate PM4/ADT coordinate spaces.
    /// Non-invasive helpers for diagnostics and analysis.
    /// </summary>
    public static class CoordinateUtils
    {
        // ADT constants (see wowdev.wiki ADT_v18 docs)
        public const double BlockSize = 533.3333333333334; // yards per 64x64 block (1600 ft)
        public const double WorldExtent = 17066.666666666668; // half-width of map in yards (±extent)
        public const int BlocksPerAxis = 64;
        public const int BlockCenter = 32; // center block index

        /// <summary>
        /// Convert world axis value (X or Y) to ADT block index in [0..63].
        /// Formula per docs: floor(32 - axis/BlockSize).
        /// </summary>
        public static int WorldAxisToBlockIndex(double worldAxis)
        {
            return (int)Math.Floor(BlockCenter - (worldAxis / BlockSize));
        }

        /// <summary>
        /// Compute local offset inside the block (0..BlockSize) given a world axis.
        /// </summary>
        public static double WorldAxisToLocalInBlock(double worldAxis)
        {
            var n = BlockCenter - (worldAxis / BlockSize);
            var frac = n - Math.Floor(n);
            return frac * BlockSize; // range [0, BlockSize)
        }

        /// <summary>
        /// Reconstruct a world axis value from block index and local offset [0..BlockSize].
        /// </summary>
        public static double BlockAndLocalToWorldAxis(int blockIndex, double local)
        {
            return (BlockCenter - (blockIndex + (local / BlockSize))) * BlockSize;
        }

        /// <summary>
        /// Two simple server->world candidate transforms inferred from field observations.
        /// A: world = server + WorldExtent
        /// B: world = WorldExtent - server
        /// Use diagnostics to determine which yields valid block indices matching tile coordinates.
        /// </summary>
        public static (double worldA, double worldB) ServerAxisToWorldCandidates(double serverAxis)
        {
            double a = serverAxis + WorldExtent;
            double b = WorldExtent - serverAxis;
            return (a, b);
        }

        /// <summary>
        /// Clamp block index to [0..63]. Useful when computing derived indices for diagnostics.
        /// </summary>
        public static int ClampBlock(int idx) => Math.Min(BlocksPerAxis - 1, Math.Max(0, idx));
    }
}
