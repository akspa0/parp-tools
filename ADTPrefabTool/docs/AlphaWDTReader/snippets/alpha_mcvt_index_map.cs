// docs/AlphaWDTReader/snippets/alpha_mcvt_index_map.cs
// Purpose: Alpha (outer81+inner64) ↔ 3.x vertex index remap tables for 145 heights.
// Approach: Generate maps algorithmically from coordinate lists to avoid copy errors.
// Verify 3.x ordering against references before use.
// References: lib/gillijimproject/wowfiles/alpha/McnkAlpha.cpp, .../lichking/McnkLk.cpp

using System;
using System.Collections.Generic;

namespace Snippets
{
    public static class AlphaMcvtIndexMap
    {
        // Lazily-built mappings
        private static int[] _alphaTo3x;
        private static int[] _threeXToAlpha;

        public static int[] AlphaTo3x => _alphaTo3x ??= BuildAlphaTo3x();
        public static int[] ThreeXToAlpha => _threeXToAlpha ??= Invert(AlphaTo3x);

        // Alpha order: 9x9 OUTER (row-major), then 8x8 INNER (row-major)
        private static List<(int x,int y)> AlphaCoords()
        {
            var alpha = new List<(int,int)>(145);
            for (int y = 0; y < 9; y++)
                for (int x = 0; x < 9; x++)
                    alpha.Add((x, y)); // outer 81
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    alpha.Add((x, y)); // inner 64
            return alpha;
        }

        // 3.x order: interleaved diamond pattern.
        // This generator encodes the common 3.x expectation:
        // - Even rows: outer vertices (9) then inner vertices (8) at half-cell offsets
        // - Odd rows: inner then outer, interleaved
        // Represent inner samples with same 0..7 index space; mapping matches by (x,y) domain per group.
        private static List<(int x,int y,bool inner)> ThreeXCoords()
        {
            var list = new List<(int,int,bool)>(145);
            // There are 17 "rows" in diamond sampling: 9 outer rows and 8 inner rows interleaved.
            // We encode as pairs per strip: OuterRow(y), InnerRow(y) for y=0..7, then final OuterRow(8)
            for (int y = 0; y < 8; y++)
            {
                // Outer row y: 9 points (x=0..8)
                for (int x = 0; x < 9; x++) list.Add((x, y, false));
                // Inner row y: 8 points (x=0..7)
                for (int x = 0; x < 8; x++) list.Add((x, y, true));
            }
            // Last outer row y=8
            for (int x = 0; x < 9; x++) list.Add((x, 8, false));
            return list;
        }

        private static int[] BuildAlphaTo3x()
        {
            var alpha = AlphaCoords(); // 145 entries: outer81 then inner64
            var threeX = ThreeXCoords(); // 145 entries with inner flag

            // Build indices in Alpha space for outer and inner separately
            // Alpha outer range: indices 0..80 for (x:0..8, y:0..8)
            // Alpha inner range: indices 81..144 for (x:0..7, y:0..7)

            int IndexAlphaOuter(int x,int y) => y*9 + x; // 0..80
            int IndexAlphaInner(int x,int y) => 81 + y*8 + x; // 81..144

            var map = new int[145];
            for (int i = 0; i < 145; i++)
            {
                var (x, y, inner) = threeX[i];
                map[i] = inner ? IndexAlphaInner(x,y) : IndexAlphaOuter(x,y);
            }
            return map;
        }

        private static int[] Invert(int[] map)
        {
            var inv = new int[map.Length];
            for (int i = 0; i < map.Length; i++) inv[map[i]] = i;
            return inv;
        }

        // Optional quick validation using a synthetic pattern
        public static void Validate()
        {
            var a2b = AlphaTo3x; var b2a = ThreeXToAlpha;
            if (a2b.Length != 145 || b2a.Length != 145) throw new Exception("map size");
            for (int i=0;i<145;i++) if (b2a[a2b[i]]!=i) throw new Exception("invert failed at "+i);
        }
    }
}
