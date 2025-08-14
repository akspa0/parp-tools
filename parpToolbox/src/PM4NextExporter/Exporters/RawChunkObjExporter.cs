using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Exporters
{
    internal static class RawChunkObjExporter
    {
        // Zero-transform OBJ writer for debugging: per-tile, per full 32-bit CompositeKey
        public static void Export(Scene scene, string outDir, bool legacyObjParity)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;

            var tileOffsets = scene.TileIndexOffsetByTileId ?? new Dictionary<int,int>();
            var tileCounts  = scene.TileIndexCountByTileId ?? new Dictionary<int,int>();
            var tileCoords  = scene.TileCoordByTileId ?? new Dictionary<int, ParpToolbox.Formats.PM4.TileCoord>();

            // Group surfaces by exact tile using [start, start+count)
            int? TileForFirstIndex(int first)
            {
                foreach (var kv in tileOffsets)
                {
                    var tid = kv.Key;
                    var start = kv.Value;
                    if (!tileCounts.TryGetValue(tid, out var cnt)) continue;
                    if (first >= start && first < start + cnt) return tid;
                }
                return null;
            }

            var byTile = scene.Surfaces.GroupBy(s =>
            {
                if (s.MsviFirstIndex > int.MaxValue) return (int?)null;
                int first = unchecked((int)s.MsviFirstIndex);
                return TileForFirstIndex(first);
            })
            .OrderBy(g => g.Key.HasValue ? 0 : 1)
            .ThenBy(g => g.Key);

            var root = Path.Combine(outDir, "debug_raw");
            Directory.CreateDirectory(root);

            foreach (var tileGroup in byTile)
            {
                var tid = tileGroup.Key;
                string tileFolderName = tid.HasValue && tileCoords.TryGetValue(tid.Value, out var coord)
                    ? $"tile_{coord.X:00}_{coord.Y:00}"
                    : tid.HasValue ? $"tile_{tid.Value:0000}" : "tile_unknown";
                var tileDir = Path.Combine(root, tileFolderName);
                Directory.CreateDirectory(tileDir);

                // Group by full CompositeKey
                var byCk = tileGroup.GroupBy(s => s.CompositeKey).OrderBy(g => g.Key);
                bool invertX = !legacyObjParity; // default behavior: flip X unless legacy parity requested
                foreach (var ckGroup in byCk)
                {
                    string file = Path.Combine(tileDir, $"ck_{ckGroup.Key:X8}.obj");
                    using var sw = new StreamWriter(file);
                    sw.WriteLine("# raw debug OBJ (no transforms)");
                    sw.WriteLine("o ck_{0:X8}", ckGroup.Key);

                    // Collect local vertices and faces for this group
                    var lVerts = new List<Vector3>();
                    var map = new Dictionary<int,int>();
                    var lTris = new List<(int,int,int)>();

                    foreach (var surf in ckGroup)
                    {
                        if (surf.MsviFirstIndex > int.MaxValue) continue;
                        int first = unchecked((int)surf.MsviFirstIndex);
                        int count = surf.IndexCount;
                        if (first < 0 || count < 3 || first + count > scene.Indices.Count) continue;
                        int triCnt = count / 3;
                        for (int t = 0; t < triCnt; t++)
                        {
                            int baseIdx = first + t*3;
                            int a = scene.Indices[baseIdx];
                            int b = scene.Indices[baseIdx+1];
                            int c = scene.Indices[baseIdx+2];
                            if (a<0||b<0||c<0||a>=scene.Vertices.Count||b>=scene.Vertices.Count||c>=scene.Vertices.Count) continue;
                            int la = Map(a, scene.Vertices, lVerts, map);
                            int lb = Map(b, scene.Vertices, lVerts, map);
                            int lc = Map(c, scene.Vertices, lVerts, map);
                            lTris.Add((la,lb,lc));
                        }
                    }

                    // Write vertices (apply optional X inversion)
                    foreach (var v in lVerts)
                    {
                        float vx = invertX ? -v.X : v.X;
                        sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "v {0} {1} {2}", vx, v.Y, v.Z));
                    }
                    // Write faces (1-based)
                    foreach (var tri in lTris)
                    {
                        if (invertX)
                        {
                            // Swap winding to preserve outward normals when mirroring on X
                            sw.WriteLine($"f {tri.Item1+1} {tri.Item3+1} {tri.Item2+1}");
                        }
                        else
                        {
                            sw.WriteLine($"f {tri.Item1+1} {tri.Item2+1} {tri.Item3+1}");
                        }
                    }
                }
            }
        }

        private static int Map(int gIdx, List<Vector3> gVerts, List<Vector3> lVerts, Dictionary<int,int> map)
        {
            if (map.TryGetValue(gIdx, out var li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIdx]);
            map[gIdx] = li;
            return li;
        }
    }
}
