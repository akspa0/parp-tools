using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace PM4NextExporter.Services
{
    internal static class CrossTileVertexResolver
    {
        public static void Audit(string inputPath, string outDir)
        {
            // Determine directory and pattern
            var dir = Directory.Exists(inputPath)
                ? inputPath
                : Path.GetDirectoryName(inputPath) ?? ".";
            var pattern = "*.pm4"; // basic pattern; could extend to PD4

            Directory.CreateDirectory(outDir);
            var diagDir = Path.Combine(outDir, "diagnostics");
            Directory.CreateDirectory(diagDir);

            var ci = CultureInfo.InvariantCulture;
            var lines = new List<string> {
                "tile_x,tile_y,vertices,indices,oob_count,oob_min,oob_max,source"
            };

            ParpToolbox.Services.PM4.Pm4GlobalTileLoader.GlobalScene global;
            try
            {
                global = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.LoadRegion(dir, pattern);
            }
            catch (Exception ex)
            {
                File.WriteAllText(Path.Combine(diagDir, "audit_error.txt"), ex.ToString());
                return;
            }

            long totalVerts = 0;
            long totalIndices = 0;
            long totalOob = 0;
            int? globalMin = null;
            int? globalMax = null;

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var verts = tile.Scene.Vertices ?? new List<System.Numerics.Vector3>();
                var indices = tile.Scene.Indices ?? new List<int>();

                int oob = 0;
                int? minIdx = null;
                int? maxIdx = null;

                for (int i = 0; i < indices.Count; i++)
                {
                    var idx = indices[i];
                    if (idx < 0 || idx >= verts.Count)
                    {
                        oob++;
                        if (minIdx == null || idx < minIdx) minIdx = idx;
                        if (maxIdx == null || idx > maxIdx) maxIdx = idx;
                    }
                }

                totalVerts += verts.Count;
                totalIndices += indices.Count;
                totalOob += oob;
                if (minIdx != null) globalMin = globalMin == null ? minIdx : Math.Min(globalMin.Value, minIdx.Value);
                if (maxIdx != null) globalMax = globalMax == null ? maxIdx : Math.Max(globalMax.Value, maxIdx.Value);

                lines.Add(string.Join(",", new [] {
                    coord.X.ToString(ci),
                    coord.Y.ToString(ci),
                    verts.Count.ToString(ci),
                    indices.Count.ToString(ci),
                    oob.ToString(ci),
                    (minIdx?.ToString(ci) ?? ""),
                    (maxIdx?.ToString(ci) ?? ""),
                    EscapeCsv(Path.GetFileName(tile.SourceFile))
                }));
            }

            var summary = new List<string>{
                "metric,value",
                $"total_vertices,{totalVerts}",
                $"total_indices,{totalIndices}",
                $"total_oob,{totalOob}",
                $"global_oob_min,{(globalMin?.ToString(ci) ?? "")}",
                $"global_oob_max,{(globalMax?.ToString(ci) ?? "")}",
                $"oob_percent,{(totalIndices > 0 ? ((double)totalOob / totalIndices).ToString(ci) : "")}",
            };

            File.WriteAllLines(Path.Combine(diagDir, "cross_tile_audit.csv"), lines);
            File.WriteAllLines(Path.Combine(diagDir, "cross_tile_audit_summary.csv"), summary);
        }

        private static string EscapeCsv(string s)
        {
            if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
            {
                return '"' + s.Replace("\"", "\"\"") + '"';
            }
            return s;
        }
    }
}
