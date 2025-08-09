using System.IO;

namespace PM4NextExporter.Services
{
    public static class DiagnosticsService
    {
        /// <summary>
        /// Writes flattened MSUR surface data to <paramref name="outDir"/>/surfaces.csv for offline analysis.
        /// </summary>
        internal static void WriteSurfaceCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;

            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "surfaces.csv");

            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            // Header
            writer.WriteLine("index,compositeKey,byteAA,byteBB,byteCC,byteDD,surfaceKeyHigh16,msviFirstIndex,indexCount,groupKey");

            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var s = scene.Surfaces[i];
                uint key = s.CompositeKey;
                byte aa = (byte)(key >> 24);
                byte bb = (byte)(key >> 16);
                byte cc = (byte)(key >> 8);
                byte dd = (byte)(key);
                writer.WriteLine(string.Join(',',
                    i,
                    $"0x{key:X8}",
                    aa,
                    bb,
                    cc,
                    dd,
                    $"0x{s.SurfaceKeyHigh16:X4}",
                    s.MsviFirstIndex,
                    s.IndexCount,
                    s.SurfaceGroupKey));
            }
        }

        // Legacy stub for snapshot; kept for compatibility
        internal static void WriteSnapshotCsv(string outDir, string name)
        {
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, $"{name}.csv");
            File.WriteAllText(path, "metric,value\nstub,1\n");
        }
        /// <summary>
        /// Writes aggregated statistics per CompositeKey to <paramref name="outDir"/>/surface_summary.csv.
        /// No data is filtered – every MSUR entry contributes to its key’s totals.
        /// Columns: compositeKey,byteAA,byteBB,byteCC,byteDD,rowCount,totalIndexCount,uniqueGroupKeys
        /// </summary>
        internal static void WriteCompositeSummaryCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;

            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "surface_summary.csv");

            // Aggregate
            var stats = new Dictionary<uint, (int rowCount, int indexSum, HashSet<byte> groups)>();
            foreach (var s in scene.Surfaces)
            {
                if (!stats.TryGetValue(s.CompositeKey, out var tuple))
                    tuple = (0, 0, new HashSet<byte>());
                tuple.rowCount++;
                tuple.indexSum += s.IndexCount;
                tuple.groups.Add(s.SurfaceGroupKey);
                stats[s.CompositeKey] = tuple;
            }

            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("compositeKey,byteAA,byteBB,byteCC,byteDD,rowCount,totalIndexCount,uniqueGroupKeys");
            foreach (var kvp in stats)
            {
                uint key = kvp.Key;
                byte aa = (byte)(key >> 24);
                byte bb = (byte)(key >> 16);
                byte cc = (byte)(key >> 8);
                byte dd = (byte)(key);
                var (cnt, sum, groups) = kvp.Value;
                writer.WriteLine(string.Join(',',
                    $"0x{key:X8}", aa, bb, cc, dd, cnt, sum, groups.Count));
            }
        }

        /// <summary>
        /// Dumps all MSCN exterior vertices to <paramref name="outDir"/>/mscn_vertices.csv for analysis.
        /// Columns: index,x,y,z,xCanonical,yCanonical,zCanonical
        /// </summary>
        internal static void WriteMscnCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.MscnVertices == null || scene.MscnVertices.Count == 0)
                return;
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "mscn_vertices.csv");
            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("index,x,y,z,xCanonical,yCanonical,zCanonical");
            for (int i = 0; i < scene.MscnVertices.Count; i++)
            {
                var v = scene.MscnVertices[i];
                var world = new System.Numerics.Vector3(v.Y, -v.X, v.Z);
                writer.WriteLine(string.Join(',', i, v.X, v.Y, v.Z, world.X, world.Y, world.Z));
            }
        }
    }
}
