using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;

namespace PM4MscnAnalyzer
{
    public class MspiAnalyzeOptions
    {
        public string RootOutput { get; set; } = string.Empty;
        public string RegionDir { get; set; } = string.Empty;
        public string Pattern { get; set; } = "*.pm4";
        public bool WithMscn { get; set; } = true;
        public int ProximitySamples { get; set; } = 1000;
        public bool EmitTriSets { get; set; } = false;
        public int? MaxTiles { get; set; }
        public bool Verbose { get; set; } = false;
        public string? Session { get; set; }
    }

    public class MspiVsMsviAnalysis
    {
        public void Run(MspiAnalyzeOptions options, Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (global == null) throw new ArgumentNullException(nameof(global));
            if (scene == null) throw new ArgumentNullException(nameof(scene));

            var root = options.RootOutput;
            Directory.CreateDirectory(root);
            WriteCsvs(root);
            if (options.EmitTriSets) InitializeTriSets(root);

            // Analysis phases
            BuildTileStats(global, scene, options);
            BuildMslkCoverage(global, scene, options);
            CompareSurfacesVsMspi(global, scene, options);
            WriteCoverageSummary(global, scene, options);
            // if (options.WithMscn && options.ProximitySamples > 0) RunMscnProximity(global, scene, options);
            // DumpTriSetsJsonl(global, scene, options) if requested

            Console.WriteLine("[mspi-analyze] Initialized CSVs at: " + root);
        }

        private static string GetRegionName(string inputPath)
        {
            if (string.IsNullOrWhiteSpace(inputPath)) return string.Empty;
            try
            {
                // Normalize and get directory name even if path ends with a separator
                var di = new DirectoryInfo(inputPath);
                if (di.Exists) return di.Name;
                // If path points to a file, fall back to file name without extension
                var name = Path.GetFileName(inputPath);
                if (!string.IsNullOrEmpty(name)) return Path.GetFileNameWithoutExtension(name);
                // As a last resort, trim separators and take last segment
                var trimmed = inputPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                return Path.GetFileName(trimmed);
            }
            catch
            {
                return Path.GetFileName(inputPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            }
        }

        private void WriteCsvs(string root)
        {
            var tilesOverviewCsv = Path.Combine(root, "tiles_overview.csv");
            var mslkMspiCoverageCsv = Path.Combine(root, "mslk_mspi_coverage.csv");
            var msurVsMspiOverlapCsv = Path.Combine(root, "msur_vs_mspi_overlap.csv");
            var mscnProximityCsv = Path.Combine(root, "mscn_proximity_summary.csv");
            var indexOobCsv = Path.Combine(root, "index_oob.csv");
            var coverageSummaryCsv = Path.Combine(root, "coverage_summary.csv");

            using (var sw = new StreamWriter(tilesOverviewCsv))
            {
                sw.WriteLine("region,tile_x,tile_y,tile_id,mspi_index_width,mspi_index_count,mspi_tri_count,mspi_min_index,mspi_max_index,mspi_oob_index_count,msvi_index_count,msvi_tri_count,msvi_min_index,msvi_max_index,msvi_oob_index_count,msur_surface_count,mslk_entry_count,mscn_vertex_count,notes");
            }
            using (var sw = new StreamWriter(mslkMspiCoverageCsv))
            {
                sw.WriteLine("region,tile_id,link_id,parent_index_0x04,mspi_first_index,mspi_index_count,mspi_index_end,mspi_index_width,in_bounds,oob_index_count,tri_count,overlap_msvi_tri_count,overlap_ratio_vs_msvi,unique_mspi_tri_count,notes");
            }
            using (var sw = new StreamWriter(msurVsMspiOverlapCsv))
            {
                sw.WriteLine("region,tile_id,surface_id,group_key,msvi_first_index,msvi_index_count,msvi_tri_count,overlap_mspi_tri_count,overlap_ratio_vs_msvi,unique_msvi_tri_count,unique_mspi_tri_count,notes");
            }
            using (var sw = new StreamWriter(mscnProximityCsv))
            {
                sw.WriteLine("region,tile_id,samples_requested,samples_effective,mspi_wins_count,msvi_wins_count,ties_count,mean_dist_mspi,mean_dist_msvi,median_dist_mspi,median_dist_msvi,p90_dist_mspi,p90_dist_msvi,notes");
            }
            using (var sw = new StreamWriter(indexOobCsv))
            {
                sw.WriteLine("region,tile_id,source,ref_id,index_width,first_index,index_count,min_index,max_index,vertex_count,oob_index_count,oob_first_occurrence,notes");
            }
            using (var sw = new StreamWriter(coverageSummaryCsv))
            {
                sw.WriteLine("region,tiles,tiles_with_mspi,tiles_with_msvi,total_mspi_tris,total_msvi_tris,total_overlap_tris,overlap_ratio,mspi_unique_tris,msvi_unique_tris,notes");
            }
        }

        private void InitializeTriSets(string root)
        {
            var triSetsJsonl = Path.Combine(root, "tri_sets.jsonl");
            using var _ = File.Create(triSetsJsonl);
        }

        // Scaffolding methods (to be implemented in subsequent steps)
        private void BuildTileStats(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            var root = options.RootOutput;
            var tilesOverviewCsv = Path.Combine(root, "tiles_overview.csv");
            var indexOobCsv = Path.Combine(root, "index_oob.csv");
            var regionName = GetRegionName(options.RegionDir);

            int processed = 0;
            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                if (options.MaxTiles.HasValue && processed >= options.MaxTiles.Value) break;

                var coord = kvp.Key;
                var tile = kvp.Value.Scene;
                var tileId = kvp.Key.ToLinearIndex();

                var indices = tile.Indices ?? new List<int>();
                var vertices = tile.Vertices ?? new List<System.Numerics.Vector3>();
                var surfaces = tile.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
                var links = tile.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                var mscn = tile.MscnVertices ?? new List<System.Numerics.Vector3>();

                int vertexCount = vertices.Count;
                int indexCount = indices.Count;
                int triCount = indexCount / 3;
                int minIndex = indexCount > 0 ? indices.Min() : 0;
                int maxIndex = indexCount > 0 ? indices.Max() : 0;
                int indexWidth = maxIndex >= 65536 ? 32 : 16;

                // MSPI OOB scan (tile-local)
                int mspiOob = 0;
                int mspiFirstOobAt = -1;
                for (int i = 0; i < indexCount; i++)
                {
                    int v = indices[i];
                    if (v < 0 || v >= vertexCount)
                    {
                        mspiOob++;
                        if (mspiFirstOobAt == -1) mspiFirstOobAt = i;
                    }
                }

                // MSVI aggregated stats and OOB across surfaces
                long msviIndexCount = 0;
                long msviTriCount = 0;
                int msviMinIndex = int.MaxValue;
                int msviMaxIndex = int.MinValue;
                long msviOobTotal = 0;

                for (int si = 0; si < surfaces.Count; si++)
                {
                    var s = surfaces[si];
                    int first = (int)s.MsviFirstIndex;
                    int count = (int)s.IndexCount;
                    if (count <= 0) continue;

                    msviIndexCount += count;
                    msviTriCount += count / 3;

                    int endExclusive = first + count;
                    // Range clamp against indices buffer length
                    int rFirst = Math.Max(0, first);
                    int rEnd = Math.Min(indexCount, endExclusive);

                    int rangeMin = int.MaxValue;
                    int rangeMax = int.MinValue;
                    long oobInSurface = 0;
                    int firstOobAt = -1;

                    for (int i = rFirst; i < rEnd; i++)
                    {
                        int v = indices[i];
                        if (v < rangeMin) rangeMin = v;
                        if (v > rangeMax) rangeMax = v;
                        if (v < 0 || v >= vertexCount)
                        {
                            oobInSurface++;
                            if (firstOobAt == -1) firstOobAt = i - first; // offset within surface span
                        }
                    }

                    if (rangeMin != int.MaxValue) msviMinIndex = Math.Min(msviMinIndex, rangeMin);
                    if (rangeMax != int.MinValue) msviMaxIndex = Math.Max(msviMaxIndex, rangeMax);
                    msviOobTotal += oobInSurface;

                    // Emit per-surface OOB diagnostics (only when any OOB)
                    if (oobInSurface > 0)
                    {
                        using var oobSw = new StreamWriter(indexOobCsv, append: true);
                        oobSw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}",
                        GetRegionName(options.RegionDir), tileId, "MSVI", si,
                        indexWidth, first, count,
                        rangeMin == int.MaxValue ? 0 : rangeMin,
                        rangeMax == int.MinValue ? 0 : rangeMax,
                        vertexCount, oobInSurface, firstOobAt, ""));
                    }
                }

                if (msviMinIndex == int.MaxValue) msviMinIndex = 0;
                if (msviMaxIndex == int.MinValue) msviMaxIndex = 0;

                // Emit MSPI-level OOB diagnostics (if any)
                if (mspiOob > 0)
                {
                    using var oobSw = new StreamWriter(indexOobCsv, append: true);
                    oobSw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}",
                        regionName, tileId, "MSPI", -1,
                        indexWidth, 0, indexCount,
                        minIndex, maxIndex,
                        vertexCount, mspiOob, mspiFirstOobAt, ""));
                }

                // Emit tiles_overview row
                using (var sw = new StreamWriter(tilesOverviewCsv, append: true))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}",
                        regionName, coord.X, coord.Y, tileId,
                        indexWidth, indexCount, triCount, minIndex, maxIndex, mspiOob,
                        msviIndexCount, msviTriCount, msviMinIndex, msviMaxIndex, msviOobTotal,
                        surfaces.Count, links.Count, mscn.Count, ""));
                }

                processed++;
                if (options.Verbose && (processed % 50 == 0))
                {
                    Console.WriteLine($"[mspi-analyze] Processed {processed} tiles...");
                }
            }
        }

        private void BuildMslkCoverage(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            var root = options.RootOutput;
            var coverageCsv = Path.Combine(root, "mslk_mspi_coverage.csv");
            var indexOobCsv = Path.Combine(root, "index_oob.csv");
            var regionName = GetRegionName(options.RegionDir);

            int processed = 0;
            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                if (options.MaxTiles.HasValue && processed >= options.MaxTiles.Value) break;

                var coord = kvp.Key;
                var tile = kvp.Value.Scene;
                var tileId = kvp.Key.ToLinearIndex();

                var indices = tile.Indices ?? new List<int>();
                var vertices = tile.Vertices ?? new List<System.Numerics.Vector3>();
                var surfaces = tile.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
                var links = tile.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();

                int vertexCount = vertices.Count;
                int indexCount = indices.Count;
                int maxIndex = indexCount > 0 ? indices.Max() : 0;
                int indexWidth = maxIndex >= 65536 ? 32 : 16;

                // Build MSVI tri set for this tile
                var msviTriSet = new HashSet<(int, int, int)>();
                for (int si = 0; si < surfaces.Count; si++)
                {
                    var s = surfaces[si];
                    int first = (int)s.MsviFirstIndex;
                    int count = (int)s.IndexCount;
                    if (count <= 0) continue;
                    AddTrianglesFromRange(indices, first, count, vertexCount, msviTriSet, out _, out _);
                }

                // Per-MSLK coverage rows
                int linkProcessed = 0;
                foreach (var link in links)
                {
                    if (!link.HasGeometry) continue;
                    // Limit link to this tile based on embedded tile coords
                    if (!link.TryDecodeTileCoordinates(out int lx, out int ly) || lx != coord.X || ly != coord.Y)
                        continue;

                    int first = link.MspiFirstIndex;
                    int count = link.MspiIndexCount;
                    int endExclusive = first + count;

                    // Build MSPI tri set for this link
                    var linkTriSet = new HashSet<(int, int, int)>();
                    AddTrianglesFromRange(indices, first, count, vertexCount, linkTriSet, out int oobCount, out int firstOobAt);

                    int triCount = count / 3;
                    bool inBounds = oobCount == 0 && first >= 0 && endExclusive <= indexCount;
                    int overlap = IntersectCount(linkTriSet, msviTriSet);
                    double overlapRatio = msviTriSet.Count > 0 ? (double)overlap / msviTriSet.Count : 0.0;
                    int uniqueMspi = linkTriSet.Count - overlap;

                    using (var sw = new StreamWriter(coverageCsv, append: true))
                    {
                        sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}",
                            regionName, tileId, link.TileCoordinate, link.ParentIndex,
                            first, count, endExclusive, indexWidth,
                            inBounds ? "true" : "false",
                            oobCount,
                            triCount,
                            overlap,
                            overlapRatio.ToString("0.######", CultureInfo.InvariantCulture),
                            uniqueMspi,
                            ""));
                    }

                    // Emit per-link OOB diagnostics if any
                    if (oobCount > 0)
                    {
                        using var oobSw = new StreamWriter(indexOobCsv, append: true);
                        // min/max in link range for diagnostics
                        int rFirst = Math.Max(0, first);
                        int rEnd = Math.Min(indexCount, endExclusive);
                        int rMin = int.MaxValue, rMax = int.MinValue;
                        for (int i = rFirst; i < rEnd; i++)
                        {
                            int v = indices[i];
                            if (v < rMin) rMin = v;
                            if (v > rMax) rMax = v;
                        }
                        if (rMin == int.MaxValue) rMin = 0;
                        if (rMax == int.MinValue) rMax = 0;
                        oobSw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}",
                            regionName, tileId, "MSLK", link.TileCoordinate,
                            indexWidth, first, count, rMin, rMax,
                            vertexCount, oobCount, firstOobAt, ""));
                    }

                    linkProcessed++;
                }

                processed++;
                if (options.Verbose && (processed % 50 == 0))
                {
                    Console.WriteLine($"[mspi-analyze] MSLK coverage processed {processed} tiles...");
                }
            }
        }

        private void CompareSurfacesVsMspi(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            var root = options.RootOutput;
            var overlapCsv = Path.Combine(root, "msur_vs_mspi_overlap.csv");
            var regionName = GetRegionName(options.RegionDir);

            int processed = 0;
            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                if (options.MaxTiles.HasValue && processed >= options.MaxTiles.Value) break;

                var coord = kvp.Key;
                var tile = kvp.Value.Scene;
                var tileId = kvp.Key.ToLinearIndex();

                var indices = tile.Indices ?? new List<int>();
                var vertices = tile.Vertices ?? new List<System.Numerics.Vector3>();
                var surfaces = tile.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
                var links = tile.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();

                int vertexCount = vertices.Count;

                // Build tile-level MSPI tri set by union of current-tile links
                var mspiTileTriSet = new HashSet<(int, int, int)>();
                foreach (var link in links)
                {
                    if (!link.HasGeometry) continue;
                    if (!link.TryDecodeTileCoordinates(out int lx, out int ly) || lx != coord.X || ly != coord.Y)
                        continue;
                    AddTrianglesFromRange(indices, link.MspiFirstIndex, link.MspiIndexCount, vertexCount, mspiTileTriSet, out _, out _);
                }

                // Per-surface overlap rows
                for (int si = 0; si < surfaces.Count; si++)
                {
                    var s = surfaces[si];
                    int first = (int)s.MsviFirstIndex;
                    int count = (int)s.IndexCount;
                    if (count <= 0) continue;

                    var msviSurfaceTriSet = new HashSet<(int, int, int)>();
                    AddTrianglesFromRange(indices, first, count, vertexCount, msviSurfaceTriSet, out _, out _);

                    int msviTriCount = count / 3;
                    int overlap = IntersectCount(msviSurfaceTriSet, mspiTileTriSet);
                    double overlapRatio = msviTriCount > 0 ? (double)overlap / msviTriCount : 0.0;
                    int uniqueMsvi = msviSurfaceTriSet.Count - overlap;
                    int uniqueMspi = mspiTileTriSet.Count - overlap;

                    using var sw = new StreamWriter(overlapCsv, append: true);
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}",
                        regionName, tileId, si, s.SurfaceGroupKey,
                        first, count, msviTriCount,
                        overlap,
                        overlapRatio.ToString("0.######", CultureInfo.InvariantCulture),
                        uniqueMsvi,
                        uniqueMspi,
                        ""));
                }

                processed++;
                if (options.Verbose && (processed % 50 == 0))
                {
                    Console.WriteLine($"[mspi-analyze] MSUR vs MSPI processed {processed} tiles...");
                }
            }
        }

        private void RunMscnProximity(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            // TODO: Implement MSCN proximity sampling and emit mscn_proximity_summary.csv
        }

        private void DumpTriSetsJsonl(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            // TODO: Implement optional tri_sets.jsonl dumping per tile
        }

        // --- Helpers ---
        private static void AddTrianglesFromRange(List<int> indices, int first, int count, int vertexCount,
            HashSet<(int, int, int)> target, out int oobVertexCount, out int firstOobAt)
        {
            oobVertexCount = 0;
            firstOobAt = -1;
            if (indices == null || indices.Count == 0 || count <= 0) return;

            int start = Math.Max(0, first);
            int endExclusive = first + count;
            int end = Math.Min(indices.Count, endExclusive);
            // ensure tri grouping: trim to multiple of 3
            int span = end - start;
            span -= span % 3;
            end = start + span;
            for (int i = start; i + 2 < end; i += 3)
            {
                int a = indices[i];
                int b = indices[i + 1];
                int c = indices[i + 2];
                // OOB accounting against vertex count
                if (a < 0 || a >= vertexCount || b < 0 || b >= vertexCount || c < 0 || c >= vertexCount)
                {
                    oobVertexCount++;
                    if (firstOobAt == -1) firstOobAt = i - first; // offset into the provided range
                }
                // canonicalize tri key by sorting ascending
                if (a > b) (a, b) = (b, a);
                if (b > c) (b, c) = (c, b);
                if (a > b) (a, b) = (b, a);
                target.Add((a, b, c));
            }
        }

        private static int IntersectCount(HashSet<(int, int, int)> a, HashSet<(int, int, int)> b)
        {
            if (a.Count == 0 || b.Count == 0) return 0;
            int count = 0;
            // iterate smaller set
            var smaller = a.Count <= b.Count ? a : b;
            var larger = ReferenceEquals(smaller, a) ? b : a;
            foreach (var t in smaller) if (larger.Contains(t)) count++;
            return count;
        }

        private void WriteCoverageSummary(Pm4GlobalTileLoader.GlobalScene global, Pm4Scene scene, MspiAnalyzeOptions options)
        {
            var root = options.RootOutput;
            var summaryCsv = Path.Combine(root, "coverage_summary.csv");
            var regionName = GetRegionName(options.RegionDir);

            int tiles = 0, tilesWithMspi = 0, tilesWithMsvi = 0;
            long totalMspiTris = 0, totalMsviTris = 0, totalOverlapTris = 0;
            long mspiUnique = 0, msviUnique = 0;

            int processed = 0;
            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                if (options.MaxTiles.HasValue && processed >= options.MaxTiles.Value) break;

                var coord = kvp.Key;
                var tile = kvp.Value.Scene;

                var indices = tile.Indices ?? new List<int>();
                var vertices = tile.Vertices ?? new List<System.Numerics.Vector3>();
                var surfaces = tile.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
                var links = tile.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();

                int vertexCount = vertices.Count;

                var msviTileTriSet = new HashSet<(int, int, int)>();
                foreach (var s in surfaces)
                {
                    int first = (int)s.MsviFirstIndex;
                    int count = (int)s.IndexCount;
                    if (count <= 0) continue;
                    AddTrianglesFromRange(indices, first, count, vertexCount, msviTileTriSet, out _, out _);
                }

                var mspiTileTriSet = new HashSet<(int, int, int)>();
                foreach (var link in links)
                {
                    if (!link.HasGeometry) continue;
                    if (!link.TryDecodeTileCoordinates(out int lx, out int ly) || lx != coord.X || ly != coord.Y)
                        continue;
                    AddTrianglesFromRange(indices, link.MspiFirstIndex, link.MspiIndexCount, vertexCount, mspiTileTriSet, out _, out _);
                }

                tiles++;
                if (mspiTileTriSet.Count > 0) tilesWithMspi++;
                if (msviTileTriSet.Count > 0) tilesWithMsvi++;

                int overlap = IntersectCount(msviTileTriSet, mspiTileTriSet);
                totalOverlapTris += overlap;
                totalMspiTris += mspiTileTriSet.Count;
                totalMsviTris += msviTileTriSet.Count;
                mspiUnique += Math.Max(0, mspiTileTriSet.Count - overlap);
                msviUnique += Math.Max(0, msviTileTriSet.Count - overlap);

                processed++;
            }

            double overlapRatio = totalMsviTris > 0 ? (double)totalOverlapTris / totalMsviTris : 0.0;
            using var sw = new StreamWriter(summaryCsv, append: true);
            sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}",
                regionName,
                tiles, tilesWithMspi, tilesWithMsvi,
                totalMspiTris, totalMsviTris, totalOverlapTris,
                overlapRatio.ToString("0.######", CultureInfo.InvariantCulture),
                mspiUnique, msviUnique,
                ""));
        }
    }
}
