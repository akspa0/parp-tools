using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using ParpToolbox.Services.PM4;

namespace PM4MscnAnalyzer
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            if (args.Length == 0 || IsHelp(args))
            {
                PrintHelp();
                return 0;
            }

            var command = args[0].ToLowerInvariant();
            var options = ParseOptions(args.Skip(1).ToArray());

            try
            {
                return command switch
                {
                    "mscn-dump" => RunMscnDump(options),
                    "object-centroids" => RunObjectCentroids(options),
                    "mprr-dump" => RunMprrDump(options),
                    "mscn-associate" => RunMscnAssociate(options),
                    _ => UnknownCommand(command)
                };
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"ERROR: {ex.Message}");
                Console.Error.WriteLine(ex.StackTrace);
                return 1;
            }
        }

        private static int UnknownCommand(string command)
        {
            Console.Error.WriteLine($"Unknown command: {command}\n");
            PrintHelp();
            return 2;
        }

        private static bool IsHelp(string[] args)
        {
            return args.Contains("-h") || args.Contains("--help") || args.Contains("help");
        }

        private static void PrintHelp()
        {
            Console.WriteLine("PM4 MSCN Analyzer");
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  PM4MscnAnalyzer mscn-dump --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer object-centroids --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer mprr-dump --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer mscn-associate --session <name> [--out <path>] [--k <int>] [--max-dist <float>]");
            Console.WriteLine();
            Console.WriteLine("Commands:");
            Console.WriteLine("  mscn-dump        Export MSCN points with tile provenance to CSV");
            Console.WriteLine("  object-centroids Compute per-tile object centroids and stats to CSV");
            Console.WriteLine("  mprr-dump        Export raw MPRR properties with tile provenance to CSV");
            Console.WriteLine("  mscn-associate   kNN from object centroids to MSCN points, emits associations.csv");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --input <dir>    Directory containing *.pm4 region tiles (required)");
            Console.WriteLine("  --pattern <glob> File search pattern, default: *.pm4");
            Console.WriteLine("  --out <path>     Output directory (default: project_output/mscn_analysis_<ts>)");
            Console.WriteLine("  --session <name> Session name to nest under output (appends timestamp)");
            Console.WriteLine("  --k <int>        Number of nearest MSCN to return (default 1)");
            Console.WriteLine("  --max-dist <f>   Optional distance cutoff in world units");
        }

        private static Dictionary<string, string?> ParseOptions(string[] args)
        {
            var dict = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
            string? current = null;
            foreach (var tok in args)
            {
                if (tok.StartsWith("--"))
                {
                    current = tok.Substring(2);
                    dict[current] = null; // may be a flag or key expecting value
                }
                else if (current != null)
                {
                    dict[current] = tok;
                    current = null;
                }
            }
            return dict;
        }

        private static int RunMscnDump(Dictionary<string, string?> opt)
        {
            if (!opt.TryGetValue("input", out var inputDir) || string.IsNullOrWhiteSpace(inputDir))
            {
                Console.Error.WriteLine("--input <region_dir> is required");
                return 2;
            }
            if (!Directory.Exists(inputDir))
            {
                Console.Error.WriteLine($"Input directory not found: {inputDir}");
                return 2;
            }

            opt.TryGetValue("pattern", out var pattern);
            if (string.IsNullOrWhiteSpace(pattern)) pattern = "*.pm4";
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var pointsDir = OutputLocator.EnsureSubfolder(root, "points");
            var csvPath = Path.Combine(pointsDir, "mscn_points.csv");

            Console.WriteLine($"[mscn-dump] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);
            var scene = Pm4GlobalTileLoader.ToStandardScene(global);

            var mscn = scene.MscnVertices ?? new List<System.Numerics.Vector3>();
            var mscnTileIds = scene.MscnVertexTileIds ?? new List<int>();
            var count = Math.Min(mscn.Count, mscnTileIds.Count);

            Directory.CreateDirectory(pointsDir);
            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("tile_id,tile_x,tile_y,x,y,z");

            for (int i = 0; i < count; i++)
            {
                var v = mscn[i];
                var tileId = mscnTileIds[i];
                var coord = Pm4GlobalTileLoader.TileCoordinate.FromLinearIndex(tileId);
                sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0},{1},{2},{3:F6},{4:F6},{5:F6}",
                    tileId, coord.X, coord.Y, v.X, v.Y, v.Z));
            }

            Console.WriteLine($"[mscn-dump] Wrote {count:N0} points → {csvPath}");
            return 0;
        }

        private static int RunObjectCentroids(Dictionary<string, string?> opt)
        {
            if (!opt.TryGetValue("input", out var inputDir) || string.IsNullOrWhiteSpace(inputDir))
            {
                Console.Error.WriteLine("--input <region_dir> is required");
                return 2;
            }
            if (!Directory.Exists(inputDir))
            {
                Console.Error.WriteLine($"Input directory not found: {inputDir}");
                return 2;
            }

            opt.TryGetValue("pattern", out var pattern);
            if (string.IsNullOrWhiteSpace(pattern)) pattern = "*.pm4";
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var objectsDir = OutputLocator.EnsureSubfolder(root, "objects");
            var csvPath = Path.Combine(objectsDir, "objects.csv");

            Console.WriteLine($"[object-centroids] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);

            int tileCount = 0;
            int objectTotal = 0;

            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("tile_id,tile_x,tile_y,object_id,center_x,center_y,center_z,vertex_count,triangle_count,surface_count");

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var tileId = coord.ToLinearIndex();

                var objects = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(tile.Scene);
                int objId = 0;
                foreach (var obj in objects)
                {
                    var c = obj.BoundingCenter;
                    int triCount = obj.Triangles?.Count ?? 0;
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4:F6},{5:F6},{6:F6},{7},{8},{9}",
                        tileId, coord.X, coord.Y, objId, c.X, c.Y, c.Z, obj.VertexCount, triCount, obj.SurfaceCount));
                    objId++;
                }

                tileCount++;
                objectTotal += objects.Count;
            }

            Console.WriteLine($"[object-centroids] Wrote {objectTotal:N0} objects from {tileCount:N0} tiles → {csvPath}");
            return 0;
        }

        private static int RunMprrDump(Dictionary<string, string?> opt)
        {
            if (!opt.TryGetValue("input", out var inputDir) || string.IsNullOrWhiteSpace(inputDir))
            {
                Console.Error.WriteLine("--input <region_dir> is required");
                return 2;
            }
            if (!Directory.Exists(inputDir))
            {
                Console.Error.WriteLine($"Input directory not found: {inputDir}");
                return 2;
            }

            opt.TryGetValue("pattern", out var pattern);
            if (string.IsNullOrWhiteSpace(pattern)) pattern = "*.pm4";
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var mprrDir = OutputLocator.EnsureSubfolder(root, "mprr");
            var csvPath = Path.Combine(mprrDir, "mprr.csv");

            Console.WriteLine($"[mprr-dump] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);

            int tileCount = 0;
            int entryTotal = 0;

            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("tile_id,tile_x,tile_y,row_index,value1,value2,is_sentinel,object_seq");

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var tileId = coord.ToLinearIndex();

                var props = tile.Scene.Properties ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MprrChunk.Entry>();
                int objectSeq = -1; // increments when encountering a sentinel row (Value1==65535)

                for (int i = 0; i < props.Count; i++)
                {
                    var e = props[i];
                    bool isSentinel = e.Value1 == 65535;
                    if (isSentinel) objectSeq++;
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4},{5},{6},{7}",
                        tileId, coord.X, coord.Y, i, e.Value1, e.Value2, isSentinel ? 1 : 0, objectSeq));
                }

                tileCount++;
                entryTotal += props.Count;
            }

            Console.WriteLine($"[mprr-dump] Wrote {entryTotal:N0} entries from {tileCount:N0} tiles → {csvPath}");
            return 0;
        }

        private static int RunMscnAssociate(Dictionary<string, string?> opt)
        {
            // Associations operate on previously generated CSVs under the same output root/session
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var objectsCsv = Path.Combine(OutputLocator.EnsureSubfolder(root, "objects"), "objects.csv");
            var pointsCsv = Path.Combine(OutputLocator.EnsureSubfolder(root, "points"), "mscn_points.csv");
            var assocDir = OutputLocator.EnsureSubfolder(root, "assoc");
            var assocCsv = Path.Combine(assocDir, "associations.csv");

            if (!File.Exists(objectsCsv))
            {
                Console.Error.WriteLine($"objects.csv not found: {objectsCsv}. Run object-centroids first (same --session/--out).");
                return 2;
            }
            if (!File.Exists(pointsCsv))
            {
                Console.Error.WriteLine($"mscn_points.csv not found: {pointsCsv}. Run mscn-dump first (same --session/--out).");
                return 2;
            }

            int k = 1;
            if (opt.TryGetValue("k", out var kStr) && int.TryParse(kStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var kVal) && kVal > 0)
                k = Math.Min(kVal, 64);
            double? maxDist = null;
            if (opt.TryGetValue("max-dist", out var mdStr) && double.TryParse(mdStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var mdVal) && mdVal > 0)
                maxDist = mdVal;

            Console.WriteLine($"[mscn-associate] Loading CSVs from {root}");

            // Load MSCN points grouped by tile
            var mscnByTile = new Dictionary<int, List<(double x,double y,double z,int localIndex)>>();
            using (var sr = new StreamReader(pointsCsv))
            {
                string? line = sr.ReadLine(); // header
                var localCounters = new Dictionary<int, int>();
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = line.Split(',');
                    if (parts.Length < 6) continue;
                    int tileId = int.Parse(parts[0], CultureInfo.InvariantCulture);
                    double x = double.Parse(parts[3], CultureInfo.InvariantCulture);
                    double y = double.Parse(parts[4], CultureInfo.InvariantCulture);
                    double z = double.Parse(parts[5], CultureInfo.InvariantCulture);
                    if (!mscnByTile.TryGetValue(tileId, out var list))
                    {
                        list = new List<(double,double,double,int)>();
                        mscnByTile[tileId] = list;
                        localCounters[tileId] = 0;
                    }
                    int localIdx = localCounters[tileId];
                    localCounters[tileId] = localIdx + 1;
                    list.Add((x,y,z,localIdx));
                }
            }

            // Utility to get neighbors up to radius 1 if needed
            static IEnumerable<int> NeighborTileIds(int tileId)
            {
                var coord = Pm4GlobalTileLoader.TileCoordinate.FromLinearIndex(tileId);
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = coord.X + dx;
                    int ny = coord.Y + dy;
                    if (nx < 0 || ny < 0 || nx > 63 || ny > 63) continue;
                    yield return ny * 64 + nx;
                }
            }

            int assocCount = 0;
            int objectCount = 0;
            using var sw = new StreamWriter(assocCsv);
            sw.WriteLine("tile_id,tile_x,tile_y,object_id,center_x,center_y,center_z,mscn_tile_id,mscn_tile_x,mscn_tile_y,mscn_index,mscn_x,mscn_y,mscn_z,distance,rank");

            using (var sr = new StreamReader(objectsCsv))
            {
                string? line = sr.ReadLine(); // header
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var p = line.Split(',');
                    if (p.Length < 10) continue;
                    int tileId = int.Parse(p[0], CultureInfo.InvariantCulture);
                    int tileX = int.Parse(p[1], CultureInfo.InvariantCulture);
                    int tileY = int.Parse(p[2], CultureInfo.InvariantCulture);
                    int objId = int.Parse(p[3], CultureInfo.InvariantCulture);
                    double cx = double.Parse(p[4], CultureInfo.InvariantCulture);
                    double cy = double.Parse(p[5], CultureInfo.InvariantCulture);
                    double cz = double.Parse(p[6], CultureInfo.InvariantCulture);

                    // Candidate points: same tile, else include neighbors
                    var candidates = new List<(int mTileId,int mIndex,double x,double y,double z,double d2)>();
                    void AddCandidatesFromTile(int tId)
                    {
                        if (!mscnByTile.TryGetValue(tId, out var list)) return;
                        foreach (var (x,y,z,locIdx) in list)
                        {
                            double dx = x - cx, dy = y - cy, dz = z - cz;
                            double d2 = dx*dx + dy*dy + dz*dz;
                            candidates.Add((tId, locIdx, x,y,z, d2));
                        }
                    }

                    AddCandidatesFromTile(tileId);
                    if (candidates.Count == 0)
                    {
                        foreach (var nId in NeighborTileIds(tileId)) AddCandidatesFromTile(nId);
                    }

                    if (candidates.Count == 0)
                    {
                        // No MSCN points at all nearby; skip emitting rows for this object
                        objectCount++;
                        continue;
                    }

                    var ordered = candidates.OrderBy(c => c.d2)
                                             .Take(k)
                                             .ToList();

                    int rank = 1;
                    foreach (var c in ordered)
                    {
                        double dist = Math.Sqrt(c.d2);
                        if (maxDist.HasValue && dist > maxDist.Value) continue;
                        var mCoord = Pm4GlobalTileLoader.TileCoordinate.FromLinearIndex(c.mTileId);
                        sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4:F6},{5:F6},{6:F6},{7},{8},{9},{10},{11:F6},{12:F6},{13:F6},{14:F6},{15}",
                            tileId, tileX, tileY, objId, cx, cy, cz,
                            c.mTileId, mCoord.X, mCoord.Y, c.mIndex, c.x, c.y, c.z, dist, rank));
                        assocCount++;
                        rank++;
                    }

                    objectCount++;
                }
            }

            Console.WriteLine($"[mscn-associate] Wrote {assocCount:N0} associations for {objectCount:N0} objects → {assocCsv}");
            return 0;
        }
    }
}

