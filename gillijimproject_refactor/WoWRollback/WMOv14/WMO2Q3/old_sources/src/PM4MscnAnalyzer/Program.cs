using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.Coordinate;
using System.Text.Json;
using System.Text;

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
                    "links-dump" => RunLinksDump(options),
                    "links-summarize" => RunLinksSummarize(options),
                    "mscn-associate" => RunMscnAssociate(options),
                    "links-pivot" => RunLinksPivot(options),
                    "mslk-export" => RunMslkExport(options),
                    "mspi-analyze" => RunMspiAnalyze(options),
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
            Console.WriteLine("  PM4MscnAnalyzer object-centroids --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>] [--debug-placements]");
            Console.WriteLine("  PM4MscnAnalyzer mprr-dump --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer links-dump --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer links-summarize --session <name> [--out <path>]");
            Console.WriteLine("  PM4MscnAnalyzer mscn-associate --session <name> [--out <path>] [--k <int>] [--max-dist <float>]");
            Console.WriteLine("  PM4MscnAnalyzer links-pivot --session <name> [--out <path>] [--triples]");
            Console.WriteLine("  PM4MscnAnalyzer mslk-export --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>]");
            Console.WriteLine("  PM4MscnAnalyzer mspi-analyze --input <region_dir> [--pattern <glob>] [--out <path>] [--session <name>] [--with-mscn [true|false]] [--proximity-samples <int>] [--emit-tri-sets] [--max-tiles <int>] [--verbose]");
            Console.WriteLine();
            Console.WriteLine("Commands:");
            Console.WriteLine("  mscn-dump        Export MSCN points with tile provenance to CSV");
            Console.WriteLine("  object-centroids Compute per-tile object centroids and stats to CSV");
            Console.WriteLine("  mprr-dump        Export raw MPRR properties with tile provenance to CSV");
            Console.WriteLine("  links-dump       Export placements.csv and mprl_mslk_links.csv without debug flags");
            Console.WriteLine("  links-summarize  Summarize link counts and non-geometry type/flag combos from CSVs");
            Console.WriteLine("  mscn-associate   kNN from object centroids to MSCN points, emits associations.csv");
            Console.WriteLine("  links-pivot      Pivot/link analytics over existing CSVs (pairs/triples/stats by surface_ref)");
            Console.WriteLine("  mslk-export      Export per-placement MSLK structure (containers+geometries) JSONL and signatures");
            Console.WriteLine("  mspi-analyze     Harvest MSPI vs MSVI/MSUR/MSLK/MSCN relationships and emit analysis CSVs");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --input <dir>    Directory containing *.pm4 region tiles (required)");
            Console.WriteLine("  --pattern <glob> File search pattern, default: *.pm4");
            Console.WriteLine("  --out <path>     Output directory (default: project_output/mscn_analysis_<ts>)");
            Console.WriteLine("  --session <name> Session name to nest under output (appends timestamp)");
            Console.WriteLine("  --k <int>        Number of nearest MSCN to return (default 1)");
            Console.WriteLine("  --max-dist <f>   Optional distance cutoff in world units");
            Console.WriteLine("  --debug-placements  With object-centroids: write placements.csv (placement provenance)");
            Console.WriteLine("  --triples        With links-pivot: also compute non-geom surface_ref triples (C(n,3))");
            Console.WriteLine("  --with-mscn      Include MSCN remap/proximity (default true; pass 'false' to disable)");
            Console.WriteLine("  --proximity-samples <int>  Number of centroid samples per source (default 1000; 0 disables)");
            Console.WriteLine("  --emit-tri-sets  Dump canonicalized triangle sets to tri_sets.jsonl (heavy)");
            Console.WriteLine("  --max-tiles <int>  Safety limit on tiles iterated during analysis (optional)");
            Console.WriteLine("  --verbose        Extra logging");
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

        private static int RunMspiAnalyze(Dictionary<string, string?> opt)
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

            bool withMscn = true;
            if (opt.TryGetValue("with-mscn", out var mscnStr))
            {
                if (!string.IsNullOrWhiteSpace(mscnStr))
                {
                    withMscn = !string.Equals(mscnStr, "false", StringComparison.OrdinalIgnoreCase);
                }
            }

            int proximitySamples = 1000;
            if (opt.TryGetValue("proximity-samples", out var proxStr) &&
                int.TryParse(proxStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var proxVal) && proxVal >= 0)
            {
                proximitySamples = proxVal;
            }

            bool emitTriSets = opt.ContainsKey("emit-tri-sets");

            int? maxTiles = null;
            if (opt.TryGetValue("max-tiles", out var maxStr) &&
                int.TryParse(maxStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var maxVal) && maxVal > 0)
            {
                maxTiles = maxVal;
            }

            bool verbose = opt.ContainsKey("verbose");

            var root = OutputLocator.ResolveRoot(outArg, session);
            Console.WriteLine($"[mspi-analyze] Output root: {root}");

            Console.WriteLine($"[mspi-analyze] Loading region: {inputDir} ({pattern}), withMscn={withMscn}");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: withMscn);
            var scene = Pm4GlobalTileLoader.ToStandardScene(global);

            var options = new MspiAnalyzeOptions
            {
                RootOutput = root,
                RegionDir = inputDir!,
                Pattern = pattern!,
                WithMscn = withMscn,
                ProximitySamples = proximitySamples,
                EmitTriSets = emitTriSets,
                MaxTiles = maxTiles,
                Verbose = verbose,
                Session = session
            };

            var analysis = new MspiVsMsviAnalysis();
            analysis.Run(options, global, scene);

            Console.WriteLine("[mspi-analyze] Completed");
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
            var debugPlacements = opt.ContainsKey("debug-placements");
            StreamWriter? placementsWriter = null;
            string? placementsCsv = null;
            StreamWriter? linksWriter = null;
            string? linksCsv = null;
            if (debugPlacements)
            {
                placementsCsv = Path.Combine(objectsDir, "placements.csv");
                placementsWriter = new StreamWriter(placementsCsv);
                placementsWriter.WriteLine("tile_id,tile_x,tile_y,mprl_index,unknown4,px,py,pz,matched_mslk_count,matched_mslk_geom_count");
                linksCsv = Path.Combine(objectsDir, "mprl_mslk_links.csv");
                linksWriter = new StreamWriter(linksCsv);
                linksWriter.WriteLine("tile_id,tile_x,tile_y,mprl_index,unknown4,mslk_index,has_geometry,mspi_first_index,mspi_index_count,surface_ref,type_0x01,flags_0x00,unknown_0x12,mslk_tile_x,mslk_tile_y");
            }

            Console.WriteLine($"[object-centroids] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);

            int tileCount = 0;
            int objectTotal = 0;

            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("tile_id,tile_x,tile_y,object_id,center_x,center_y,center_z,center_geom_x,center_geom_y,center_geom_z,placement_count,vertex_count,triangle_count,surface_count");

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var tileId = coord.ToLinearIndex();

                var objects = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(tile.Scene);
                
                if (debugPlacements && tile.Scene.Placements != null)
                {
                    // Emit placement provenance rows for diagnostics
                    var links = tile.Scene.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                    foreach (var tuple in tile.Scene.Placements.Select((p, idx) => (p, idx)))
                    {
                        var p = tuple.p;
                        int idx = tuple.idx;
                        var worldPos = CoordinateTransformationService.ApplyPm4Transformation(new Vector3(p.Position.X, p.Position.Y, p.Position.Z));
                        uint key = (uint)p.Unknown4;
                        var matched = links.Select((l, li) => (l, li)).Where(t => t.l.ParentIndex == key).ToList();
                        int matchCount = matched.Count;
                        int geomCount = matched.Count(t => t.l.HasGeometry);
                        // Note: output px,py,pz as X,Z,Y respectively so that py reflects height (Z-axis)
                        placementsWriter!.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4},{5:F6},{6:F6},{7:F6},{8},{9}",
                            tileId, coord.X, coord.Y, idx, key, worldPos.X, worldPos.Z, worldPos.Y, matchCount, geomCount));

                        // Emit one row per matched MSLK entry with detailed fields
                        foreach (var t in matched)
                        {
                            var ml = t.l;
                            int mslkIndex = t.li;
                            int hasGeom = ml.HasGeometry ? 1 : 0;
                            // Note: ReferenceIndex is a convenience accessor for SurfaceRefIndex
                            int mspiFirst = ml.MspiFirstIndex;
                            int mspiCount = ml.MspiIndexCount;
                            int surfRef = ml.ReferenceIndex;
                            int type01 = ml.Type_0x01;
                            int flags00 = ml.Flags_0x00;
                            int unk12 = ml.Unknown_0x12;
                            int mslkTileX = ml.LinkIdTileX;
                            int mslkTileY = ml.LinkIdTileY;
                            linksWriter!.WriteLine(string.Format(CultureInfo.InvariantCulture,
                                "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}",
                                tileId, coord.X, coord.Y, idx, key, mslkIndex, hasGeom, mspiFirst, mspiCount, surfRef, type01, flags00, unk12, mslkTileX, mslkTileY));
                        }
                    }
                }
                int objId = 0;
                foreach (var obj in objects)
                {
                    var pc = obj.PlacementCenter; // primary center for association
                    var gc = obj.BoundingCenter;  // geometry centroid for diagnostics
                    int triCount = obj.Triangles?.Count ?? 0;
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1},{2},{3},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6},{9:F6},{10},{11},{12},{13}",
                        tileId, coord.X, coord.Y, objId,
                        pc.X, pc.Y, pc.Z,
                        gc.X, gc.Y, gc.Z,
                        obj.PlacementCount,
                        obj.VertexCount, triCount, obj.SurfaceCount));
                    objId++;
                }

                tileCount++;
                objectTotal += objects.Count;
            }

            placementsWriter?.Dispose();
            linksWriter?.Dispose();
            if (placementsCsv != null)
            {
                Console.WriteLine($"[object-centroids] Wrote placements provenance → {placementsCsv}");
            }
            if (linksCsv != null)
            {
                Console.WriteLine($"[object-centroids] Wrote MPRL⇄MSLK link details → {linksCsv}");
            }
            Console.WriteLine($"[object-centroids] Wrote {objectTotal:N0} objects from {tileCount:N0} tiles → {csvPath}");
            return 0;
        }

        private static int RunLinksDump(Dictionary<string, string?> opt)
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

            var placementsCsv = Path.Combine(objectsDir, "placements.csv");
            var linksCsv = Path.Combine(objectsDir, "mprl_mslk_links.csv");

            Console.WriteLine($"[links-dump] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);

            using var placementsWriter = new StreamWriter(placementsCsv);
            placementsWriter.WriteLine("tile_id,tile_x,tile_y,mprl_index,unknown4,px,py,pz,matched_mslk_count,matched_mslk_geom_count");
            using var linksWriter = new StreamWriter(linksCsv);
            linksWriter.WriteLine("tile_id,tile_x,tile_y,mprl_index,unknown4,mslk_index,has_geometry,mspi_first_index,mspi_index_count,surface_ref,type_0x01,flags_0x00,unknown_0x12,mslk_tile_x,mslk_tile_y");

            int tileCount = 0;
            int placementRows = 0;
            int linkRows = 0;

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var tileId = coord.ToLinearIndex();

                var links = tile.Scene.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                if (tile.Scene.Placements != null)
                {
                    foreach (var tuple in tile.Scene.Placements.Select((p, idx) => (p, idx)))
                    {
                        var p = tuple.p;
                        int idx = tuple.idx;
                        var worldPos = CoordinateTransformationService.ApplyPm4Transformation(new Vector3(p.Position.X, p.Position.Y, p.Position.Z));
                        uint key = (uint)p.Unknown4;
                        var matched = links.Select((l, li) => (l, li)).Where(t => t.l.ParentIndex == key).ToList();
                        int matchCount = matched.Count;
                        int geomCount = matched.Count(t => t.l.HasGeometry);
                        placementsWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4},{5:F6},{6:F6},{7:F6},{8},{9}",
                            tileId, coord.X, coord.Y, idx, key, worldPos.X, worldPos.Z, worldPos.Y, matchCount, geomCount));
                        placementRows++;

                        foreach (var t in matched)
                        {
                            var ml = t.l;
                            int mslkIndex = t.li;
                            int hasGeom = ml.HasGeometry ? 1 : 0;
                            int mspiFirst = ml.MspiFirstIndex;
                            int mspiCount = ml.MspiIndexCount;
                            int surfRef = ml.ReferenceIndex;
                            int type01 = ml.Type_0x01;
                            int flags00 = ml.Flags_0x00;
                            int unk12 = ml.Unknown_0x12;
                            int mslkTileX = ml.LinkIdTileX;
                            int mslkTileY = ml.LinkIdTileY;
                            linksWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                                "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}",
                                tileId, coord.X, coord.Y, idx, key, mslkIndex, hasGeom, mspiFirst, mspiCount, surfRef, type01, flags00, unk12, mslkTileX, mslkTileY));
                            linkRows++;
                        }
                    }
                }

                tileCount++;
            }

            Console.WriteLine($"[links-dump] Wrote {placementRows:N0} placement rows → {placementsCsv}");
            Console.WriteLine($"[links-dump] Wrote {linkRows:N0} link rows → {linksCsv}");
            Console.WriteLine($"[links-dump] Processed {tileCount:N0} tiles");
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

        private static int RunLinksSummarize(Dictionary<string, string?> opt)
        {
            // Summarize previously generated placements.csv and mprl_mslk_links.csv
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var objectsDir = OutputLocator.EnsureSubfolder(root, "objects");
            var placementsCsv = Path.Combine(objectsDir, "placements.csv");
            var linksCsv = Path.Combine(objectsDir, "mprl_mslk_links.csv");

            if (!File.Exists(placementsCsv) || !File.Exists(linksCsv))
            {
                Console.Error.WriteLine($"Missing inputs under {objectsDir}. Run links-dump or object-centroids --debug-placements first.");
                return 2;
            }

            // Load placement category counts: (matched_mslk_count, matched_mslk_geom_count)
            var catCounts = new Dictionary<(int mc,int gc), int>();
            using (var sr = new StreamReader(placementsCsv))
            {
                string? line = sr.ReadLine(); // header
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var p = line.Split(',');
                    if (p.Length < 10) continue;
                    int mc = int.Parse(p[8], CultureInfo.InvariantCulture);
                    int gc = int.Parse(p[9], CultureInfo.InvariantCulture);
                    var key = (mc, gc);
                    catCounts[key] = catCounts.TryGetValue(key, out var c) ? c + 1 : 1;
                }
            }

            // Load non-geometry type/flag combos from links
            var nonGeomCounts = new Dictionary<(int t,int f), int>();
            using (var sr = new StreamReader(linksCsv))
            {
                string? line = sr.ReadLine(); // header
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var p = line.Split(',');
                    if (p.Length < 15) continue;
                    int hasGeom = int.Parse(p[6], CultureInfo.InvariantCulture);
                    if (hasGeom != 0) continue;
                    int type01 = int.Parse(p[10], CultureInfo.InvariantCulture);
                    int flags00 = int.Parse(p[11], CultureInfo.InvariantCulture);
                    var key = (type01, flags00);
                    nonGeomCounts[key] = nonGeomCounts.TryGetValue(key, out var c) ? c + 1 : 1;
                }
            }

            // Write summaries
            var catCsv = Path.Combine(objectsDir, "link_category_counts.csv");
            using (var sw = new StreamWriter(catCsv))
            {
                sw.WriteLine("matched_mslk_count,matched_mslk_geom_count,count");
                foreach (var kv in catCounts.OrderBy(k => k.Key.mc).ThenBy(k => k.Key.gc))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2}", kv.Key.mc, kv.Key.gc, kv.Value));
                }
            }

            var nonGeomCsv = Path.Combine(objectsDir, "nongeom_type_flag_counts.csv");
            using (var sw = new StreamWriter(nonGeomCsv))
            {
                sw.WriteLine("type_0x01,flags_0x00,count");
                foreach (var kv in nonGeomCounts.OrderByDescending(k => k.Value))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2}", kv.Key.t, kv.Key.f, kv.Value));
                }
            }

            Console.WriteLine($"[links-summarize] Wrote category counts → {catCsv}");
            Console.WriteLine($"[links-summarize] Wrote non-geometry type/flag counts → {nonGeomCsv}");
            return 0;
        }

        private static int RunLinksPivot(Dictionary<string, string?> opt)
        {
            // Pivots over existing CSVs
            opt.TryGetValue("out", out var outArg);
            opt.TryGetValue("session", out var session);

            var root = OutputLocator.ResolveRoot(outArg, session);
            var objectsDir = OutputLocator.EnsureSubfolder(root, "objects");
            var placementsCsv = Path.Combine(objectsDir, "placements.csv");
            var linksCsv = Path.Combine(objectsDir, "mprl_mslk_links.csv");

            if (!File.Exists(placementsCsv) || !File.Exists(linksCsv))
            {
                Console.Error.WriteLine($"Missing inputs under {objectsDir}. Run links-dump first (same --session/--out).");
                return 2;
            }

            bool doTriples = opt.ContainsKey("triples");

            // Load placement categories per tile
            var tileCatCounts = new Dictionary<(int tileId,int mc,int gc), int>();
            var tileXY = new Dictionary<int, (int x,int y)>();
            using (var sr = new StreamReader(placementsCsv))
            {
                string? line = sr.ReadLine();
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var p = line.Split(',');
                    if (p.Length < 10) continue;
                    int tileId = int.Parse(p[0], CultureInfo.InvariantCulture);
                    int tileX = int.Parse(p[1], CultureInfo.InvariantCulture);
                    int tileY = int.Parse(p[2], CultureInfo.InvariantCulture);
                    int mc = int.Parse(p[8], CultureInfo.InvariantCulture);
                    int gc = int.Parse(p[9], CultureInfo.InvariantCulture);
                    tileXY[tileId] = (tileX, tileY);
                    var k = (tileId, mc, gc);
                    tileCatCounts[k] = tileCatCounts.TryGetValue(k, out var c) ? c + 1 : 1;
                }
            }

            // Load non-geometry links grouped per placement for pairs/triples
            var pairCounts = new Dictionary<(int a,int b), long>();
            var tripleCounts = new Dictionary<(int a,int b,int c), long>();
            var refStats_total = new Dictionary<int, long>();
            var refStats_placements = new Dictionary<int, HashSet<(int tileId,int mprlIdx)>>();
            var refStats_tiles = new Dictionary<int, HashSet<int>>();
            var geomTypeFlagCounts = new Dictionary<(int t,int f), long>();

            using (var sr = new StreamReader(linksCsv))
            {
                string? line = sr.ReadLine();
                // Map of placement key -> set of non-geom refs; but linksCsv is sorted by tile/placement naturally
                var perPlacementNonGeom = new Dictionary<(int tileId,int mprlIdx), HashSet<int>>();
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var p = line.Split(',');
                    if (p.Length < 15) continue;
                    int tileId = int.Parse(p[0], CultureInfo.InvariantCulture);
                    int mprlIdx = int.Parse(p[3], CultureInfo.InvariantCulture);
                    int hasGeom = int.Parse(p[6], CultureInfo.InvariantCulture);
                    int surfRef = int.Parse(p[9], CultureInfo.InvariantCulture);
                    int type01 = int.Parse(p[10], CultureInfo.InvariantCulture);
                    int flags00 = int.Parse(p[11], CultureInfo.InvariantCulture);

                    if (hasGeom == 1)
                    {
                        var keyG = (type01, flags00);
                        geomTypeFlagCounts[keyG] = geomTypeFlagCounts.TryGetValue(keyG, out var c) ? c + 1 : 1;
                    }
                    else
                    {
                        // Update stats
                        refStats_total[surfRef] = refStats_total.TryGetValue(surfRef, out var c) ? c + 1 : 1;
                        if (!refStats_placements.TryGetValue(surfRef, out var hsP)) { hsP = new HashSet<(int,int)>(); refStats_placements[surfRef] = hsP; }
                        hsP.Add((tileId, mprlIdx));
                        if (!refStats_tiles.TryGetValue(surfRef, out var hsT)) { hsT = new HashSet<int>(); refStats_tiles[surfRef] = hsT; }
                        hsT.Add(tileId);

                        var pk = (tileId, mprlIdx);
                        if (!perPlacementNonGeom.TryGetValue(pk, out var set))
                        {
                            set = new HashSet<int>();
                            perPlacementNonGeom[pk] = set;
                        }
                        set.Add(surfRef);
                    }
                }

                // After reading all, produce pairs/triples by placement
                foreach (var set in perPlacementNonGeom.Values)
                {
                    if (set.Count >= 2)
                    {
                        var arr = set.OrderBy(v => v).ToArray();
                        for (int i = 0; i < arr.Length; i++)
                        for (int j = i + 1; j < arr.Length; j++)
                        {
                            var k = (arr[i], arr[j]);
                            pairCounts[k] = pairCounts.TryGetValue(k, out var c) ? c + 1 : 1;
                        }
                        if (doTriples && arr.Length >= 3)
                        {
                            for (int i = 0; i < arr.Length; i++)
                            for (int j = i + 1; j < arr.Length; j++)
                            for (int k3 = j + 1; k3 < arr.Length; k3++)
                            {
                                var k = (arr[i], arr[j], arr[k3]);
                                tripleCounts[k] = tripleCounts.TryGetValue(k, out var c) ? c + 1 : 1;
                            }
                        }
                    }
                }
            }

            // Write outputs
            var pairsCsv = Path.Combine(objectsDir, "nongeom_surface_ref_pairs.csv");
            using (var sw = new StreamWriter(pairsCsv))
            {
                sw.WriteLine("ref_a,ref_b,count");
                foreach (var kv in pairCounts.OrderByDescending(k => k.Value))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2}", kv.Key.a, kv.Key.b, kv.Value));
                }
            }
            if (doTriples)
            {
                var triplesCsv = Path.Combine(objectsDir, "nongeom_surface_ref_triples.csv");
                using var swt = new StreamWriter(triplesCsv);
                swt.WriteLine("ref_a,ref_b,ref_c,count");
                foreach (var kv in tripleCounts.OrderByDescending(k => k.Value))
                {
                    swt.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3}", kv.Key.a, kv.Key.b, kv.Key.c, kv.Value));
                }
            }

            var statsCsv = Path.Combine(objectsDir, "nongeom_surface_ref_stats.csv");
            using (var sw = new StreamWriter(statsCsv))
            {
                sw.WriteLine("surface_ref,total_rows,distinct_placements,distinct_tiles");
                foreach (var refId in refStats_total.Keys.OrderByDescending(k => refStats_total[k]))
                {
                    long total = refStats_total[refId];
                    int distinctPlacements = refStats_placements.TryGetValue(refId, out var hsP) ? hsP.Count : 0;
                    int distinctTiles = refStats_tiles.TryGetValue(refId, out var hsT) ? hsT.Count : 0;
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3}", refId, total, distinctPlacements, distinctTiles));
                }
            }

            var geomTfCsv = Path.Combine(objectsDir, "geom_type_flag_counts.csv");
            using (var sw = new StreamWriter(geomTfCsv))
            {
                sw.WriteLine("type_0x01,flags_0x00,count");
                foreach (var kv in geomTypeFlagCounts.OrderByDescending(k => k.Value))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2}", kv.Key.t, kv.Key.f, kv.Value));
                }
            }

            var profileCsv = Path.Combine(objectsDir, "placement_profiles_by_tile.csv");
            using (var sw = new StreamWriter(profileCsv))
            {
                sw.WriteLine("tile_id,tile_x,tile_y,matched_mslk_count,matched_mslk_geom_count,count");
                foreach (var kv in tileCatCounts.OrderBy(k => k.Key.tileId).ThenBy(k => k.Key.mc).ThenBy(k => k.Key.gc))
                {
                    tileXY.TryGetValue(kv.Key.tileId, out var xy);
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5}", kv.Key.tileId, xy.x, xy.y, kv.Key.mc, kv.Key.gc, kv.Value));
                }
            }

            Console.WriteLine($"[links-pivot] Wrote pairs → {pairsCsv}");
            Console.WriteLine($"[links-pivot] Wrote stats → {statsCsv}");
            Console.WriteLine($"[links-pivot] Wrote geom type/flag counts → {geomTfCsv}");
            Console.WriteLine($"[links-pivot] Wrote per-tile placement profiles → {profileCsv}");
            return 0;
        }

        private static int RunMslkExport(Dictionary<string, string?> opt)
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
            var jsonlPath = Path.Combine(objectsDir, "mslk_by_placement.jsonl");
            var contSigCsv = Path.Combine(objectsDir, "mslk_container_signatures.csv");
            var geomSigCsv = Path.Combine(objectsDir, "mslk_geom_signatures.csv");

            Console.WriteLine($"[mslk-export] Loading region: {inputDir} ({pattern})");
            var global = Pm4GlobalTileLoader.LoadRegion(inputDir!, pattern!, applyMscnRemap: true);

            var jsonOptions = new JsonSerializerOptions { WriteIndented = false };
            var containerSigCounts = new Dictionary<string, long>();
            var geomSigCounts = new Dictionary<string, long>();

            int placementTotal = 0;
            int tileCount = 0;
            using var jw = new StreamWriter(jsonlPath, false, Encoding.UTF8);

            foreach (var kvp in global.LoadedTiles.OrderBy(k => k.Key.Y).ThenBy(k => k.Key.X))
            {
                var coord = kvp.Key;
                var tile = kvp.Value;
                var tileId = coord.ToLinearIndex();

                var links = tile.Scene.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                var placements = tile.Scene.Placements ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry>();

                foreach (var tuple in placements.Select((p, idx) => (p, idx)))
                {
                    var p = tuple.p;
                    int idx = tuple.idx;
                    uint key = (uint)p.Unknown4;
                    var matched = links.Where(l => l.ParentIndex == key).ToList();

                    var nonGeom = new List<object>();
                    var geom = new List<object>();
                    var contRefs = new HashSet<int>();
                    var geomRefs = new HashSet<int>();

                    foreach (var ml in matched)
                    {
                        if (ml.HasGeometry)
                        {
                            geom.Add(new {
                                surface_ref = ml.ReferenceIndex,
                                type_0x01 = ml.Type_0x01,
                                flags_0x00 = ml.Flags_0x00,
                                mspi_first_index = ml.MspiFirstIndex,
                                mspi_index_count = ml.MspiIndexCount,
                                mslk_tile_x = ml.LinkIdTileX,
                                mslk_tile_y = ml.LinkIdTileY
                            });
                            geomRefs.Add(ml.ReferenceIndex);
                        }
                        else
                        {
                            nonGeom.Add(new {
                                surface_ref = ml.ReferenceIndex,
                                type_0x01 = ml.Type_0x01,
                                flags_0x00 = ml.Flags_0x00,
                                unknown_0x12 = ml.Unknown_0x12,
                                mslk_tile_x = ml.LinkIdTileX,
                                mslk_tile_y = ml.LinkIdTileY
                            });
                            contRefs.Add(ml.ReferenceIndex);
                        }
                    }

                    // World position, following csv convention X,Z,Y → px,py,pz
                    var worldPos = CoordinateTransformationService.ApplyPm4Transformation(new Vector3(p.Position.X, p.Position.Y, p.Position.Z));

                    var row = new {
                        tile_id = tileId,
                        tile_x = coord.X,
                        tile_y = coord.Y,
                        mprl_index = idx,
                        unknown4 = p.Unknown4,
                        px = worldPos.X,
                        py = worldPos.Z,
                        pz = worldPos.Y,
                        counts = new { total = matched.Count, geom = matched.Count(m => m.HasGeometry), nongeom = matched.Count(m => !m.HasGeometry) },
                        containers = nonGeom,
                        geometries = geom
                    };
                    jw.WriteLine(JsonSerializer.Serialize(row, jsonOptions));
                    placementTotal++;

                    // Signatures
                    if (contRefs.Count > 0)
                    {
                        var sig = string.Join('+', contRefs.OrderBy(v => v));
                        containerSigCounts[sig] = containerSigCounts.TryGetValue(sig, out var c) ? c + 1 : 1;
                    }
                    if (geomRefs.Count > 0)
                    {
                        var sig = string.Join('+', geomRefs.OrderBy(v => v));
                        geomSigCounts[sig] = geomSigCounts.TryGetValue(sig, out var c) ? c + 1 : 1;
                    }
                }

                tileCount++;
            }

            using (var sw = new StreamWriter(contSigCsv))
            {
                sw.WriteLine("signature,count");
                foreach (var kv in containerSigCounts.OrderByDescending(k => k.Value))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1}", kv.Key, kv.Value));
                }
            }
            using (var sw = new StreamWriter(geomSigCsv))
            {
                sw.WriteLine("signature,count");
                foreach (var kv in geomSigCounts.OrderByDescending(k => k.Value))
                {
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1}", kv.Key, kv.Value));
                }
            }

            Console.WriteLine($"[mslk-export] Wrote {placementTotal:N0} placement rows → {jsonlPath}");
            Console.WriteLine($"[mslk-export] Wrote container signatures → {contSigCsv}");
            Console.WriteLine($"[mslk-export] Wrote geometry signatures → {geomSigCsv}");
            Console.WriteLine($"[mslk-export] Processed {tileCount:N0} tiles");
            return 0;
        }
    }
}

