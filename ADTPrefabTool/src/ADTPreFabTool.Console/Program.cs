using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.ADT;
using System.Numerics;
using System.Text;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using System.Text.RegularExpressions;

namespace ADTPreFabTool
{
    public class Program
    {
        public static void Main(string[] args)
        {
            System.Console.WriteLine("ADTPreFabTool - WoW ADT Terrain Mesh Extractor");
            System.Console.WriteLine("==================================================");

            if (args.Length < 1)
            {
                System.Console.WriteLine("Usage: ADTPreFabTool.Console <adt_file_or_folder_path> [output_directory_or_root] [--recursive|--no-recursive] [--no-comments] [--glb] [--gltf] [--glb-per-file|--no-glb-per-file] [--manifest|--no-manifest] [--output-root <path>] [--timestamped|--no-timestamp] [--chunks-manifest|--no-chunks-manifest] [--meta|--no-meta] [--similarity-only] [--tiles x_y,...] [--tile-range x1,y1,x2,y2] [--max-hamming N] [--chunk-min-similarity S] [--prefab-scan] [--prefab-sizes 2x2,4x4,...] [--prefab-stride N] [--prefab-max-hamming N] [--prefab-min-similarity S] [--prefab-min-ruggedness R] [--prefab-min-edge-density E] [--prefab-cross-tiles|--no-prefab-cross-tiles] [--export-matches] [--export-prefab-matches] [--export-max N] [--superblock-patterns] [--superblock-sizes 12x12,...] [--seeds <path>] [--edge-trim-max N] [--delta-export] [--minimap-root <path>] [--trs <path>] [--export-minimap-overlays] [--generate-seed-template] [--data-version V] [--cache-root <path>] [--decode-minimap]");
                System.Console.WriteLine("Defaults (directory input): --recursive --glb-per-file --manifest --timestamped --chunks-manifest --meta");
                System.Console.WriteLine("Examples:");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/terrain.adt\" \"output/\" --glb");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --output-root project_output --timestamped");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --no-recursive --no-manifest --no-glb-per-file");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --similarity-only --tile-range 30,41,32,43 --max-hamming 6");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --prefab-scan --prefab-sizes 2x2,4x4 --prefab-max-hamming 12 --export-prefab-matches --export-max 50");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --superblock-patterns --superblock-sizes 12x12 --seeds seeds.csv --edge-trim-max 2 --delta-export --minimap-root minimap_root --trs trs.txt --export-minimap-overlays --generate-seed-template");
                return;
            }

            string inputPath = args[0];
            string? outputArg = (args.Length > 1 && !args[1].StartsWith("--")) ? args[1] : null;
            // Defaults: recursive + per-file GLBs + manifest + timestamped runs + chunks/meta per tile
            bool recursive = true;
            bool noComments = args.Any(a => a.Equals("--no-comments", StringComparison.OrdinalIgnoreCase));
            bool flagGlb = args.Any(a => a.Equals("--glb", StringComparison.OrdinalIgnoreCase));
            bool flagGltf = args.Any(a => a.Equals("--gltf", StringComparison.OrdinalIgnoreCase));
            bool flagGlbPerFile = true;
            bool flagManifest = true;
            bool flagTimestamped = true;
            bool flagChunksManifest = true;
            bool flagMeta = true;
            bool similarityOnly = args.Any(a => a.Equals("--similarity-only", StringComparison.OrdinalIgnoreCase));
            bool prefabScan = args.Any(a => a.Equals("--prefab-scan", StringComparison.OrdinalIgnoreCase));
            bool exportMatches = args.Any(a => a.Equals("--export-matches", StringComparison.OrdinalIgnoreCase));
            bool exportPrefabMatches = args.Any(a => a.Equals("--export-prefab-matches", StringComparison.OrdinalIgnoreCase));
            int exportMax = 50;
            int maxHamming = 6;
            (int x1, int y1, int x2, int y2)? tileRange = null;
            HashSet<(int tx, int ty)> tileSet = new HashSet<(int, int)>();
            string outputRoot = "project_output";

            // Superblock + minimap/TRS seed workflow flags
            bool superblockPatterns = args.Any(a => a.Equals("--superblock-patterns", StringComparison.OrdinalIgnoreCase));
            var superblockSizes = new List<(int w, int h)>() { (12, 12) }; // default includes 12x12
            string? seedsCsvPath = null;
            int edgeTrimMax = 2;
            bool deltaExport = args.Any(a => a.Equals("--delta-export", StringComparison.OrdinalIgnoreCase));
            bool exportMinimapOverlays = args.Any(a => a.Equals("--export-minimap-overlays", StringComparison.OrdinalIgnoreCase));
            bool generateSeedTemplate = args.Any(a => a.Equals("--generate-seed-template", StringComparison.OrdinalIgnoreCase));
            string? minimapRoot = null; // root containing World\Textures\Minimap\<MapName>
            string? trsPath = null;     // mapping file path
            string? dataVersion = null; // e.g., 0.6.0
            string? cacheRoot = null;   // root for caches (defaults to outputRoot)
            bool decodeMinimap = args.Any(a => a.Equals("--decode-minimap", StringComparison.OrdinalIgnoreCase));

            // Allow overrides
            if (args.Any(a => a.Equals("--no-recursive", StringComparison.OrdinalIgnoreCase))) recursive = false;
            if (args.Any(a => a.Equals("--recursive", StringComparison.OrdinalIgnoreCase))) recursive = true;
            if (args.Any(a => a.Equals("--no-glb-per-file", StringComparison.OrdinalIgnoreCase))) flagGlbPerFile = false;
            if (args.Any(a => a.Equals("--glb-per-file", StringComparison.OrdinalIgnoreCase))) flagGlbPerFile = true;
            if (args.Any(a => a.Equals("--no-manifest", StringComparison.OrdinalIgnoreCase))) flagManifest = false;
            if (args.Any(a => a.Equals("--manifest", StringComparison.OrdinalIgnoreCase))) flagManifest = true;
            if (args.Any(a => a.Equals("--no-timestamp", StringComparison.OrdinalIgnoreCase) || a.Equals("--no-timestamped", StringComparison.OrdinalIgnoreCase))) flagTimestamped = false;
            if (args.Any(a => a.Equals("--timestamped", StringComparison.OrdinalIgnoreCase) || a.Equals("--timestamp", StringComparison.OrdinalIgnoreCase))) flagTimestamped = true;
            if (args.Any(a => a.Equals("--no-chunks-manifest", StringComparison.OrdinalIgnoreCase))) flagChunksManifest = false;
            if (args.Any(a => a.Equals("--chunks-manifest", StringComparison.OrdinalIgnoreCase))) flagChunksManifest = true;
            if (args.Any(a => a.Equals("--no-meta", StringComparison.OrdinalIgnoreCase))) flagMeta = false;
            if (args.Any(a => a.Equals("--meta", StringComparison.OrdinalIgnoreCase))) flagMeta = true;

            // Prefab defaults
            int prefabStride = 2;
            int prefabMaxHamming = 26; // ~80% similarity on 128-bit (ceil(0.2*128))
            float prefabMinRuggedness = 0.3f;
            float prefabMinEdgeDensity = 0.15f;
            var prefabSizes = new List<(int w, int h)>() { (8,8), (16,16) }; // larger defaults for better structure
            bool prefabCrossTiles = true; // allow cross-tile scanning by default

            // Optional similarity percentage mapping
            float? chunkMinSimilarity = null; // maps to 64-bit hamming
            float? prefabMinSimilarity = null; // maps to 128-bit hamming

            // Parse similarity/prefab flags
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].Equals("--max-hamming", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && int.TryParse(args[i + 1], out var mh))
                {
                    maxHamming = Math.Max(0, Math.Min(64, mh));
                }
                if (args[i].Equals("--tile-range", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    var parts = args[i + 1].Split(',', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 4 && int.TryParse(parts[0], out var rx1) && int.TryParse(parts[1], out var ry1) && int.TryParse(parts[2], out var rx2) && int.TryParse(parts[3], out var ry2))
                    {
                        tileRange = (Math.Min(rx1, rx2), Math.Min(ry1, ry2), Math.Max(rx1, rx2), Math.Max(ry1, ry2));
                    }
                }
                if (args[i].Equals("--tiles", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    foreach (var t in args[i + 1].Split(',', StringSplitOptions.RemoveEmptyEntries))
                    {
                        var p = t.Split('_');
                        if (p.Length == 2 && int.TryParse(p[0], out var tx) && int.TryParse(p[1], out var ty)) tileSet.Add((tx, ty));
                    }
                }
                if (args[i].Equals("--prefab-sizes", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    prefabSizes.Clear();
                    foreach (var s in args[i + 1].Split(',', StringSplitOptions.RemoveEmptyEntries))
                    {
                        var p = s.Split('x');
                        if (p.Length == 2 && int.TryParse(p[0], out var w) && int.TryParse(p[1], out var h) && w > 0 && h > 0)
                            prefabSizes.Add((w, h));
                    }
                }
                if (args[i].Equals("--prefab-stride", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && int.TryParse(args[i + 1], out var ps))
                {
                    prefabStride = Math.Max(1, ps);
                }
                if (args[i].Equals("--prefab-max-hamming", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && int.TryParse(args[i + 1], out var pmh))
                {
                    prefabMaxHamming = Math.Max(0, Math.Min(128, pmh));
                }
                if (args[i].Equals("--prefab-min-ruggedness", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var pr))
                {
                    prefabMinRuggedness = MathF.Max(0f, MathF.Min(1f, pr));
                }
                if (args[i].Equals("--prefab-min-edge-density", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var pe))
                {
                    prefabMinEdgeDensity = MathF.Max(0f, MathF.Min(1f, pe));
                }
                if (args[i].Equals("--prefab-min-similarity", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var pms))
                {
                    prefabMinSimilarity = MathF.Max(0f, MathF.Min(1f, pms));
                }
                if (args[i].Equals("--chunk-min-similarity", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var cms))
                {
                    chunkMinSimilarity = MathF.Max(0f, MathF.Min(1f, cms));
                }
                if (args[i].Equals("--no-prefab-cross-tiles", StringComparison.OrdinalIgnoreCase))
                {
                    prefabCrossTiles = false;
                }
                if (args[i].Equals("--prefab-cross-tiles", StringComparison.OrdinalIgnoreCase))
                {
                    prefabCrossTiles = true;
                }
                if (args[i].Equals("--export-max", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && int.TryParse(args[i + 1], out var em))
                {
                    exportMax = Math.Max(0, em);
                }
                if (args[i].Equals("--superblock-sizes", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    superblockSizes.Clear();
                    foreach (var s in args[i + 1].Split(',', StringSplitOptions.RemoveEmptyEntries))
                    {
                        var p = s.Split('x');
                        if (p.Length == 2 && int.TryParse(p[0], out var w) && int.TryParse(p[1], out var h) && w > 0 && h > 0)
                            superblockSizes.Add((w, h));
                    }
                }
                if (args[i].Equals("--seeds", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    seedsCsvPath = args[i + 1];
                }
                if (args[i].Equals("--edge-trim-max", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length && int.TryParse(args[i + 1], out var etm))
                {
                    edgeTrimMax = Math.Max(0, Math.Min(8, etm));
                }
                if (args[i].Equals("--minimap-root", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    minimapRoot = args[i + 1];
                }
                if (args[i].Equals("--trs", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    trsPath = args[i + 1];
                }
                if (args[i].Equals("--data-version", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    dataVersion = args[i + 1];
                }
                if (args[i].Equals("--cache-root", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    cacheRoot = args[i + 1];
                }
            }

            // Apply similarity percentage mappings if provided
            if (chunkMinSimilarity.HasValue)
            {
                maxHamming = Math.Clamp((int)MathF.Floor((1f - chunkMinSimilarity.Value) * 64f), 0, 64);
            }
            if (prefabMinSimilarity.HasValue)
            {
                prefabMaxHamming = Math.Clamp((int)MathF.Floor((1f - prefabMinSimilarity.Value) * 128f), 0, 128);
            }

            // Parse --output-root <path>
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].Equals("--output-root", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    outputRoot = args[i + 1];
                    break;
                }
            }

            try
            {
                if (Directory.Exists(inputPath))
                {
                    // Determine run output directory
                    string inputName = new DirectoryInfo(inputPath).Name;
                    string runDir = outputArg ?? outputRoot;
                    if (string.IsNullOrWhiteSpace(runDir)) runDir = outputRoot;
                    if (flagTimestamped)
                    {
                        string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
                        runDir = Path.Combine(runDir, $"{inputName}-{stamp}");
                    }
                    Directory.CreateDirectory(runDir);

                    if (superblockPatterns || exportMinimapOverlays || generateSeedTemplate)
                    {
                        // Optional TRS parsing for minimap support
                        List<(string mapName,int tileX,int tileY,string fullPath)> minimapEntries = new();
                        // Auto-detect md5translate file under minimap root if not provided
                        if (string.IsNullOrWhiteSpace(trsPath) && !string.IsNullOrWhiteSpace(minimapRoot))
                        {
                            var candidateTrs = Path.Combine(minimapRoot!, "md5translate.trs");
                            var candidateTxt = Path.Combine(minimapRoot!, "md5translate.txt");
                            if (File.Exists(candidateTrs)) trsPath = candidateTrs;
                            else if (File.Exists(candidateTxt)) trsPath = candidateTxt;
                        }
                        if (!string.IsNullOrWhiteSpace(minimapRoot))
                        {
                            try
                            {
                                if (!string.IsNullOrWhiteSpace(trsPath))
                                {
                                    minimapEntries = ParseTrsFile(trsPath!, minimapRoot!);
                                    System.Console.WriteLine($"TRS parsed: entries={minimapEntries.Count}");
                                }
                                else
                                {
                                    System.Console.WriteLine("No md5translate file found under minimap root; skipping TRS parsing.");
                                }
                            }
                            catch (Exception ex)
                            {
                                System.Console.WriteLine($"Warn: failed to parse TRS '{trsPath}': {ex.Message}");
                            }
                        }

                        // Minimal cache scaffolding (no decoding yet)
                        if (!string.IsNullOrWhiteSpace(minimapRoot))
                        {
                            string versionTag = dataVersion ?? FindVersionTagFromPath(minimapRoot!) ?? "unknown";
                            string cacheBase = cacheRoot ?? outputRoot;
                            string minimapCacheDir = Path.Combine(cacheBase, "_cache", versionTag, "minimap_png");
                            Directory.CreateDirectory(minimapCacheDir);
                            var manifest = new StringBuilder();
                            manifest.AppendLine("{");
                            manifest.AppendLine($"  \"version\": \"{versionTag}\",");
                            manifest.AppendLine($"  \"minimapRoot\": \"{minimapRoot!.Replace("\\", "/")}\",");
                            manifest.AppendLine($"  \"trsPath\": \"{(trsPath ?? "").Replace("\\", "/")}\",");
                            manifest.AppendLine($"  \"decoded\": {decodeMinimap.ToString().ToLowerInvariant()},");
                            manifest.AppendLine($"  \"entries\": {minimapEntries.Count}");
                            manifest.AppendLine("}");
                            File.WriteAllText(Path.Combine(minimapCacheDir, "cache_manifest.json"), manifest.ToString());
                            System.Console.WriteLine($"Cache ready at: {minimapCacheDir}");
                        }

                        // Seed/overlay helpers
                        if (exportMinimapOverlays)
                        {
                            ExportMinimapOverlays(runDir, minimapEntries);
                        }
                        if (generateSeedTemplate)
                        {
                            GenerateSeedTemplate(runDir, inputPath, tileSet, tileRange);
                        }
                        if (superblockPatterns)
                        {
                            ProcessSuperblockPatternsGlobal(inputPath, runDir, tileSet, tileRange, superblockSizes, seedsCsvPath, edgeTrimMax, deltaExport);
                        }
                    }
                    else if (prefabScan)
                    {
                        if (prefabCrossTiles)
                        {
                            ProcessPrefabDirectoryGlobal(inputPath, runDir, recursive, tileSet, tileRange, prefabSizes, prefabStride, prefabMaxHamming, prefabMinRuggedness, prefabMinEdgeDensity, exportPrefabMatches, exportMax);
                        }
                        else
                        {
                            ProcessPrefabDirectory(inputPath, runDir, recursive, tileSet, tileRange, prefabSizes, prefabStride, prefabMaxHamming, prefabMinRuggedness, prefabMinEdgeDensity, exportPrefabMatches, exportMax);
                        }
                    }
                    else if (similarityOnly)
                    {
                        ProcessSimilarityDirectory(inputPath, runDir, recursive, tileSet, tileRange, maxHamming);
                    }
                    else
                    {
                        ProcessADTDirectory(inputPath, runDir, recursive, noComments, flagGlbPerFile, flagManifest, flagChunksManifest, flagMeta);
                    }
                }
                else
                {
                    // Single file mode keeps legacy output behavior unless explicit root provided
                    string outputDir = outputArg ?? "output";
                    ProcessADTFile(inputPath, outputDir, flagGlb, flagGltf);
                }
            }
            catch (Exception ex)
            {
                System.Console.WriteLine($"Error processing input: {ex.Message}");
                System.Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        // === Minimap/TRS helpers ===
        private static List<(string mapName,int tileX,int tileY,string fullPath)> ParseTrsFile(string trsPath, string minimapRoot)
        {
            // Format supports multiple blocks:
            //   dir: <map_basename>
            //   <map_basename>\mapX_Y.blp\t<actual_filename>
            // actual_filename is relative to textures/minimap/
            // Also handle legacy order where left/right are swapped.
            var baseDir = Path.GetDirectoryName(trsPath);
            if (string.IsNullOrEmpty(baseDir)) baseDir = minimapRoot;
            var results = new List<(string,int,int,string)>();
            string? currentMap = null;
            foreach (var raw in File.ReadAllLines(trsPath))
            {
                var line = raw.Trim();
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
                {
                    currentMap = line.Substring(4).Trim();
                    continue;
                }
                if (currentMap == null) continue;
                var parts = line.Split('\t');
                if (parts.Length != 2) continue;
                string a = parts[0].Trim();
                string b = parts[1].Trim();
                // Identify which side is the map-stem and which is the actual filename
                // Current format: left == "<map>\\mapX_Y.blp", right == actual filename
                // Legacy format: left == actual filename, right == "<map>\\mapX_Y.blp"
                string mapSide;
                string actualSide;
                if (a.Contains("map") && a.Contains(".blp", StringComparison.OrdinalIgnoreCase)) { mapSide = a; actualSide = b; }
                else { mapSide = b; actualSide = a; }

                // Extract X,Y from mapSide
                var stem = Path.GetFileNameWithoutExtension(mapSide);
                if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
                var xy = stem.Substring(3).Split('_');
                if (xy.Length != 2 || !int.TryParse(xy[0], out var tx) || !int.TryParse(xy[1], out var ty)) continue;

                // actualSide is relative to textures/minimap/
                string fullPath = Path.Combine(baseDir!, actualSide.Replace('/', Path.DirectorySeparatorChar));
                results.Add((currentMap, tx, ty, fullPath));
            }
            return results;
        }

        private static void ExportMinimapOverlays(string runDir, List<(string mapName,int tileX,int tileY,string fullPath)> entries)
        {
            // Minimal implementation: write an index CSV to locate images and assist seed authoring.
            // Image decoding to PNG can be added later if needed.
            string dir = Path.Combine(runDir, "minimap_overlay");
            Directory.CreateDirectory(dir);
            var sb = new StringBuilder();
            sb.AppendLine("mapName,tileX,tileY,blp_path");
            foreach (var e in entries.OrderBy(e => e.mapName).ThenBy(e => e.tileY).ThenBy(e => e.tileX))
            {
                sb.AppendLine($"{e.mapName},{e.tileX},{e.tileY},\"{e.fullPath.Replace("\\", "/")}\"");
            }
            File.WriteAllText(Path.Combine(dir, "minimap_index.csv"), sb.ToString());
            System.Console.WriteLine($"Wrote minimap_index.csv with {entries.Count} rows");
        }

        private static void GenerateSeedTemplate(string runDir, string inputDir, HashSet<(int tx,int ty)> tileSet, (int x1,int y1,int x2,int y2)? tileRange)
        {
            string dir = Path.Combine(runDir, "seeding");
            Directory.CreateDirectory(dir);
            string chunkIndexPath = Path.Combine(dir, "chunk_index.csv");
            using var w = new StreamWriter(chunkIndexPath, false, Encoding.UTF8);
            w.WriteLine("tile, tileX, tileY, chunkX, chunkY, gx, gy");

            var enumOption = SearchOption.AllDirectories;
            foreach (var adtPath in Directory.EnumerateFiles(inputDir, "*.adt", enumOption))
            {
                var stem = Path.GetFileNameWithoutExtension(adtPath);
                if (!ParseTileXYFromStem(stem, out int tileX, out int tileY)) continue;
                bool includeTile = tileSet.Count == 0 && tileRange == null
                    || (tileSet.Contains((tileX, tileY)))
                    || (tileRange != null && tileX >= tileRange.Value.x1 && tileX <= tileRange.Value.x2 && tileY >= tileRange.Value.y1 && tileY <= tileRange.Value.y2);
                if (!includeTile) continue;
                for (int cy = 0; cy < 16; cy++)
                {
                    for (int cx = 0; cx < 16; cx++)
                    {
                        int gx = tileX * 16 + cx;
                        int gy = tileY * 16 + cy;
                        w.WriteLine($"{stem}, {tileX}, {tileY}, {cx}, {cy}, {gx}, {gy}");
                    }
                }
            }
            w.Flush();

            File.WriteAllText(Path.Combine(dir, "seeds_template.csv"), "gx,gy,width,height,label\n");
            System.Console.WriteLine($"Wrote seeding helpers: {chunkIndexPath} and seeds_template.csv");
        }

        private static List<(int gx,int gy,int w,int h,string label)> ReadSeedsCsv(string seedsCsvPath)
        {
            var list = new List<(int,int,int,int,string)>();
            foreach (var raw in File.ReadAllLines(seedsCsvPath))
            {
                var line = raw.Trim();
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("gx,")) continue;
                var p = line.Split(',');
                if (p.Length < 4) continue;
                if (int.TryParse(p[0], out int gx) && int.TryParse(p[1], out int gy) && int.TryParse(p[2], out int w) && int.TryParse(p[3], out int h))
                {
                    string label = p.Length > 4 ? string.Join(',', p.Skip(4)).Trim() : "";
                    list.Add((gx, gy, w, h, label));
                }
            }
            return list;
        }

        private static void ProcessSuperblockPatternsGlobal(
            string inputDir,
            string runDir,
            HashSet<(int tx,int ty)> tileSet,
            (int x1,int y1,int x2,int y2)? tileRange,
            List<(int w,int h)> sizes,
            string? seedsCsvPath,
            int edgeTrimMax,
            bool deltaExport)
        {
            System.Console.WriteLine("Superblock patterns: seeds-first workflow");
            var seeds = new List<(int gx,int gy,int w,int h,string label)>();
            if (!string.IsNullOrWhiteSpace(seedsCsvPath) && File.Exists(seedsCsvPath))
            {
                seeds = ReadSeedsCsv(seedsCsvPath!);
                System.Console.WriteLine($"Loaded {seeds.Count} seeds from {seedsCsvPath}");
            }
            else
            {
                System.Console.WriteLine("No seeds provided; currently seed-first mode expects --seeds. Skipping.");
                return;
            }

            // TODO: Build global chunk grid and token map (reuse existing ComputeChunkGrid8/feature hashing) and implement:
            //  - window signature build for each seed
            //  - candidate search with optional 8-way canonicalization
            //  - nibble alignment in range [0..edgeTrimMax] per side
            //  - similarity check and delta export (optional)

            string outDir = Path.Combine(runDir, "prefab_match_exports");
            Directory.CreateDirectory(outDir);
            File.WriteAllText(Path.Combine(outDir, "prefab_superblock_stats.json"), "{\n  \"status\": \"seed-stub\"\n}\n");
            System.Console.WriteLine("Superblock stub completed (implementation TODO)");
        }

        // === Restored minimal stubs to satisfy calls from Main() ===
        private static bool ParseTileXYFromStem(string stem, out int x, out int y)
        {
            x = 0; y = 0;
            var parts = stem.Split(new[] { '_', '-' }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = parts.Length - 1; i >= 1; i--)
            {
                if (int.TryParse(parts[i - 1], out var px) && int.TryParse(parts[i], out var py)) { x = px; y = py; return true; }
            }
            return false;
        }

        private static void ProcessADTFile(string inputPath, string outputDir, bool flagGlb, bool flagGltf)
        {
            Directory.CreateDirectory(outputDir);
            System.Console.WriteLine($"[stub] ProcessADTFile: {inputPath} -> {outputDir} (glb={flagGlb}, gltf={flagGltf})");
        }

        private static void ProcessADTDirectory(
            string inputDir,
            string outputDir,
            bool recursive,
            bool noComments,
            bool flagGlbPerFile,
            bool flagManifest,
            bool flagChunksManifest,
            bool flagMeta)
        {
            Directory.CreateDirectory(outputDir);
            System.Console.WriteLine($"[stub] ProcessADTDirectory: {inputDir} -> {outputDir} (recursive={recursive})");
        }

        private static void ProcessSimilarityDirectory(
            string inputDir,
            string outputDir,
            bool recursive,
            HashSet<(int tx, int ty)> tileSet,
            (int x1, int y1, int x2, int y2)? tileRange,
            int maxHamming)
        {
            Directory.CreateDirectory(outputDir);
            System.Console.WriteLine($"[stub] ProcessSimilarityDirectory: {inputDir} -> {outputDir}, maxHamming={maxHamming}");
        }

        private static void ProcessPrefabDirectory(
            string inputDir,
            string outputDir,
            bool recursive,
            HashSet<(int tx, int ty)> tileSet,
            (int x1, int y1, int x2, int y2)? tileRange,
            List<(int w, int h)> sizes,
            int stride,
            int maxHamming128,
            float minRuggedness,
            float minEdgeDensity,
            bool exportMatches,
            int exportMax)
        {
            Directory.CreateDirectory(outputDir);
            System.Console.WriteLine($"[stub] ProcessPrefabDirectory: {inputDir} -> {outputDir}, sizes={string.Join(',', sizes.Select(s => $"{s.w}x{s.h}"))}");
        }

        private static void ProcessPrefabDirectoryGlobal(
            string inputDir,
            string outputDir,
            bool recursive,
            HashSet<(int tx, int ty)> tileSet,
            (int x1, int y1, int x2, int y2)? tileRange,
            List<(int w, int h)> sizes,
            int stride,
            int maxHamming128,
            float minRuggedness,
            float minEdgeDensity,
            bool exportMatches,
            int exportMax)
        {
            Directory.CreateDirectory(outputDir);
            System.Console.WriteLine($"[stub] ProcessPrefabDirectoryGlobal: {inputDir} -> {outputDir}, cross-tiles enabled");
        }

        private static string? FindVersionTagFromPath(string path)
        {
            try
            {
                var dir = new DirectoryInfo(path);
                for (int i = 0; i < 6 && dir != null; i++, dir = dir.Parent)
                {
                    if (dir == null) break;
                    var name = dir.Name;
                    if (Regex.IsMatch(name, "^[0-9]+(\\.[0-9]+)+$")) return name; // e.g., 0.6.0
                }
            }
            catch { }
            return null;
        }
    }
}
