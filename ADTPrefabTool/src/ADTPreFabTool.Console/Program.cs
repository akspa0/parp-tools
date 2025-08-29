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
using SharpGLTF.Transforms;
using SharpGLTF.Memory;
using System.Text.RegularExpressions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;
using BLPSharp;

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
                System.Console.WriteLine("Usage: ADTPreFabTool.Console <adt_file_or_folder_path> [output_directory_or_root] [--recursive|--no-recursive] [--no-comments] [--glb] [--gltf] [--glb-per-file|--no-glb-per-file] [--manifest|--no-manifest] [--output-root <path>] [--timestamped|--no-timestamp] [--chunks-manifest|--no-chunks-manifest] [--meta|--no-meta] [--similarity-only] [--tiles x_y,...] [--tile-range x1,y1,x2,y2] [--max-hamming N] [--chunk-min-similarity S] [--prefab-scan] [--prefab-sizes 2x2,4x4,...] [--prefab-stride N] [--prefab-max-hamming N] [--prefab-min-similarity S] [--prefab-min-ruggedness R] [--prefab-min-edge-density E] [--prefab-cross-tiles|--no-prefab-cross-tiles] [--export-matches] [--export-prefab-matches] [--export-max N] [--superblock-patterns] [--superblock-sizes 12x12,...] [--seeds <path>] [--edge-trim-max N] [--delta-export] [--minimap-root <path>] [--trs <path>] [--export-minimap-overlays] [--generate-seed-template] [--data-version V] [--cache-root <path>] [--decode-minimap] [--world-minimap-root <path>] [--export-minimap-grid] [--yflip|--no-yflip] [--xflip|--no-xflip] [--export-chunk-selection] [--select-tile Map:tx:ty] [--select-chunks cx,cy;cx,cy] [--export-minimap-obj] [--export-minimap-glb]");
                System.Console.WriteLine("Defaults (directory input): --recursive --glb-per-file --manifest --timestamped --chunks-manifest --meta --yflip (use --no-yflip to disable)");
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
            string? worldMinimapRoot = null; // optional: World/Minimaps root for WMO-named tiles
            bool exportMinimapGrid = args.Any(a => a.Equals("--export-minimap-grid", StringComparison.OrdinalIgnoreCase));
            // yFlip defaults to true; use --no-yflip to disable
            bool yFlip = true;
            if (args.Any(a => a.Equals("--no-yflip", StringComparison.OrdinalIgnoreCase))) yFlip = false;
            else if (args.Any(a => a.Equals("--yflip", StringComparison.OrdinalIgnoreCase))) yFlip = true; // explicit opt-in (default)
            // xFlip defaults to true; use --no-xflip to disable
            bool xFlip = true;
            if (args.Any(a => a.Equals("--no-xflip", StringComparison.OrdinalIgnoreCase))) xFlip = false;
            else if (args.Any(a => a.Equals("--xflip", StringComparison.OrdinalIgnoreCase))) xFlip = true; // explicit opt-in (default)
            bool exportChunkSelection = args.Any(a => a.Equals("--export-chunk-selection", StringComparison.OrdinalIgnoreCase));
            bool exportMinimapObj = args.Any(a => a.Equals("--export-minimap-obj", StringComparison.OrdinalIgnoreCase));
            bool exportMinimapGlb = args.Any(a => a.Equals("--export-minimap-glb", StringComparison.OrdinalIgnoreCase));
            string? selectTile = null; // format: MapName:tx:ty
            string? selectChunks = null; // format: cx,cy;cx,cy

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
                if (args[i].Equals("--world-minimap-root", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    worldMinimapRoot = args[i + 1];
                }
                if (args[i].Equals("--select-tile", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    selectTile = args[i + 1];
                }
                if (args[i].Equals("--select-chunks", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    selectChunks = args[i + 1];
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

                    if (superblockPatterns || exportMinimapOverlays || generateSeedTemplate || exportMinimapGrid || exportChunkSelection || exportMinimapObj || exportMinimapGlb)
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
                                // Auto-detect World/Minimaps if not provided
                                if (string.IsNullOrWhiteSpace(worldMinimapRoot))
                                {
                                    try
                                    {
                                        // Try to locate ../../World/Minimaps relative to textures/Minimap
                                        var texturesDir = Directory.GetParent(minimapRoot!);
                                        var treeRoot = texturesDir?.Parent; // .../tree
                                        var candidate = treeRoot != null ? Path.Combine(treeRoot.FullName, "World", "Minimaps") : null;
                                        if (!string.IsNullOrEmpty(candidate) && Directory.Exists(candidate)) worldMinimapRoot = candidate;
                                    }
                                    catch { }
                                }
                            }
                            catch (Exception ex)
                            {
                                System.Console.WriteLine($"Warn: failed to parse TRS '{trsPath}': {ex.Message}");
                            }
                        }

                        // Versioned cache scaffolding + optional decoding (preserve TRS folder structure)
                        if (!string.IsNullOrWhiteSpace(minimapRoot))
                        {
                            // Prefer explicit --data-version, else infer from input folder, else minimap root
                            string versionTag = dataVersion ?? FindVersionTagFromPath(inputPath) ?? FindVersionTagFromPath(minimapRoot!) ?? "unknown";
                            string cacheBase = cacheRoot ?? outputRoot;
                            string minimapCacheDir = Path.Combine(cacheBase, "_cache", versionTag, "minimap_png");
                            Directory.CreateDirectory(minimapCacheDir);
                            // Ensure per-entry directories mirroring TRS relative paths
                            if (minimapEntries.Count > 0)
                            {
                                foreach (var e in minimapEntries)
                                {
                                    try
                                    {
                                        string rel = Path.GetRelativePath(minimapRoot!, e.fullPath);
                                        string? relDir = Path.GetDirectoryName(rel);
                                        string targetDir = string.IsNullOrEmpty(relDir) ? minimapCacheDir : Path.Combine(minimapCacheDir, relDir);
                                        Directory.CreateDirectory(targetDir);
                                    }
                                    catch { }
                                }
                            }

                            // Build unified entry list including legacy "second copy" minimaps for very old versions
                            var allEntries = new List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)>();
                            var tileHash = new Dictionary<(string,int,int), string>();
                            var duplicateOf = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase); // fullPath -> duplicateOfFullPath
                            // Seed with TRS entries first (authoritative)
                            foreach (var e in minimapEntries)
                            {
                                string md5 = ComputeFileMD5(e.fullPath);
                                tileHash[(e.mapName.ToLowerInvariant(), e.tileX, e.tileY)] = md5;
                                allEntries.Add((e.mapName, e.tileX, e.tileY, e.fullPath, "trs", false, "", md5, ""));
                            }
                            // WMO-named tiles from World/Minimaps (if available)
                            if (!string.IsNullOrWhiteSpace(worldMinimapRoot) && Directory.Exists(worldMinimapRoot))
                            {
                                foreach (var w in ScanWorldMinimaps(worldMinimapRoot!))
                                {
                                    string md5 = ComputeFileMD5(w.fullPath);
                                    // For WMO, do not dedup by (mapName,X,Y); preserve each path as its own output
                                    allEntries.Add((w.mapName, w.tileX, w.tileY, w.fullPath, "wmo", false, w.wmoAsset, md5, w.relPng));
                                }
                            }
                            // If version is one of the legacy ones, scan alt layout
                            if (versionTag == "0.5.3" || versionTag == "0.5.5" || versionTag == "0.6.0")
                            {
                                foreach (var (mapName,tileX,tileY,fullPath) in ScanAltMinimapFolders(minimapRoot!))
                                {
                                    string md5 = ComputeFileMD5(fullPath);
                                    var key = (mapName.ToLowerInvariant(),tileX,tileY);
                                    if (tileHash.TryGetValue(key, out var existing))
                                    {
                                        if (string.Equals(existing, md5, StringComparison.OrdinalIgnoreCase))
                                        {
                                            duplicateOf[fullPath] = allEntries.First(e => e.mapName==mapName && e.tileX==tileX && e.tileY==tileY).fullPath;
                                        }
                                        else
                                        {
                                            allEntries.Add((mapName,tileX,tileY,fullPath,"alt", true, "", md5, ""));
                                        }
                                    }
                                    else
                                    {
                                        tileHash[key] = md5;
                                        allEntries.Add((mapName,tileX,tileY,fullPath,"alt", false, "", md5, ""));
                                    }
                                }
                            }

                            // Decode BLP→PNG once into cache if requested
                            if (decodeMinimap && allEntries.Count > 0)
                            {
                                int decoded = 0; int skipped = 0; int failed = 0; int logged = 0;
                                foreach (var e in allEntries)
                                {
                                    try
                                    {
                                        string targetPng;
                                        if (e.sourceKind == "wmo" && !string.IsNullOrWhiteSpace(e.wmoRelPng))
                                        {
                                            // Write under dedicated wmo hierarchy using original stem
                                            var rel = e.wmoRelPng.Replace("/", Path.DirectorySeparatorChar.ToString());
                                            targetPng = Path.Combine(minimapCacheDir, rel);
                                        }
                                        else
                                        {
                                            string fileName = e.altSuffix ? $"{e.mapName}_{e.tileX}_{e.tileY}__alt.png" : $"{e.mapName}_{e.tileX}_{e.tileY}.png";
                                            targetPng = Path.Combine(minimapCacheDir, e.mapName, fileName);
                                        }
                                        string? targetDir = Path.GetDirectoryName(targetPng);
                                        if (!string.IsNullOrEmpty(targetDir)) Directory.CreateDirectory(targetDir);
                                        if (File.Exists(targetPng)) { skipped++; continue; }

                                        using (var fs = File.OpenRead(e.fullPath))
                                        using (var blp = new BLPFile(fs))
                                        {
                                            var pixels = blp.GetPixels(0, out var w, out var h);
                                            using var img = Image.LoadPixelData<Bgra32>(pixels, w, h);
                                            img.SaveAsPng(targetPng);
                                        }
                                        decoded++;
                                    }
                                    catch (Exception ex)
                                    {
                                        failed++;
                                        if (logged < 5)
                                        {
                                            System.Console.WriteLine($"Decode fail: {e.fullPath} -> {ex.Message}");
                                            logged++;
                                        }
                                    }
                                }
                                System.Console.WriteLine($"Decoded minimaps: {decoded} new, {skipped} cached, {failed} failed");
                            }

                            var manifest = new StringBuilder();
                            manifest.AppendLine("{");
                            manifest.AppendLine($"  \"version\": \"{versionTag}\",");
                            manifest.AppendLine($"  \"minimapRoot\": \"{minimapRoot!.Replace("\\", "/")}\",");
                            manifest.AppendLine($"  \"trsPath\": \"{(trsPath ?? "").Replace("\\", "/")}\",");
                            if (!string.IsNullOrWhiteSpace(worldMinimapRoot)) manifest.AppendLine($"  \"worldMinimapRoot\": \"{worldMinimapRoot.Replace("\\", "/")}\",");
                            manifest.AppendLine($"  \"decoded\": {decodeMinimap.ToString().ToLowerInvariant()},");
                            manifest.AppendLine($"  \"entries\": {(minimapEntries.Count)}");
                            // counts per source
                            var c_trs = allEntries.Count(e => e.sourceKind=="trs");
                            var c_wmo = allEntries.Count(e => e.sourceKind=="wmo");
                            var c_alt = allEntries.Count(e => e.sourceKind=="alt");
                            manifest.AppendLine($"  ,\"counts\": {{ \"trs\": {c_trs}, \"wmo\": {c_wmo}, \"alt\": {c_alt}, \"duplicates\": {duplicateOf.Count} }}");
                            manifest.AppendLine("}");
                            File.WriteAllText(Path.Combine(minimapCacheDir, "cache_manifest.json"), manifest.ToString());
                            System.Console.WriteLine($"Cache ready at: {minimapCacheDir}");

                            // Export overlays CSV with optional png_path
                            if (exportMinimapOverlays)
                            {
                                ExportMinimapOverlaysV2(runDir, minimapEntries, minimapRoot!, minimapCacheDir, decodeMinimap, allEntries, duplicateOf);
                            }

                            // Grid overlay export for ADT tiles only
                            if (exportMinimapGrid)
                            {
                                ExportGridOverlays(runDir, minimapCacheDir, allEntries, tileSet, tileRange, yFlip);
                            }

                            // Chunk selection export (stub: persist selection list)
                            if (exportChunkSelection)
                            {
                                ExportSelectedChunks(runDir, minimapCacheDir, allEntries, selectTile, selectChunks, yFlip);
                            }

                            // Minimap OBJ export: textured XZ plane with cached PNG
                            if (exportMinimapObj)
                            {
                                bool objXFlip = xFlip; // OBJ uses CLI defaults as-is
                                bool objYFlip = yFlip;
                                ExportMinimapOBJ(runDir, minimapCacheDir, inputPath, allEntries, tileSet, tileRange, objYFlip, objXFlip);
                            }

                            // Minimap GLB export: single GLB per tile, per-chunk nodes, shared minimap material
                            if (exportMinimapGlb)
                            {
                                bool glbXFlip = xFlip; // GLB may get different defaults if not explicitly set
                                bool glbYFlip = yFlip;
                                if (!args.Any(a => a.Equals("--xflip", StringComparison.OrdinalIgnoreCase) || a.Equals("--no-xflip", StringComparison.OrdinalIgnoreCase))) glbXFlip = true;   // GLB default: horizontal flip
                                if (!args.Any(a => a.Equals("--yflip", StringComparison.OrdinalIgnoreCase) || a.Equals("--no-yflip", StringComparison.OrdinalIgnoreCase))) glbYFlip = false;  // GLB default: vertical flip
                                ExportMinimapGLB(runDir, minimapCacheDir, inputPath, allEntries, tileSet, tileRange, glbYFlip, glbXFlip);
                            }
                        }

                        // Generate seed template CSV
                        if (generateSeedTemplate)
                        {
                            GenerateSeedTemplate(runDir, inputPath, tileSet, tileRange);
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

        private static void ExportGridOverlays(
            string runDir,
            string minimapCacheDir,
            List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)> allEntries,
            HashSet<(int tx,int ty)> tileSet,
            (int x1,int y1,int x2,int y2)? tileRange,
            bool yFlip)
        {
            // Group by mapName, then pick ADT tiles matching filters
            var adtTiles = allEntries
                .Where(e => e.sourceKind == "trs" || e.sourceKind == "alt")
                .GroupBy(e => (mapName: e.mapName, tileX: e.tileX, tileY: e.tileY))
                .Select(g => g.Key)
                .ToList();

            string outDir = Path.Combine(runDir, "minimap_grid_overlays");
            Directory.CreateDirectory(outDir);

            int done = 0, missing = 0;
            foreach (var (mapName, tx, ty) in adtTiles.OrderBy(t => t.mapName).ThenBy(t => t.tileX).ThenBy(t => t.tileY))
            {
                bool include = tileSet.Count == 0 && tileRange == null
                    || tileSet.Contains((tx, ty))
                    || (tileRange != null && tx >= tileRange.Value.x1 && tx <= tileRange.Value.x2 && ty >= tileRange.Value.y1 && ty <= tileRange.Value.y2);
                if (!include) continue;

                string pngMain = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}.png");
                string pngAlt = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}__alt.png");
                string src = File.Exists(pngMain) ? pngMain : (File.Exists(pngAlt) ? pngAlt : "");
                if (string.IsNullOrEmpty(src)) { missing++; continue; }

                string dst = Path.Combine(outDir, $"{mapName}_{tx}_{ty}__grid.png");
                try
                {
                    using var img = Image.Load<Bgra32>(src);
                    DrawGridInPlace(img, 16, yFlip);
                    img.SaveAsPng(dst);
                    done++;
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Grid overlay failed for {src}: {ex.Message}");
                }
            }
            System.Console.WriteLine($"Grid overlays: {done} written, {missing} missing source PNGs");
        }

        private static void DrawGridInPlace(Image<Bgra32> img, int cells, bool yFlip)
        {
            int w = img.Width;
            int h = img.Height;
            int stepX = Math.Max(1, w / cells);
            int stepY = Math.Max(1, h / cells);
            var line = new Bgra32(255, 0, 0, 255); // red

            // Vertical lines
            for (int x = 0; x <= w; x += stepX)
            {
                int xx = Math.Min(x, w - 1);
                for (int y = 0; y < h; y++) img[xx, y] = line;
            }
            // Horizontal lines
            for (int y = 0; y <= h; y += stepY)
            {
                int yy = Math.Min(y, h - 1);
                for (int x = 0; x < w; x++) img[x, yy] = line;
            }
        }

        private static void ExportSelectedChunks(
            string runDir,
            string minimapCacheDir,
            List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)> allEntries,
            string? selectTile,
            string? selectChunks,
            bool yFlip)
        {
            if (string.IsNullOrWhiteSpace(selectTile) || string.IsNullOrWhiteSpace(selectChunks))
            {
                System.Console.WriteLine("--export-chunk-selection requires --select-tile Map:tx:ty and --select-chunks cx,cy;cx,cy");
                return;
            }
            // Parse tile spec
            var tparts = selectTile.Split(':');
            if (tparts.Length != 3 || !int.TryParse(tparts[1], out var tx) || !int.TryParse(tparts[2], out var ty))
            {
                System.Console.WriteLine("Invalid --select-tile format. Use MapName:tx:ty");
                return;
            }
            string map = tparts[0];

            // Parse chunk list
            var chunks = new List<(int cx,int cy)>();
            foreach (var item in selectChunks.Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                var p = item.Split(',', StringSplitOptions.RemoveEmptyEntries);
                if (p.Length == 2 && int.TryParse(p[0], out var cx) && int.TryParse(p[1], out var cy))
                {
                    if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16) chunks.Add((cx, cy));
                }
            }
            if (chunks.Count == 0)
            {
                System.Console.WriteLine("No valid chunks parsed from --select-chunks");
                return;
            }

            // Ensure PNG exists (optional visual reference)
            string pngMain = Path.Combine(minimapCacheDir, map, $"{map}_{tx}_{ty}.png");
            string pngAlt = Path.Combine(minimapCacheDir, map, $"{map}_{tx}_{ty}__alt.png");
            string src = File.Exists(pngMain) ? pngMain : (File.Exists(pngAlt) ? pngAlt : "");
            if (string.IsNullOrEmpty(src))
            {
                System.Console.WriteLine("Warning: no cached PNG found for selected tile; proceeding with CSV export only.");
            }

            // Write a simple CSV dataset for the selection
            string outDir = Path.Combine(runDir, "chunk_selections");
            Directory.CreateDirectory(outDir);
            string csvPath = Path.Combine(outDir, $"{map}_{tx}_{ty}_selection.csv");
            using (var w = new StreamWriter(csvPath, false, Encoding.UTF8))
            {
                w.WriteLine("mapName,tileX,tileY,chunkX,chunkY");
                foreach (var (cx, cy) in chunks)
                {
                    w.WriteLine($"{map},{tx},{ty},{cx},{cy}");
                }
            }

            // Optional: produce a visualization PNG highlighting selected chunks
            if (!string.IsNullOrEmpty(src))
            {
                try
                {
                    using var img = Image.Load<Bgra32>(src);
                    HighlightChunksInPlace(img, chunks, yFlip);
                    string dst = Path.Combine(outDir, $"{map}_{tx}_{ty}_selection.png");
                    img.SaveAsPng(dst);
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Selection preview failed: {ex.Message}");
                }
            }

            // Stub: actual mesh sub-export to OBJ/GLB would map these MCNKs in the ADT file
            System.Console.WriteLine($"Selection persisted: {csvPath} (mesh export TODO)");
        }

        private static void HighlightChunksInPlace(Image<Bgra32> img, List<(int cx,int cy)> chunks, bool yFlip)
        {
            int w = img.Width, h = img.Height;
            int stepX = Math.Max(1, w / 16);
            int stepY = Math.Max(1, h / 16);
            var tint = new Bgra32(0, 255, 0, 80); // semi-transparent green

            foreach (var (cx, cy) in chunks)
            {
                int px = cx * stepX;
                int py = (yFlip ? (15 - cy) : cy) * stepY;
                for (int y = py; y < Math.Min(py + stepY, h); y++)
                {
                    for (int x = px; x < Math.Min(px + stepX, w); x++)
                    {
                        // Simple alpha blend over existing pixel
                        var p = img[x, y];
                        byte a = tint.A;
                        img[x, y] = new Bgra32(
                            (byte)((tint.R * a + p.R * (255 - a)) / 255),
                            (byte)((tint.G * a + p.G * (255 - a)) / 255),
                            (byte)((tint.B * a + p.B * (255 - a)) / 255),
                            p.A);
                    }
                }
            }
        }

        private static void ExportMinimapOverlaysV2(
            string runDir,
            List<(string mapName,int tileX,int tileY,string fullPath)> trsEntries,
            string? minimapRoot,
            string? pngCacheRoot,
            bool includePng,
            List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)> allEntries,
            Dictionary<string,string> duplicateOf)
        {
            // Write an index CSV: includes blp_path and optional png_path (if decoded/cached)
            string dir = Path.Combine(runDir, "minimap_overlay");
            Directory.CreateDirectory(dir);
            var sb = new StringBuilder();
            sb.AppendLine("mapName,tileX,tileY,source_kind,duplicate_of,wmo_asset,content_md5,blp_path,png_path");
            foreach (var e in allEntries.OrderBy(e => e.sourceKind).ThenBy(e => e.mapName).ThenBy(e => e.tileY).ThenBy(e => e.tileX))
            {
                string blp = e.fullPath.Replace("\\", "/");
                string png = "";
                if (includePng && !string.IsNullOrWhiteSpace(minimapRoot) && !string.IsNullOrWhiteSpace(pngCacheRoot))
                {
                    try
                    {
                        string targetPng;
                        if (e.sourceKind == "wmo" && !string.IsNullOrWhiteSpace(e.wmoRelPng))
                        {
                            var rel = e.wmoRelPng.Replace("/", Path.DirectorySeparatorChar.ToString());
                            targetPng = Path.Combine(pngCacheRoot!, rel);
                        }
                        else
                        {
                            string fileName = e.altSuffix ? $"{e.mapName}_{e.tileX}_{e.tileY}__alt.png" : $"{e.mapName}_{e.tileX}_{e.tileY}.png";
                            targetPng = Path.Combine(pngCacheRoot!, e.mapName, fileName);
                        }
                        if (File.Exists(targetPng)) png = targetPng.Replace("\\", "/");
                    }
                    catch { }
                }
                string dup = duplicateOf.TryGetValue(e.fullPath, out var ofp) ? ofp.Replace("\\", "/") : "";
                sb.AppendLine($"{e.mapName},{e.tileX},{e.tileY},{e.sourceKind},\"{dup}\",\"{e.wmoAsset}\",{e.md5},\"{blp}\",\"{png}\"");
            }
            File.WriteAllText(Path.Combine(dir, "minimap_index.csv"), sb.ToString());
            System.Console.WriteLine($"Wrote minimap_index.csv with {allEntries.Count} rows");
        }

        private static void GenerateSeedTemplate(string runDir, string inputPath, HashSet<(int tx,int ty)> tileSet, (int x1,int y1,int x2,int y2)? tileRange)
        {
            string dir = Path.Combine(runDir, "seeding");
            Directory.CreateDirectory(dir);
            string chunkIndexPath = Path.Combine(dir, "chunk_index.csv");
            using var w = new StreamWriter(chunkIndexPath, false, Encoding.UTF8);
            w.WriteLine("tile, tileX, tileY, chunkX, chunkY, gx, gy");

            var enumOption = SearchOption.AllDirectories;
            foreach (var adtPath in Directory.EnumerateFiles(inputPath, "*.adt", enumOption))
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

        private static IEnumerable<(string mapName,int tileX,int tileY,string fullPath,string wmoAsset,string relPng)> ScanWorldMinimaps(string worldMinimapRoot)
        {
            // Recursively enumerate .blp and infer (mapName, tileX, tileY) and wmoAsset from path
            var results = new List<(string,int,int,string,string,string)>();
            if (!Directory.Exists(worldMinimapRoot)) return results;
            foreach (var blp in Directory.EnumerateFiles(worldMinimapRoot, "*.blp", SearchOption.AllDirectories))
            {
                var rel = Path.GetRelativePath(worldMinimapRoot, blp);
                var parts = rel.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                if (parts.Length < 2) continue; // need at least top-level dir and a filename
                // If path starts with wmo/..., use the second segment (continent) as mapName; else use first segment
                string mapName = parts[0].Equals("wmo", StringComparison.OrdinalIgnoreCase)
                    ? (parts.Length > 1 ? parts[1] : "wmo")
                    : parts[0];
                string stem = Path.GetFileNameWithoutExtension(blp);
                // Parse last two numeric groups as X,Y
                var m = Regex.Match(stem, @".*_(\d+)_(\d+)$", RegexOptions.IgnoreCase);
                if (!m.Success || !int.TryParse(m.Groups[1].Value, out var tx) || !int.TryParse(m.Groups[2].Value, out var ty)) continue;
                string relDir = Path.GetDirectoryName(rel)?.Replace("\\", "/") ?? ""; // directory under World/Minimaps
                string wmoAsset = relDir; // full directory path under World/Minimaps
                // Strip leading 'wmo/' from relDir for output rooting
                string relDirNoPrefix = relDir;
                if (!string.IsNullOrEmpty(relDirNoPrefix))
                {
                    var segs = relDirNoPrefix.Split('/');
                    if (segs.Length > 0 && segs[0].Equals("wmo", StringComparison.OrdinalIgnoreCase))
                    {
                        relDirNoPrefix = string.Join('/', segs.Skip(1));
                    }
                }
                // Build relative PNG path under cache: wmo/<relDirNoPrefix>/<stem>.png
                string relPng = string.IsNullOrEmpty(relDirNoPrefix)
                    ? Path.Combine("wmo", Path.GetFileNameWithoutExtension(blp) + ".png").Replace("\\", "/")
                    : Path.Combine("wmo", relDirNoPrefix, Path.GetFileNameWithoutExtension(blp) + ".png").Replace("\\", "/");
                results.Add((mapName, tx, ty, blp, wmoAsset, relPng));
            }
            return results;
        }

        private static IEnumerable<(string mapName,int tileX,int tileY,string fullPath)> ScanAltMinimapFolders(string minimapRoot)
        {
            var results = new List<(string,int,int,string)>();
            if (!Directory.Exists(minimapRoot)) return results;
            foreach (var dir in Directory.EnumerateDirectories(minimapRoot))
            {
                string mapName = Path.GetFileName(dir);
                foreach (var blp in Directory.EnumerateFiles(dir, "*.blp", SearchOption.TopDirectoryOnly))
                {
                    var stem = Path.GetFileNameWithoutExtension(blp);
                    var m = Regex.Match(stem, "^map_(\\d+)[ _](\\d+)$", RegexOptions.IgnoreCase);
                    if (!m.Success && !Regex.IsMatch(stem, "^(.*)_(\\d+)_(\\d+)_(\\d+)$", RegexOptions.IgnoreCase)) continue;
                    var m2 = Regex.Match(stem, "^(.*)_(\\d+)_(\\d+)_(\\d+)$", RegexOptions.IgnoreCase);
                    if (m2.Success && int.TryParse(m2.Groups[3].Value, out var tx2) && int.TryParse(m2.Groups[4].Value, out var ty2))
                    {
                        results.Add((mapName, tx2, ty2, blp));
                    }
                }
            }
            return results;
        }

        private static string ComputeFileMD5(string path)
        {
            using var md5 = System.Security.Cryptography.MD5.Create();
            using var fs = File.OpenRead(path);
            var hash = md5.ComputeHash(fs);
            return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        }

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

        private static void ExportMinimapOBJ(
            string runDir,
            string minimapCacheDir,
            string inputPath,
            List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)> allEntries,
            HashSet<(int tx,int ty)> tileSet,
            (int x1,int y1,int x2,int y2)? tileRange,
            bool yFlip,
            bool xFlip)
        {
            var adtTiles = allEntries
                .Where(e => e.sourceKind == "trs" || e.sourceKind == "alt")
                .GroupBy(e => (mapName: e.mapName, tileX: e.tileX, tileY: e.tileY))
                .Select(g => g.Key)
                .OrderBy(t => t.mapName).ThenBy(t => t.tileX).ThenBy(t => t.tileY)
                .ToList();

            string outRoot = Path.Combine(runDir, "minimap_obj");
            Directory.CreateDirectory(outRoot);

            int done = 0, missingPng = 0, missingAdt = 0;
            foreach (var (mapName, tx, ty) in adtTiles)
            {
                bool include = tileSet.Count == 0 && tileRange == null
                    || tileSet.Contains((tx, ty))
                    || (tileRange != null && tx >= tileRange.Value.x1 && tx <= tileRange.Value.x2 && ty >= tileRange.Value.y1 && ty <= tileRange.Value.y2);
                if (!include) continue;

                // Locate cached minimap PNG
                string pngMain = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}.png");
                string pngAlt = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}__alt.png");
                string srcPng = File.Exists(pngMain) ? pngMain : (File.Exists(pngAlt) ? pngAlt : "");
                if (string.IsNullOrEmpty(srcPng)) { missingPng++; /* we'll still export geometry without texture */ }

                // Find matching ADT file under the input map dir (recursive)
                string? adtPath = null;
                try
                {
                    var enumOption = SearchOption.AllDirectories;
                    string pattern = $"{mapName}_{tx}_{ty}.adt";
                    foreach (var f in Directory.EnumerateFiles(inputPath, "*.adt", enumOption))
                    {
                        if (string.Equals(Path.GetFileName(f), pattern, StringComparison.OrdinalIgnoreCase)) { adtPath = f; break; }
                    }
                }
                catch { }

                if (string.IsNullOrWhiteSpace(adtPath) || !File.Exists(adtPath)) { missingAdt++; continue; }

                string outDir = Path.Combine(outRoot, mapName);
                Directory.CreateDirectory(outDir);
                string baseName = $"{mapName}_{tx}_{ty}";
                string objPath = Path.Combine(outDir, baseName + ".obj");
                string mtlPath = Path.Combine(outDir, baseName + ".mtl");
                string texFileName = baseName + ".png";
                string texOutPath = Path.Combine(outDir, texFileName);

                // Copy texture next to OBJ for simple relative referencing (if present)
                if (!string.IsNullOrEmpty(srcPng)) { try { if (!File.Exists(texOutPath)) File.Copy(srcPng, texOutPath, false); } catch { } }

                // Prepare MTL (always write; mesh can be viewed untextured if no PNG)
                var mtl = new StringBuilder();
                mtl.AppendLine("# Minimap material");
                mtl.AppendLine("newmtl Minimap");
                mtl.AppendLine("Kd 1.000 1.000 1.000");
                if (!string.IsNullOrEmpty(srcPng)) mtl.AppendLine($"map_Kd {texFileName}");
                File.WriteAllText(mtlPath, mtl.ToString());

                // Read ADT and emit real terrain mesh with UVs normalized to tile extents
                var adtReader = new ADTReader();
                using (var stream = File.OpenRead(adtPath))
                {
                    adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                }
                var adt = adtReader.adtfile;

                // First pass: compute vertex positions and bounds (X/Z) to create UVs
                const float TILE_SIZE = 533.333333f;
                const float CHUNK_SIZE = TILE_SIZE / 16f;
                const float UNIT_SIZE = CHUNK_SIZE / 8f;
                const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

                var positions = new List<(float x, float y, float z)>();
                var chunkStartIndex = new List<int>(); // starting vertex index per chunk
                float minX = float.PositiveInfinity, maxX = float.NegativeInfinity;
                float minZ = float.PositiveInfinity, maxZ = float.NegativeInfinity;

                foreach (var chunk in adt.chunks)
                {
                    if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
                    {
                        chunkStartIndex.Add(positions.Count);
                        continue;
                    }
                    chunkStartIndex.Add(positions.Count);

                    int idx = 0;
                    for (int row = 0; row < 17; row++)
                    {
                        bool isShort = (row % 2) == 1;
                        int colCount = isShort ? 8 : 9;
                        for (int col = 0; col < colCount; col++)
                        {
                            float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                            if (isShort) vx -= UNIT_SIZE_HALF;
                            float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                            float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                            positions.Add((vx, vy, vz));
                            if (vx < minX) minX = vx; if (vx > maxX) maxX = vx;
                            if (vz < minZ) minZ = vz; if (vz > maxZ) maxZ = vz;
                            idx++;
                        }
                    }
                }

                float spanX = Math.Max(1e-6f, maxX - minX);
                float spanZ = Math.Max(1e-6f, maxZ - minZ);

                // Prepare UVs from original positions (texture stays as-is)
                var uvs = new List<(float u, float v)>(positions.Count);
                for (int i = 0; i < positions.Count; i++)
                {
                    var p = positions[i];
                    // WoW coords: X (north), Y (west). Our p: x=worldY, z=worldX.
                    // Minimap u increases to the right (east) -> increase with worldY (west positive)
                    float u = (p.x - minX) / spanX; 
                    // Minimap v increases downward -> decrease with worldX (north positive)
                    float v = (maxZ - p.z) / spanZ; 
                    if (yFlip) v = 1f - v;
                    if (xFlip) u = 1f - u;
                    uvs.Add((u, v));
                }

                // Write OBJ with v, vt and faces (respecting holes)
                using (var fs = new FileStream(objPath, FileMode.Create, FileAccess.Write, FileShare.Read))
                using (var writer = new StreamWriter(fs))
                {
                    writer.WriteLine("# ADT Terrain Mesh (Textured with minimap)");
                    writer.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");
                    writer.WriteLine("usemtl Minimap");

                    // v and vt in same order so indices match
                    for (int i = 0; i < positions.Count; i++)
                    {
                        var p = positions[i];
                        writer.WriteLine($"v {p.z:F6} {p.x:F6} {p.y:F6}");
                    }
                    for (int i = 0; i < uvs.Count; i++)
                    {
                        var t = uvs[i];
                        writer.WriteLine($"vt {t.u.ToString(CultureInfo.InvariantCulture)} {t.v.ToString(CultureInfo.InvariantCulture)}");
                    }

                    int totalVertices = 0;
                    int chunkIndex = 0;
                    foreach (var chunk in adt.chunks)
                    {
                        if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0) { chunkIndex++; continue; }

                        // Faces: follow the same indexing pattern as old AppendADTMesh
                        for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
                        {
                            if (xx >= 8) { xx = 0; yy++; }

                            bool isHole = true;
                            if (((uint)chunk.header.flags & 0x10000u) == 0)
                            {
                                int current = 1 << ((xx / 2) + (yy / 2) * 4);
                                if ((chunk.header.holesLowRes & current) == 0) isHole = false;
                            }
                            else
                            {
                                byte holeByte = yy switch
                                {
                                    0 => chunk.header.holesHighRes_0,
                                    1 => chunk.header.holesHighRes_1,
                                    2 => chunk.header.holesHighRes_2,
                                    3 => chunk.header.holesHighRes_3,
                                    4 => chunk.header.holesHighRes_4,
                                    5 => chunk.header.holesHighRes_5,
                                    6 => chunk.header.holesHighRes_6,
                                    _ => chunk.header.holesHighRes_7,
                                };
                                if (((holeByte >> xx) & 1) == 0) isHole = false;
                            }

                            if (!isHole)
                            {
                                int baseIndex = chunkStartIndex[chunkIndex];
                                int i0 = j;
                                int a = baseIndex + i0 + 1;
                                int b = baseIndex + (i0 - 9) + 1;
                                int c = baseIndex + (i0 + 8) + 1;
                                int d = baseIndex + (i0 - 8) + 1;
                                int e = baseIndex + (i0 + 9) + 1;
                                // use v/vt same indices
                                writer.WriteLine($"f {a}/{a} {b}/{b} {c}/{c}");
                                writer.WriteLine($"f {a}/{a} {d}/{d} {b}/{b}");
                                writer.WriteLine($"f {a}/{a} {e}/{e} {d}/{d}");
                                writer.WriteLine($"f {a}/{a} {c}/{c} {e}/{e}");
                            }

                            if (((j + 1) % (9 + 8)) == 0) j += 9;
                        }

                        totalVertices += 145;
                        chunkIndex++;
                    }
                }

                done++;
            }
            System.Console.WriteLine($"Minimap OBJ export (terrain mesh): {done} written, {missingPng} missing PNGs, {missingAdt} missing ADTs");
        }

        private static void ExportMinimapGLB(
            string runDir,
            string minimapCacheDir,
            string inputPath,
            List<(string mapName,int tileX,int tileY,string fullPath,string sourceKind,bool altSuffix,string wmoAsset,string md5,string wmoRelPng)> allEntries,
            HashSet<(int tx,int ty)> tileSet,
            (int x1,int y1,int x2,int y2)? tileRange,
            bool yFlip,
            bool xFlip)
        {
            var adtTiles = allEntries
                .Where(e => e.sourceKind == "trs" || e.sourceKind == "alt")
                .GroupBy(e => (mapName: e.mapName, tileX: e.tileX, tileY: e.tileY))
                .Select(g => g.Key)
                .OrderBy(t => t.mapName).ThenBy(t => t.tileX).ThenBy(t => t.tileY)
                .ToList();

            string outRoot = Path.Combine(runDir, "minimap_glb");
            Directory.CreateDirectory(outRoot);

            int done = 0, missingPng = 0, missingAdt = 0;
            foreach (var (mapName, tx, ty) in adtTiles)
            {
                bool include = tileSet.Count == 0 && tileRange == null
                    || tileSet.Contains((tx, ty))
                    || (tileRange != null && tx >= tileRange.Value.x1 && tx <= tileRange.Value.x2 && ty >= tileRange.Value.y1 && ty <= tileRange.Value.y2);
                if (!include) continue;

                // Locate cached minimap PNG
                string pngMain = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}.png");
                string pngAlt = Path.Combine(minimapCacheDir, mapName, $"{mapName}_{tx}_{ty}__alt.png");
                string srcPng = File.Exists(pngMain) ? pngMain : (File.Exists(pngAlt) ? pngAlt : "");
                if (string.IsNullOrEmpty(srcPng)) { missingPng++; }

                // Find matching ADT file under the input map dir (recursive)
                string? adtPath = null;
                try
                {
                    var enumOption = SearchOption.AllDirectories;
                    string pattern = $"{mapName}_{tx}_{ty}.adt";
                    foreach (var f in Directory.EnumerateFiles(inputPath, "*.adt", enumOption))
                    {
                        if (string.Equals(Path.GetFileName(f), pattern, StringComparison.OrdinalIgnoreCase)) { adtPath = f; break; }
                    }
                }
                catch { }
                if (string.IsNullOrWhiteSpace(adtPath) || !File.Exists(adtPath)) { missingAdt++; continue; }

                string outDir = Path.Combine(outRoot, mapName);
                Directory.CreateDirectory(outDir);
                string baseName = $"{mapName}_{tx}_{ty}";
                string glbPath = Path.Combine(outDir, baseName + ".glb");

                // Read ADT
                var adtReader = new ADTReader();
                using (var stream = File.OpenRead(adtPath))
                {
                    adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                }
                var adt = adtReader.adtfile;

                // First pass: collect per-chunk positions and global bounds for UVs
                const float TILE_SIZE = 533.333333f;
                const float CHUNK_SIZE = TILE_SIZE / 16f;
                const float UNIT_SIZE = CHUNK_SIZE / 8f;
                const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

                var perChunkPositions = new List<(int ix,int iy,List<(float x,float y,float z)> verts)>();
                float minX = float.PositiveInfinity, maxX = float.NegativeInfinity;
                float minZ = float.PositiveInfinity, maxZ = float.NegativeInfinity;

                foreach (var chunk in adt.chunks)
                {
                    if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0) continue;
                    var verts = new List<(float x,float y,float z)>(145);
                    int idx = 0;
                    for (int row = 0; row < 17; row++)
                    {
                        bool isShort = (row % 2) == 1;
                        int colCount = isShort ? 8 : 9;
                        for (int col = 0; col < colCount; col++)
                        {
                            float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                            if (isShort) vx -= UNIT_SIZE_HALF;
                            float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                            float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                            verts.Add((vx, vy, vz));
                            if (vx < minX) minX = vx; if (vx > maxX) maxX = vx;
                            if (vz < minZ) minZ = vz; if (vz > maxZ) maxZ = vz;
                            idx++;
                        }
                        if (isShort) { /* conceptual pad to keep 9 per row alignment for indexing */ verts.Add((float.NaN,float.NaN,float.NaN)); }
                    }
                    perChunkPositions.Add(((int)chunk.header.indexX, (int)chunk.header.indexY, verts));
                }

                float spanX = Math.Max(1e-6f, maxX - minX);
                float spanZ = Math.Max(1e-6f, maxZ - minZ);

                // Build scene and material
                var scene = new SceneBuilder();
                var mat = new MaterialBuilder("Minimap").WithMetallicRoughness();
                try
                {
                    if (!string.IsNullOrEmpty(srcPng))
                    {
                        var bytes = File.ReadAllBytes(srcPng);
                        var chan = mat.UseChannel(KnownChannel.BaseColor);
                        chan.UseTexture().WithPrimaryImage((ImageBuilder)new MemoryImage(bytes));
                    }
                }
                catch
                {
                    // fall back to untextured if any issue occurs
                }

                // For each chunk, create a node and mesh
                foreach (var (ix, iy, verts) in perChunkPositions)
                {
                    var mesh = new MeshBuilder<VertexPosition, VertexTexture1, VertexEmpty>($"MCNK_{ix}_{iy}");
                    var prim = mesh.UsePrimitive(mat);

                    // Generate UVs parallel to verts list (skip padded NaNs)
                    var texcoords = new List<(float u,float v)>(verts.Count);
                    foreach (var p in verts)
                    {
                        if (float.IsNaN(p.x)) { texcoords.Add((0,0)); continue; }
                        float u = (p.x - minX) / spanX;
                        float v = (maxZ - p.z) / spanZ;
                        if (yFlip) v = 1f - v;
                        if (xFlip) u = 1f - u;
                        texcoords.Add((u, v));
                    }

                    // Emit faces identical to OBJ path (skip holes)
                    // We need access to the corresponding ADT chunk again; fetch by indices
                    var chunk = adt.chunks.FirstOrDefault(c => (int)c.header.indexX == ix && (int)c.header.indexY == iy);
                    if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
                    {
                        continue;
                    }

                    for (int j = 9, xx = 0, yy = 0; j < 145; j++, xx++)
                    {
                        if (xx >= 8) { xx = 0; yy++; }

                        bool isHole = true;
                        if (((uint)chunk.header.flags & 0x10000u) == 0)
                        {
                            int current = 1 << ((xx / 2) + (yy / 2) * 4);
                            if ((chunk.header.holesLowRes & current) == 0) isHole = false;
                        }
                        else
                        {
                            byte holeByte = yy switch
                            {
                                0 => chunk.header.holesHighRes_0,
                                1 => chunk.header.holesHighRes_1,
                                2 => chunk.header.holesHighRes_2,
                                3 => chunk.header.holesHighRes_3,
                                4 => chunk.header.holesHighRes_4,
                                5 => chunk.header.holesHighRes_5,
                                6 => chunk.header.holesHighRes_6,
                                _ => chunk.header.holesHighRes_7,
                            };
                            if (((holeByte >> xx) & 1) == 0) isHole = false;
                        }

                        if (!isHole)
                        {
                            int baseIndex = 0;
                            int i0 = j;
                            int a = baseIndex + i0 + 1;
                            int b = baseIndex + (i0 - 9) + 1;
                            int c = baseIndex + (i0 + 8) + 1;
                            int d = baseIndex + (i0 - 8) + 1;
                            int e = baseIndex + (i0 + 9) + 1;
                            // use v/vt same indices
                            var va = new VertexBuilder<VertexPosition, VertexTexture1, VertexEmpty>(new VertexPosition(new Vector3(verts[a - 1].z, verts[a - 1].x, verts[a - 1].y)), new VertexTexture1(new Vector2(texcoords[a - 1].u, texcoords[a - 1].v)));
                            var vb = new VertexBuilder<VertexPosition, VertexTexture1, VertexEmpty>(new VertexPosition(new Vector3(verts[b - 1].z, verts[b - 1].x, verts[b - 1].y)), new VertexTexture1(new Vector2(texcoords[b - 1].u, texcoords[b - 1].v)));
                            var vc = new VertexBuilder<VertexPosition, VertexTexture1, VertexEmpty>(new VertexPosition(new Vector3(verts[c - 1].z, verts[c - 1].x, verts[c - 1].y)), new VertexTexture1(new Vector2(texcoords[c - 1].u, texcoords[c - 1].v)));
                            var vd = new VertexBuilder<VertexPosition, VertexTexture1, VertexEmpty>(new VertexPosition(new Vector3(verts[d - 1].z, verts[d - 1].x, verts[d - 1].y)), new VertexTexture1(new Vector2(texcoords[d - 1].u, texcoords[d - 1].v)));
                            var ve = new VertexBuilder<VertexPosition, VertexTexture1, VertexEmpty>(new VertexPosition(new Vector3(verts[e - 1].z, verts[e - 1].x, verts[e - 1].y)), new VertexTexture1(new Vector2(texcoords[e - 1].u, texcoords[e - 1].v)));

                            prim.AddTriangle(va, vb, vc);
                            prim.AddTriangle(va, vd, vb);
                            prim.AddTriangle(va, ve, vd);
                            prim.AddTriangle(va, vc, ve);
                        }

                        if (((j + 1) % (9 + 8)) == 0) j += 9;
                    }

                    scene.AddRigidMesh(mesh, AffineTransform.Identity);
                }

                // Save GLB
                var model = scene.ToGltf2();
                model.SaveGLB(glbPath);
                done++;
            }
            System.Console.WriteLine($"Minimap GLB export: {done} written, {missingPng} missing PNGs, {missingAdt} missing ADTs");
        }
    }
}
