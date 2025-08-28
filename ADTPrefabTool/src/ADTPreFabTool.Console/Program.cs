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
                System.Console.WriteLine("Usage: ADTPreFabTool.Console <adt_file_or_folder_path> [output_directory_or_root] [--recursive|--no-recursive] [--no-comments] [--glb] [--gltf] [--glb-per-file|--no-glb-per-file] [--manifest|--no-manifest] [--output-root <path>] [--timestamped|--no-timestamp] [--chunks-manifest|--no-chunks-manifest] [--meta|--no-meta] [--similarity-only] [--tiles x_y,...] [--tile-range x1,y1,x2,y2] [--max-hamming N] [--chunk-min-similarity S] [--prefab-scan] [--prefab-sizes 2x2,4x4,...] [--prefab-stride N] [--prefab-max-hamming N] [--prefab-min-similarity S] [--prefab-min-ruggedness R] [--prefab-min-edge-density E] [--prefab-cross-tiles|--no-prefab-cross-tiles] [--export-matches] [--export-prefab-matches] [--export-max N]");
                System.Console.WriteLine("Defaults (directory input): --recursive --glb-per-file --manifest --timestamped --chunks-manifest --meta");
                System.Console.WriteLine("Examples:");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/terrain.adt\" \"output/\" --glb");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --output-root project_output --timestamped");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --no-recursive --no-manifest --no-glb-per-file");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --similarity-only --tile-range 30,41,32,43 --max-hamming 6");
                System.Console.WriteLine("  ADTPreFabTool.Console \"path/to/map_folder\" --prefab-scan --prefab-sizes 2x2,4x4 --prefab-max-hamming 12 --export-prefab-matches --export-max 50");
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

                    if (prefabScan)
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

        private static void ProcessSimilarityDirectory(string inputDir, string outputDir, bool recursive, HashSet<(int tx, int ty)> tileSet, (int x1, int y1, int x2, int y2)? tileRange, int maxHamming)
        {
            System.Console.WriteLine($"Similarity-only analysis on: {inputDir} (recursive={recursive})");
            var enumOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            var allChunks = new List<ChunkSigRow>(50000);

            foreach (var adtPath in Directory.EnumerateFiles(inputDir, "*.adt", enumOption))
            {
                try
                {
                    var stem = Path.GetFileNameWithoutExtension(adtPath);
                    int tileX = 0, tileY = 0;
                    ParseTileXYFromStem(stem, out tileX, out tileY);
                    bool includeTile = tileSet.Count == 0 && tileRange == null
                        || (tileSet.Contains((tileX, tileY)))
                        || (tileRange != null && tileX >= tileRange.Value.x1 && tileX <= tileRange.Value.x2 && tileY >= tileRange.Value.y1 && tileY <= tileRange.Value.y2);

                    var adtReader = new ADTReader();
                    using (var stream = File.OpenRead(adtPath)) adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                    var adt = adtReader.adtfile;

                    string tileDir = Path.Combine(outputDir, stem);
                    Directory.CreateDirectory(tileDir);
                    string sigsPathAbs = Path.Combine(tileDir, "chunk_signatures.ndjson");
                    if (File.Exists(sigsPathAbs)) File.Delete(sigsPathAbs);

                    int idx = 0;
                    foreach (var c in adt.chunks)
                    {
                        if (c.vertices.vertices == null || c.vertices.vertices.Length == 0) { idx++; continue; }
                        var sig = ComputeChunkSignature(c, out float mean, out int rotCode, out float heightRange, out float avgSlope);
                        int ix = (int)c.header.indexX;
                        int iy = (int)c.header.indexY;
                        string row = "{\"tile\":\"" + stem + "\",\"tileX\":" + tileX
                                     + ",\"tileY\":" + tileY
                                     + ",\"indexX\":" + ix
                                     + ",\"indexY\":" + iy
                                     + ",\"sig64\":\"" + sig.ToString("X16") + "\""
                                     + ",\"sigMean\":" + mean.ToString("F4", CultureInfo.InvariantCulture)
                                     + ",\"rot\":" + rotCode
                                     + ",\"heightRange\":" + heightRange.ToString("F3", CultureInfo.InvariantCulture)
                                     + ",\"avgSlope\":" + avgSlope.ToString("F3", CultureInfo.InvariantCulture)
                                     + "}";
                        File.AppendAllText(sigsPathAbs, row + Environment.NewLine);
                        if (includeTile)
                        {
                            allChunks.Add(new ChunkSigRow { Tile = stem, TileX = tileX, TileY = tileY, IndexX = ix, IndexY = iy, Sig64 = sig });
                        }
                        idx++;
                    }
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Warn: similarity pass failed for {adtPath}: {ex.Message}");
                }
            }

            // Pairwise compare within filtered set using bucket on top 16 bits
            var buckets = new Dictionary<ushort, List<int>>();
            for (int i = 0; i < allChunks.Count; i++)
            {
                ushort key = (ushort)(allChunks[i].Sig64 >> 48);
                if (!buckets.TryGetValue(key, out var list)) { list = new List<int>(); buckets[key] = list; }
                list.Add(i);
            }

            string pairsPath = Path.Combine(outputDir, "similar_pairs.ndjson");
            if (File.Exists(pairsPath)) File.Delete(pairsPath);
            var freq = new Dictionary<ulong, int>();
            int pairCount = 0;
            foreach (var kv in buckets)
            {
                var idxs = kv.Value;
                for (int a = 0; a < idxs.Count; a++)
                {
                    var ca = allChunks[idxs[a]];
                    for (int b = a + 1; b < idxs.Count; b++)
                    {
                        var cb = allChunks[idxs[b]];
                        int hd = PopCount(ca.Sig64 ^ cb.Sig64);
                        if (hd <= maxHamming)
                        {
                            string row = "{\"tileA\":\"" + ca.Tile + "\",\"xA\":" + ca.IndexX
                                         + ",\"yA\":" + ca.IndexY
                                         + ",\"tileB\":\"" + cb.Tile + "\",\"xB\":" + cb.IndexX
                                         + ",\"yB\":" + cb.IndexY
                                         + ",\"hamming\":" + hd + "}";
                            File.AppendAllText(pairsPath, row + Environment.NewLine);
                            pairCount++;
                            freq[ca.Sig64] = freq.TryGetValue(ca.Sig64, out var f1) ? f1 + 1 : 1;
                            freq[cb.Sig64] = freq.TryGetValue(cb.Sig64, out var f2) ? f2 + 1 : 1;
                        }
                    }
                }
            }

            // Report
            var report = new StringBuilder();
            report.AppendLine("Chunk Similarity Report");
            report.AppendLine("=======================");
            report.AppendLine($"Filtered chunks: {allChunks.Count}, pairs (<= {maxHamming}): {pairCount}");
            if (tileSet.Count > 0) report.AppendLine($"Tiles: {string.Join(',', tileSet.Select(t => $"{t.tx}_{t.ty}"))}");
            if (tileRange != null) report.AppendLine($"Tile range: {tileRange.Value.x1},{tileRange.Value.y1}..{tileRange.Value.x2},{tileRange.Value.y2}");
            report.AppendLine();
            report.AppendLine("Top repeating signatures:");
            foreach (var kv2 in freq.OrderByDescending(k => k.Value).Take(50))
            {
                report.AppendLine($"  {kv2.Key:X16}: {kv2.Value} matches");
            }
            File.WriteAllText(Path.Combine(outputDir, "similarity_report.txt"), report.ToString());
            System.Console.WriteLine("Similarity analysis completed.");
        }

        private struct ChunkSigRow { public string Tile; public int TileX, TileY, IndexX, IndexY; public ulong Sig64; }

        private static ulong ComputeChunkSignature(MCNK chunk, out float sigMean, out int rotCode, out float heightRange, out float avgSlope)
        {
            // Metrics
            float minH = chunk.vertices.vertices.Min();
            float maxH = chunk.vertices.vertices.Max();
            heightRange = maxH - minH;
            float totalSlope = 0; int cnt = 0;
            for (int i = 0; i < chunk.vertices.vertices.Length - 1; i++) { totalSlope += Math.Abs(chunk.vertices.vertices[i + 1] - chunk.vertices.vertices[i]); cnt++; }
            avgSlope = cnt > 0 ? totalSlope / cnt : 0f;

            // Normalize heights by robust scale
            var hs = chunk.vertices.vertices.ToArray();
            Array.Sort(hs);
            float p05 = hs[(int)(hs.Length * 0.05f)];
            float p95 = hs[(int)(hs.Length * 0.95f)];
            float med = hs[hs.Length / 2];
            float scale = Math.Max(1e-3f, p95 - p05);

            // Build conceptual 17x9 grid with odd rows 8 samples offset; fill NaNs in last col for short rows
            float[,] grid = new float[17, 9];
            bool[,] has = new bool[17, 9];
            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool shortRow = (row % 2) == 1;
                int cols = shortRow ? 8 : 9;
                for (int col = 0; col < cols; col++)
                {
                    float h = chunk.vertices.vertices[idx++];
                    h = MathF.Max(-1f, MathF.Min(1f, (h - med) / scale));
                    int c = col;
                    grid[row, c] = h; has[row, c] = true;
                }
                if (shortRow)
                {
                    // Mark last cell missing
                    has[row, 8] = false;
                }
            }
            // Inpaint missing by nearest neighbor from same row
            for (int r = 0; r < 17; r++) if (!has[r, 8]) { grid[r, 8] = grid[r, 7]; has[r, 8] = true; }

            // Resample to 8x8 by bilinear over [0..16]x[0..8]
            float[,] g8 = new float[8, 8];
            float sum = 0;
            for (int y = 0; y < 8; y++)
            {
                float srcY = (y + 0.5f) * (16f / 8f); // centers
                int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, 16);
                int y1 = Math.Clamp(y0 + 1, 0, 16);
                float ty = srcY - y0;
                for (int x = 0; x < 8; x++)
                {
                    float srcX = (x + 0.5f) * (8f / 8f); // 0.5..8.5
                    int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, 8);
                    int x1 = Math.Clamp(x0 + 1, 0, 8);
                    float tx = srcX - x0;
                    float v00 = grid[y0, x0];
                    float v10 = grid[y0, x1];
                    float v01 = grid[y1, x0];
                    float v11 = grid[y1, x1];
                    float v0 = v00 + (v10 - v00) * tx;
                    float v1 = v01 + (v11 - v01) * tx;
                    float v = v0 + (v1 - v0) * ty;
                    g8[y, x] = v;
                    sum += v;
                }
            }
            sigMean = sum / 64f;
            // Binary hash
            ulong raw = 0UL; int bit = 0;
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++, bit++) if (g8[y, x] > sigMean) raw |= (1UL << bit);

            // Canonicalize under 8 symmetries (rot 0..3 and mirror X)
            ulong best = raw; int bestCode = 0;
            for (int code = 1; code < 8; code++)
            {
                ulong t = TransformBitboard(raw, code);
                if (t < best) { best = t; bestCode = code; }
            }
            rotCode = bestCode;
            return best;
        }

        private static ulong TransformBitboard(ulong bb, int code)
        {
            // code 0..3: rotate 0,90,180,270; +4 means mirror X before rotate 0..3
            bool mirror = (code & 0x4) != 0; int rot = code & 0x3;
            ulong b = mirror ? MirrorX(bb) : bb;
            for (int i = 0; i < rot; i++) b = Rotate90(b);
            return b;
        }
        private static ulong MirrorX(ulong bb)
        {
            ulong res = 0UL;
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                {
                    int s = y * 8 + x;
                    int dx = 7 - x; int d = y * 8 + dx;
                    if (((bb >> s) & 1UL) != 0) res |= (1UL << d);
                }
            return res;
        }
        private static ulong Rotate90(ulong bb)
        {
            // rotate clockwise: (x,y)->(7-y,x)
            ulong res = 0UL;
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                {
                    int s = y * 8 + x;
                    int dx = 7 - y, dy = x;
                    int d = dy * 8 + dx;
                    if (((bb >> s) & 1UL) != 0) res |= (1UL << d);
                }
            return res;
        }
        private static int PopCount(ulong v)
        {
            // builtin if available: System.Numerics.BitOperations.PopCount
            return System.Numerics.BitOperations.PopCount(v);
        }

        // ===== Prefab Scan =====
        private struct BlockSig
        {
            public string Tile;
            public int TileX, TileY;
            public int OriginX, OriginY; // chunk origin
            public int W, H; // in chunks
            public ulong H0; // height channel
            public ulong H1; // gradient channel
            public float Ruggedness;
            public float EdgeDensity;
        }

        private struct BlockSigGlobal
        {
            public int OriginGX, OriginGY; // global chunk origin
            public int W, H; // in chunks
            public ulong H0; // height channel
            public ulong H1; // gradient channel
            public float Ruggedness;
            public float EdgeDensity;
            public (int minTX, int minTY, int maxTX, int maxTY) TileSpan;
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
            System.Console.WriteLine($"Prefab scan on: {inputDir} (recursive={recursive})");
            var enumOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

            var allBlocks = new List<BlockSig>(50000);
            int exported = 0;
            string exportDir = Path.Combine(outputDir, "prefab_match_exports");
            if (exportMatches) Directory.CreateDirectory(exportDir);

            foreach (var adtPath in Directory.EnumerateFiles(inputDir, "*.adt", enumOption))
            {
                try
                {
                    var stem = Path.GetFileNameWithoutExtension(adtPath);
                    int tileX = 0, tileY = 0; ParseTileXYFromStem(stem, out tileX, out tileY);
                    bool includeTile = tileSet.Count == 0 && tileRange == null
                        || (tileSet.Contains((tileX, tileY)))
                        || (tileRange != null && tileX >= tileRange.Value.x1 && tileX <= tileRange.Value.x2 && tileY >= tileRange.Value.y1 && tileY <= tileRange.Value.y2);

                    var adtReader = new ADTReader();
                    using (var stream = File.OpenRead(adtPath)) adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                    var adt = adtReader.adtfile;

                    // Index chunks by (ix,iy) and precompute per-chunk 8x8 normalized grid
                    var chunkMap = new Dictionary<(int ix, int iy), MCNK>();
                    var grid8 = new Dictionary<(int ix, int iy), float[,]>();
                    foreach (var c in adt.chunks)
                    {
                        if (c.vertices.vertices == null || c.vertices.vertices.Length == 0) continue;
                        int ix = (int)c.header.indexX; int iy = (int)c.header.indexY;
                        chunkMap[(ix, iy)] = c;
                        grid8[(ix, iy)] = ComputeChunkGrid8(c);
                    }

                    if (includeTile)
                    {
                        foreach (var sz in sizes)
                        {
                            int W = sz.w, H = sz.h;
                            for (int oy = 0; oy <= 16 - H; oy += stride)
                            {
                                for (int ox = 0; ox <= 16 - W; ox += stride)
                                {
                                    // require full block present
                                    bool full = true;
                                    for (int by = 0; by < H && full; by++)
                                        for (int bx = 0; bx < W && full; bx++)
                                            if (!grid8.ContainsKey((ox + bx, oy + by))) full = false;
                                    if (!full) continue;

                                    // tile chunk 8x8 grids into big patch (H*8 x W*8)
                                    int PH = H * 8, PW = W * 8;
                                    float[,] patch = new float[PH, PW];
                                    for (int by = 0; by < H; by++)
                                    {
                                        for (int bx = 0; bx < W; bx++)
                                        {
                                            var g = grid8[(ox + bx, oy + by)];
                                            for (int yy = 0; yy < 8; yy++)
                                                for (int xx = 0; xx < 8; xx++)
                                                    patch[by * 8 + yy, bx * 8 + xx] = g[yy, xx];
                                        }
                                    }

                                    // compute gradient magnitude and roughness
                                    float[,] grad = new float[PH, PW];
                                    float gradSum = 0f, gradMax = 1e-6f;
                                    for (int y = 1; y < PH - 1; y++)
                                    {
                                        for (int x = 1; x < PW - 1; x++)
                                        {
                                            float gx = (patch[y, x + 1] - patch[y, x - 1]) * 0.5f;
                                            float gy = (patch[y + 1, x] - patch[y - 1, x]) * 0.5f;
                                            float g = MathF.Sqrt(gx * gx + gy * gy);
                                            grad[y, x] = g; gradSum += g; if (g > gradMax) gradMax = g;
                                        }
                                    }
                                    float edgeDensity = 0f;
                                    int edgeCount = 0, totalCount = (PH - 2) * (PW - 2);
                                    float edgeThresh = 0.3f * gradMax; // robust threshold
                                    for (int y = 1; y < PH - 1; y++)
                                        for (int x = 1; x < PW - 1; x++)
                                            if (grad[y, x] >= edgeThresh) edgeCount++;
                                    edgeDensity = totalCount > 0 ? (float)edgeCount / totalCount : 0f;

                                    // ruggedness as normalized mean gradient
                                    float ruggedness = totalCount > 0 ? (gradSum / totalCount) / (gradMax + 1e-6f) : 0f;

                                    if (ruggedness < minRuggedness || edgeDensity < minEdgeDensity) continue; // suppress flats

                                    // downsample to 16x16 channels
                                    float[,] h16 = DownsampleBilinear(patch, PH, PW, 16, 16);
                                    float[,] g16 = DownsampleBilinear(grad, PH, PW, 16, 16);

                                    // normalize channels
                                    float hMean = 0, gMean = 0; for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) { hMean += h16[y, x]; gMean += g16[y, x]; }
                                    hMean /= 256f; gMean /= 256f;

                                    // 64-bit aHash each
                                    ulong hBits = 0UL, gBits = 0UL; int bit = 0;
                                    for (int y = 0; y < 16; y += 2)
                                    {
                                        for (int x = 0; x < 16; x += 2)
                                        {
                                            // average 2x2 to 1 bit placement
                                            float ha = (h16[y, x] + h16[y, x + 1] + h16[y + 1, x] + h16[y + 1, x + 1]) * 0.25f;
                                            float ga = (g16[y, x] + g16[y, x + 1] + g16[y + 1, x] + g16[y + 1, x + 1]) * 0.25f;
                                            if (ha > hMean) hBits |= (1UL << bit);
                                            if (ga > gMean) gBits |= (1UL << bit);
                                            bit++;
                                        }
                                    }

                                    // canonicalize across 8 symmetries using paired transform
                                    (ulong ch0, ulong ch1, int code) = Canonicalize128(hBits, gBits);

                                    allBlocks.Add(new BlockSig
                                    {
                                        Tile = stem,
                                        TileX = tileX,
                                        TileY = tileY,
                                        OriginX = ox,
                                        OriginY = oy,
                                        W = W,
                                        H = H,
                                        H0 = ch0,
                                        H1 = ch1,
                                        Ruggedness = ruggedness,
                                        EdgeDensity = edgeDensity
                                    });
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Warn: prefab scan failed for {adtPath}: {ex.Message}");
                }
            }

            // Bucket by top 12 bits of H0 to prune
            var buckets = new Dictionary<ushort, List<int>>();
            for (int i = 0; i < allBlocks.Count; i++)
            {
                ushort key = (ushort)(allBlocks[i].H0 >> 52);
                if (!buckets.TryGetValue(key, out var list)) { list = new List<int>(); buckets[key] = list; }
                list.Add(i);
            }

            string pairsPath = Path.Combine(outputDir, "similar_blocks.ndjson");
            if (File.Exists(pairsPath)) File.Delete(pairsPath);
            int pairCount = 0;
            var groupFreq = new Dictionary<(ulong, ulong), int>();

            foreach (var kv in buckets)
            {
                var idxs = kv.Value;
                for (int a = 0; a < idxs.Count; a++)
                {
                    var A = allBlocks[idxs[a]];
                    for (int b = a + 1; b < idxs.Count; b++)
                    {
                        var B = allBlocks[idxs[b]];
                        if (A.W != B.W || A.H != B.H) continue;
                        int hd = PopCount(A.H0 ^ B.H0) + PopCount(A.H1 ^ B.H1);
                        if (hd <= maxHamming128)
                        {
                            string row = "{\"size\":\"" + A.W + "x" + A.H + "\",\"tileA\":\"" + A.Tile + "\",\"oxA\":" + A.OriginX + ",\"oyA\":" + A.OriginY
                                       + ",\"tileB\":\"" + B.Tile + "\",\"oxB\":" + B.OriginX + ",\"oyB\":" + B.OriginY
                                       + ",\"hamming\":" + hd + ",\"rugA\":" + A.Ruggedness.ToString("F3", CultureInfo.InvariantCulture) + ",\"edgeA\":" + A.EdgeDensity.ToString("F3", CultureInfo.InvariantCulture)
                                       + ",\"rugB\":" + B.Ruggedness.ToString("F3", CultureInfo.InvariantCulture) + ",\"edgeB\":" + B.EdgeDensity.ToString("F3", CultureInfo.InvariantCulture) + "}";
                            File.AppendAllText(pairsPath, row + Environment.NewLine);
                            pairCount++;
                            var key128 = (A.H0, A.H1);
                            groupFreq[key128] = groupFreq.TryGetValue(key128, out var f) ? f + 1 : 1;

                            if (exportMatches && exported < exportMax)
                            {
                                try
                                {
                                    // Export representative blocks A and B
                                    ExportBlockOBJ(inputDir, A, Path.Combine(exportDir, $"A_{A.Tile}_{A.OriginX}_{A.OriginY}_{A.W}x{A.H}_hd{hd}.obj"), recursive);
                                    exported++;
                                    if (exported < exportMax)
                                    {
                                        ExportBlockOBJ(inputDir, B, Path.Combine(exportDir, $"B_{B.Tile}_{B.OriginX}_{B.OriginY}_{B.W}x{B.H}_hd{hd}.obj"), recursive);
                                        exported++;
                                    }
                                }
                                catch (Exception ex)
                                {
                                    System.Console.WriteLine($"Warn: export block failed: {ex.Message}");
                                }
                            }
                        }
                    }
                }
            }

            var report = new StringBuilder();
            report.AppendLine("Prefab Similarity Report");
            report.AppendLine("=======================");
            report.AppendLine($"Blocks scanned: {allBlocks.Count}, pairs (<= {maxHamming128}): {pairCount}");
            report.AppendLine("Sizes: " + string.Join(',', sizes.Select(s => $"{s.w}x{s.h}")));
            File.WriteAllText(Path.Combine(outputDir, "prefab_similarity_report.txt"), report.ToString());
            System.Console.WriteLine("Prefab similarity analysis completed.");
        }

        private static (ulong, ulong, int) Canonicalize128(ulong a, ulong b)
        {
            // Apply same symmetry transform on both channels, choose minimal (a,b) lexicographically
            ulong bestA = a, bestB = b; int bestCode = 0;
            for (int code = 0; code < 8; code++)
            {
                ulong ta = TransformBitboard(a, code);
                ulong tb = TransformBitboard(b, code);
                bool better = (ta < bestA) || (ta == bestA && tb < bestB);
                if (better) { bestA = ta; bestB = tb; bestCode = code; }
            }
            return (bestA, bestB, bestCode);
        }

        private static float[,] ComputeChunkGrid8(MCNK chunk)
        {
            // Similar to ComputeChunkSignature but returns normalized 8x8 grid
            var hs = chunk.vertices.vertices.ToArray();
            Array.Sort(hs);
            float p05 = hs[(int)(hs.Length * 0.05f)];
            float p95 = hs[(int)(hs.Length * 0.95f)];
            float med = hs[hs.Length / 2];
            float scale = Math.Max(1e-3f, p95 - p05);

            float[,] grid = new float[17, 9];
            bool[,] has = new bool[17, 9];
            int idx = 0;
            for (int row = 0; row < 17; row++)
            {
                bool shortRow = (row % 2) == 1;
                int cols = shortRow ? 8 : 9;
                for (int col = 0; col < cols; col++)
                {
                    float h = chunk.vertices.vertices[idx++];
                    h = MathF.Max(-1f, MathF.Min(1f, (h - med) / scale));
                    grid[row, col] = h; has[row, col] = true;
                }
                if (shortRow) has[row, 8] = false;
            }
            for (int r = 0; r < 17; r++) if (!has[r, 8]) { grid[r, 8] = grid[r, 7]; has[r, 8] = true; }
            return DownsampleBilinear(grid, 17, 9, 8, 8);
        }

        private static float[,] DownsampleBilinear(float[,] src, int sh, int sw, int dh, int dw)
        {
            float[,] dst = new float[dh, dw];
            for (int y = 0; y < dh; y++)
            {
                float srcY = (y + 0.5f) * ((sh - 1f) / dh);
                int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, sh - 1);
                int y1 = Math.Clamp(y0 + 1, 0, sh - 1);
                float ty = srcY - y0;
                for (int x = 0; x < dw; x++)
                {
                    float srcX = (x + 0.5f) * ((sw - 1f) / dw);
                    int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, sw - 1);
                    int x1 = Math.Clamp(x0 + 1, 0, sw - 1);
                    float tx = srcX - x0;
                    float v00 = src[y0, x0];
                    float v10 = src[y0, x1];
                    float v01 = src[y1, x0];
                    float v11 = src[y1, x1];
                    float v0 = v00 + (v10 - v00) * tx;
                    float v1 = v01 + (v11 - v01) * tx;
                    float v = v0 + (v1 - v0) * ty;
                    dst[y, x] = v;
                }
            }
            return dst;
        }

        private static void ExportBlockOBJ(string inputDir, BlockSig blk, string outPath, bool recursive)
        {
            // Re-read the ADT for the tile and stitch the block
            var enumOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            string? adtPath = Directory.EnumerateFiles(inputDir, blk.Tile + ".adt", enumOption).FirstOrDefault();
            if (adtPath == null) return;
            var adtReader = new ADTReader();
            using (var stream = File.OpenRead(adtPath)) adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
            var adt = adtReader.adtfile;
            var chunkMap = adt.chunks.ToDictionary(c => ((int)c.header.indexX, (int)c.header.indexY));

            using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var w = new StreamWriter(fs);
            w.WriteLine("# Prefab Block Export");
            w.WriteLine($"# Tile {blk.Tile}, Origin ({blk.OriginX},{blk.OriginY}), Size {blk.W}x{blk.H}");

            int vbase = 0;
            for (int by = 0; by < blk.H; by++)
            {
                for (int bx = 0; bx < blk.W; bx++)
                {
                    if (!chunkMap.TryGetValue((blk.OriginX + bx, blk.OriginY + by), out var chunk)) continue;

                    // Write vertices
                    int idx = 0;
                    for (int row = 0; row < 17; row++)
                    {
                        bool isShort = (row % 2) == 1;
                        int colCount = isShort ? 8 : 9;
                        for (int col = 0; col < colCount; col++)
                        {
                            // Reuse coordinate convention from merged export
                            const float TILE_SIZE = 533.333333f;
                            const float CHUNK_SIZE = TILE_SIZE / 16f;
                            const float UNIT_SIZE = CHUNK_SIZE / 8f;
                            const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

                            float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                            if (isShort) vx -= UNIT_SIZE_HALF;
                            float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                            float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                            w.WriteLine($"v {vx:F6} {vy:F6} {vz:F6}");
                            idx++;
                        }
                    }

                    // Write faces (same as single chunk)
                    int localBase = vbase;
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
                            int i0 = j;
                            int a = i0, b = i0 - 9, c = i0 + 8;
                            int d = i0 - 8, e = i0 + 9;
                            w.WriteLine($"f {localBase + a + 1} {localBase + b + 1} {localBase + c + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + d + 1} {localBase + b + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + e + 1} {localBase + d + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + c + 1} {localBase + e + 1}");
                        }

                        if (((j + 1) % (9 + 8)) == 0) j += 9;
                    }

                    vbase += 145; // 145 verts per chunk
                }
            }
        }

        // ===== Cross-tile Prefab Scan (global) =====
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
            System.Console.WriteLine($"Prefab scan (cross-tiles) on: {inputDir} (recursive={recursive})");
            var enumOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

            // Build tile file map and load included tiles
            var tileFile = new Dictionary<(int tx, int ty), string>();
            var tileAdt = new Dictionary<(int tx, int ty), ADT>();

            foreach (var adtPath in Directory.EnumerateFiles(inputDir, "*.adt", enumOption))
            {
                string stem = Path.GetFileNameWithoutExtension(adtPath);
                if (!ParseTileXYFromStem(stem, out int tx, out int ty)) continue;
                bool includeTile = tileSet.Count == 0 && tileRange == null
                    || tileSet.Contains((tx, ty))
                    || (tileRange != null && tx >= tileRange.Value.x1 && tx <= tileRange.Value.x2 && ty >= tileRange.Value.y1 && ty <= tileRange.Value.y2);
                if (!includeTile) continue;
                tileFile[(tx, ty)] = adtPath;
                var r = new ADTReader();
                using (var s = File.OpenRead(adtPath)) r.ReadRootFile(s, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                tileAdt[(tx, ty)] = r.adtfile;
            }

            // Global chunk grid of 8x8 normalized height patches
            var grid8 = new Dictionary<(int gx, int gy), float[,]>();
            int minGX = int.MaxValue, minGY = int.MaxValue, maxGX = int.MinValue, maxGY = int.MinValue;
            foreach (var kv in tileAdt)
            {
                int tx = kv.Key.tx, ty = kv.Key.ty; var adt = kv.Value;
                foreach (var c in adt.chunks)
                {
                    if (c.vertices.vertices == null || c.vertices.vertices.Length == 0) continue;
                    int ix = (int)c.header.indexX; int iy = (int)c.header.indexY;
                    int gx = tx * 16 + ix; int gy = ty * 16 + iy;
                    grid8[(gx, gy)] = ComputeChunkGrid8(c);
                    if (gx < minGX) minGX = gx; if (gx > maxGX) maxGX = gx;
                    if (gy < minGY) minGY = gy; if (gy > maxGY) maxGY = gy;
                }
            }

            var allBlocks = new List<BlockSigGlobal>(50000);
            int exported = 0;
            string exportDir = Path.Combine(outputDir, "prefab_match_exports");
            if (exportMatches) Directory.CreateDirectory(exportDir);

            foreach (var sz in sizes)
            {
                int W = sz.w, H = sz.h;
                for (int oy = minGY; oy <= maxGY - H + 1; oy += stride)
                {
                    for (int ox = minGX; ox <= maxGX - W + 1; ox += stride)
                    {
                        // require full block present
                        bool full = true;
                        for (int by = 0; by < H && full; by++)
                            for (int bx = 0; bx < W && full; bx++)
                                if (!grid8.ContainsKey((ox + bx, oy + by))) full = false;
                        if (!full) continue;

                        // stitch to big patch
                        int PH = H * 8, PW = W * 8;
                        float[,] patch = new float[PH, PW];
                        for (int by = 0; by < H; by++)
                        {
                            for (int bx = 0; bx < W; bx++)
                            {
                                var g = grid8[(ox + bx, oy + by)];
                                for (int yy = 0; yy < 8; yy++)
                                    for (int xx = 0; xx < 8; xx++)
                                        patch[by * 8 + yy, bx * 8 + xx] = g[yy, xx];
                            }
                        }

                        // gradient + ruggedness/edge density
                        float[,] grad = new float[PH, PW];
                        float gradSum = 0f, gradMax = 1e-6f;
                        for (int y = 1; y < PH - 1; y++)
                        {
                            for (int x = 1; x < PW - 1; x++)
                            {
                                float gx = (patch[y, x + 1] - patch[y, x - 1]) * 0.5f;
                                float gy = (patch[y + 1, x] - patch[y - 1, x]) * 0.5f;
                                float g = MathF.Sqrt(gx * gx + gy * gy);
                                grad[y, x] = g; gradSum += g; if (g > gradMax) gradMax = g;
                            }
                        }
                        int totalCount = (PH - 2) * (PW - 2);
                        int edgeCount = 0; float edgeThresh = 0.3f * gradMax;
                        for (int y = 1; y < PH - 1; y++)
                            for (int x = 1; x < PW - 1; x++)
                                if (grad[y, x] >= edgeThresh) edgeCount++;
                        float edgeDensity = totalCount > 0 ? (float)edgeCount / totalCount : 0f;
                        float ruggedness = totalCount > 0 ? (gradSum / totalCount) / (gradMax + 1e-6f) : 0f;
                        if (ruggedness < minRuggedness || edgeDensity < minEdgeDensity) continue;

                        // downsample to 16x16 channels then 64-bit per channel
                        float[,] h16 = DownsampleBilinear(patch, PH, PW, 16, 16);
                        float[,] g16 = DownsampleBilinear(grad, PH, PW, 16, 16);
                        float hMean = 0, gMean = 0; for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) { hMean += h16[y, x]; gMean += g16[y, x]; }
                        hMean /= 256f; gMean /= 256f;
                        ulong hBits = 0UL, gBits = 0UL; int bit = 0;
                        for (int y = 0; y < 16; y += 2)
                        {
                            for (int x = 0; x < 16; x += 2)
                            {
                                float ha = (h16[y, x] + h16[y, x + 1] + h16[y + 1, x] + h16[y + 1, x + 1]) * 0.25f;
                                float ga = (g16[y, x] + g16[y, x + 1] + g16[y + 1, x] + g16[y + 1, x + 1]) * 0.25f;
                                if (ha > hMean) hBits |= (1UL << bit);
                                if (ga > gMean) gBits |= (1UL << bit);
                                bit++;
                            }
                        }
                        (ulong ch0, ulong ch1, int code) = Canonicalize128(hBits, gBits);

                        // compute tile span for metadata
                        int minTX = (int)Math.Floor(ox / 16.0);
                        int minTY = (int)Math.Floor(oy / 16.0);
                        int maxTX = (int)Math.Floor((ox + W - 1) / 16.0);
                        int maxTY = (int)Math.Floor((oy + H - 1) / 16.0);

                        allBlocks.Add(new BlockSigGlobal
                        {
                            OriginGX = ox,
                            OriginGY = oy,
                            W = W,
                            H = H,
                            H0 = ch0,
                            H1 = ch1,
                            Ruggedness = ruggedness,
                            EdgeDensity = edgeDensity,
                            TileSpan = (minTX, minTY, maxTX, maxTY)
                        });
                    }
                }
            }

            // Bucket by top 16 bits to prune for larger windows
            var buckets = new Dictionary<ushort, List<int>>();
            for (int i = 0; i < allBlocks.Count; i++)
            {
                ushort key = (ushort)(allBlocks[i].H0 >> 48);
                if (!buckets.TryGetValue(key, out var list)) { list = new List<int>(); buckets[key] = list; }
                list.Add(i);
            }

            string pairsPath = Path.Combine(outputDir, "similar_blocks.ndjson");
            if (File.Exists(pairsPath)) File.Delete(pairsPath);
            int pairCount = 0;
            var groupFreq = new Dictionary<(ulong, ulong), int>();

            foreach (var kv in buckets)
            {
                var idxs = kv.Value;
                for (int a = 0; a < idxs.Count; a++)
                {
                    var A = allBlocks[idxs[a]];
                    for (int b = a + 1; b < idxs.Count; b++)
                    {
                        var B = allBlocks[idxs[b]];
                        if (A.W != B.W || A.H != B.H) continue;
                        int hd = PopCount(A.H0 ^ B.H0) + PopCount(A.H1 ^ B.H1);
                        if (hd <= maxHamming128)
                        {
                            float sim = 1f - (hd / 128f);
                            string row = "{" +
                                "\"size\":\"" + A.W + "x" + A.H + "\"," +
                                "\"goxA\":" + A.OriginGX + ",\"goyA\":" + A.OriginGY + "," +
                                "\"goxB\":" + B.OriginGX + ",\"goyB\":" + B.OriginGY + "," +
                                "\"hamming\":" + hd + ",\"similarity\":" + sim.ToString("F3", CultureInfo.InvariantCulture) + "," +
                                "\"tileSpanA\":[" + A.TileSpan.minTX + "," + A.TileSpan.minTY + "," + A.TileSpan.maxTX + "," + A.TileSpan.maxTY + "]," +
                                "\"tileSpanB\":[" + B.TileSpan.minTX + "," + B.TileSpan.minTY + "," + B.TileSpan.maxTX + "," + B.TileSpan.maxTY + "]}";
                            File.AppendAllText(pairsPath, row + Environment.NewLine);
                            pairCount++;
                            var key128 = (A.H0, A.H1);
                            groupFreq[key128] = groupFreq.TryGetValue(key128, out var f) ? f + 1 : 1;

                            if (exportMatches && exported < exportMax)
                            {
                                try
                                {
                                    ExportBlockOBJGlobal(tileFile, inputDir, (A.OriginGX, A.OriginGY), (A.W, A.H), Path.Combine(exportDir, $"A_g{A.OriginGX}_{A.OriginGY}_{A.W}x{A.H}_hd{hd}.obj"));
                                    exported++;
                                    if (exported < exportMax)
                                    {
                                        ExportBlockOBJGlobal(tileFile, inputDir, (B.OriginGX, B.OriginGY), (B.W, B.H), Path.Combine(exportDir, $"B_g{B.OriginGX}_{B.OriginGY}_{B.W}x{B.H}_hd{hd}.obj"));
                                        exported++;
                                    }
                                }
                                catch (Exception ex)
                                {
                                    System.Console.WriteLine($"Warn: export global block failed: {ex.Message}");
                                }
                            }
                        }
                    }
                }
            }

            var report = new StringBuilder();
            report.AppendLine("Prefab Similarity Report (Cross-Tiles)");
            report.AppendLine("==============================");
            report.AppendLine($"Blocks scanned: {allBlocks.Count}, pairs (<= {maxHamming128}): {pairCount}");
            report.AppendLine("Sizes: " + string.Join(',', sizes.Select(s => $"{s.w}x{s.h}")));
            File.WriteAllText(Path.Combine(outputDir, "prefab_similarity_report.txt"), report.ToString());
            System.Console.WriteLine("Prefab similarity analysis completed.");
        }

        private static void ExportBlockOBJGlobal(Dictionary<(int tx, int ty), string> tileFile, string inputDir, (int gox, int goy) origin, (int W, int H) size, string outPath)
        {
            // Determine needed tiles
            int gox = origin.gox, goy = origin.goy; int W = size.W, H = size.H;
            int minTX = (int)Math.Floor(gox / 16.0);
            int minTY = (int)Math.Floor(goy / 16.0);
            int maxTX = (int)Math.Floor((gox + W - 1) / 16.0);
            int maxTY = (int)Math.Floor((goy + H - 1) / 16.0);

            // Load required tiles
            var tiles = new Dictionary<(int tx, int ty), ADT>();
            for (int ty = minTY; ty <= maxTY; ty++)
            {
                for (int tx = minTX; tx <= maxTX; tx++)
                {
                    if (!tileFile.TryGetValue((tx, ty), out var path)) return; // missing tile; abort export
                    var r = new ADTReader();
                    using (var s = File.OpenRead(path)) r.ReadRootFile(s, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                    tiles[(tx, ty)] = r.adtfile;
                }
            }

            // Build chunk map per tile
            var chunkMap = new Dictionary<(int gx, int gy), MCNK>();
            foreach (var kv in tiles)
            {
                int tx = kv.Key.tx, ty = kv.Key.ty; var adt = kv.Value;
                foreach (var c in adt.chunks)
                {
                    int ix = (int)c.header.indexX; int iy = (int)c.header.indexY;
                    chunkMap[(tx * 16 + ix, ty * 16 + iy)] = c;
                }
            }

            using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var w = new StreamWriter(fs);
            w.WriteLine("# Prefab Block Export (Global)");
            w.WriteLine($"# OriginGX {gox}, OriginGY {goy}, Size {W}x{H}");

            int vbase = 0;
            for (int by = 0; by < H; by++)
            {
                for (int bx = 0; bx < W; bx++)
                {
                    if (!chunkMap.TryGetValue((gox + bx, goy + by), out var chunk)) continue;

                    // Write vertices (same convention as other exporters)
                    int idx = 0;
                    for (int row = 0; row < 17; row++)
                    {
                        bool isShort = (row % 2) == 1;
                        int colCount = isShort ? 8 : 9;
                        for (int col = 0; col < colCount; col++)
                        {
                            const float TILE_SIZE = 533.333333f;
                            const float CHUNK_SIZE = TILE_SIZE / 16f;
                            const float UNIT_SIZE = CHUNK_SIZE / 8f;
                            const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

                            float vx = chunk.header.position.Y - (col * UNIT_SIZE);
                            if (isShort) vx -= UNIT_SIZE_HALF;
                            float vy = chunk.vertices.vertices[idx] + chunk.header.position.Z;
                            float vz = chunk.header.position.X - (row * UNIT_SIZE_HALF);
                            w.WriteLine($"v {vx:F6} {vy:F6} {vz:F6}");
                            idx++;
                        }
                    }

                    int localBase = vbase;
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
                            int i0 = j; int a = i0, b = i0 - 9, c = i0 + 8; int d = i0 - 8, e = i0 + 9;
                            w.WriteLine($"f {localBase + a + 1} {localBase + b + 1} {localBase + c + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + d + 1} {localBase + b + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + e + 1} {localBase + d + 1}");
                            w.WriteLine($"f {localBase + a + 1} {localBase + c + 1} {localBase + e + 1}");
                        }

                        if (((j + 1) % (9 + 8)) == 0) j += 9;
                    }

                    vbase += 145;
                }
            }
        }

        private static void ProcessADTDirectory(string inputDir, string outputDir, bool recursive, bool noComments, bool glbPerFile, bool writeManifest, bool writeChunksPerTile, bool writeMeta)
        {
            System.Console.WriteLine($"Processing ADT directory: {inputDir} (recursive={recursive})");
            Directory.CreateDirectory(outputDir);

            var objPath = Path.Combine(outputDir, "terrain_merged.obj");
            using var fs = new FileStream(objPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var writer = new StreamWriter(fs);
            writer.WriteLine("# ADT Terrain Mesh (Merged)");
            writer.WriteLine("# Generated by ADTPreFabTool");
            writer.WriteLine("# Based on wow.export triangulation approach");
            writer.WriteLine();

            var patternCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var patternSamples = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
            int totalVertices = 0;
            int chunkIndex = 0;
            int fileCount = 0;

            // tiles_manifest.json at run root (NDJSON)
            string tilesManifestPath = Path.Combine(outputDir, "tiles_manifest.json");
            if (File.Exists(tilesManifestPath)) File.Delete(tilesManifestPath);

            var enumOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var adtPath in Directory.EnumerateFiles(inputDir, "*.adt", enumOption))
            {
                try
                {
                    var adtReader = new ADTReader();
                    using (var stream = File.OpenRead(adtPath))
                    {
                        adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
                    }
                    var adt = adtReader.adtfile;

                    if (!noComments)
                        writer.WriteLine($"# File: {Path.GetFileName(adtPath)}");
                    AppendADTMesh(adt, writer, ref totalVertices, ref chunkIndex, noComments);

                    // Aggregate patterns
                    foreach (var chunk in adt.chunks)
                    {
                        if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
                            continue;
                        var p = AnalyzeChunkPattern(chunk);
                        if (p != null)
                        {
                            patternCounts[p.Type] = patternCounts.TryGetValue(p.Type, out var c) ? c + 1 : 1;
                            if (!patternSamples.TryGetValue(p.Type, out var list))
                            {
                                list = new List<string>();
                                patternSamples[p.Type] = list;
                            }
                            if (list.Count < 3)
                            {
                                list.Add($"  - Chunk ({p.ChunkX}, {p.ChunkY}): {p.Description}");
                            }
                        }
                    }

                    string stem = Path.GetFileNameWithoutExtension(adtPath);
                    string tileDir = Path.Combine(outputDir, stem);
                    Directory.CreateDirectory(tileDir);

                    // Optional per-file GLB (write into per-tile folder)
                    if (glbPerFile)
                    {
                        BuildGlbForADT(adt, tileDir, stem, binary: true, computeNormals: true);
                    }

                    // Per-tile chunks.ndjson (if enabled)
                    string chunksPathRel = Path.Combine(stem, "chunks.ndjson");
                    string chunksPathAbs = Path.Combine(tileDir, "chunks.ndjson");
                    if (writeChunksPerTile)
                    {
                        if (File.Exists(chunksPathAbs)) File.Delete(chunksPathAbs);
                        WriteChunksNdjson(adt, chunksPathAbs);
                    }

                    // Per-tile meta JSON (if enabled)
                    string metaPathRel = Path.Combine(stem, $"{stem}.meta.json");
                    string metaPathAbs = Path.Combine(tileDir, $"{stem}.meta.json");
                    if (writeMeta)
                    {
                        WriteTileMetaJson(adt, adtPath, metaPathAbs);
                    }

                    // tiles_manifest.json entry at run root
                    if (writeManifest)
                    {
                        // Compute tile AABB from chunk positions + vertex heights
                        float minX = float.PositiveInfinity, minY = float.PositiveInfinity, minZ = float.PositiveInfinity;
                        float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity, maxZ = float.NegativeInfinity;
                        foreach (var c in adt.chunks)
                        {
                            if (c.vertices.vertices == null || c.vertices.vertices.Length == 0) continue;
                            float cx = c.header.position.X;
                            float cy = c.header.position.Y;
                            float cz = c.header.position.Z;
                            const float TILE_SIZE = 533.333333f;
                            const float CHUNK_SIZE = TILE_SIZE / 16f;
                            const float HALF = CHUNK_SIZE / 2f;
                            // Horizontal extents from chunk center
                            minX = Math.Min(minX, cx - HALF);
                            maxX = Math.Max(maxX, cx + HALF);
                            minZ = Math.Min(minZ, cz - HALF);
                            maxZ = Math.Max(maxZ, cz + HALF);
                            // Vertical from height samples
                            float vmin = c.vertices.vertices.Min() + cy;
                            float vmax = c.vertices.vertices.Max() + cy;
                            minY = Math.Min(minY, vmin);
                            maxY = Math.Max(maxY, vmax);
                        }
                        string glbRel = glbPerFile ? Path.Combine(stem, stem + ".glb") : "";
                        string metaRel = writeMeta ? metaPathRel : "";
                        string chunksRel = writeChunksPerTile ? chunksPathRel : "";
                        string json = "{\"file\":\"" + Path.GetFileName(adtPath) + "\",\"dir\":\"" + stem + "/\",\"glb\":\"" + glbRel.Replace("\\", "/") + "\",\"meta\":\"" + metaRel.Replace("\\", "/") + "\",\"chunks\":\"" + chunksRel.Replace("\\", "/") + "\",\"aabb\":[" + minX.ToString("F3", CultureInfo.InvariantCulture) + "," + minY.ToString("F3", CultureInfo.InvariantCulture) + "," + minZ.ToString("F3", CultureInfo.InvariantCulture) + "," + maxX.ToString("F3", CultureInfo.InvariantCulture) + "," + maxY.ToString("F3", CultureInfo.InvariantCulture) + "," + maxZ.ToString("F3", CultureInfo.InvariantCulture) + "]}";
                        File.AppendAllText(tilesManifestPath, json + Environment.NewLine);
                    }

                    fileCount++;
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"Warning: failed to process '{adtPath}': {ex.Message}");
                }
            }

            System.Console.WriteLine($"Merged terrain mesh saved to: {objPath}");
            System.Console.WriteLine($"Files: {fileCount}, Chunks: {chunkIndex}, Vertices: {totalVertices}");

            // Write aggregated pattern report
            var patternAnalysis = new StringBuilder();
            patternAnalysis.AppendLine("Terrain Pattern Analysis (Merged)");
            patternAnalysis.AppendLine("===============================");
            int totalPatterns = patternCounts.Values.Sum();
            patternAnalysis.AppendLine($"Total patterns found: {totalPatterns}");
            patternAnalysis.AppendLine();

            foreach (var kvp in patternCounts.OrderByDescending(k => k.Value))
            {
                patternAnalysis.AppendLine($"{kvp.Key}: {kvp.Value} instances");
                if (patternSamples.TryGetValue(kvp.Key, out var list))
                    foreach (var sample in list)
                        patternAnalysis.AppendLine(sample);
                patternAnalysis.AppendLine();
            }

            string patternPath = Path.Combine(outputDir, "terrain_patterns.txt");
            File.WriteAllText(patternPath, patternAnalysis.ToString());
            System.Console.WriteLine($"Pattern analysis saved to: {patternPath}");
        }

        private static void WriteChunksNdjson(ADT adt, string chunksPathAbs)
        {
            const float TILE_SIZE = 533.333333f;
            const float CHUNK_SIZE = TILE_SIZE / 16f;
            const float HALF = CHUNK_SIZE / 2f;

            var sb = new StringBuilder(8192);
            int chunkIdx = 0;
            foreach (var c in adt.chunks)
            {
                if (c.vertices.vertices == null || c.vertices.vertices.Length == 0)
                {
                    chunkIdx++;
                    continue;
                }

                float cx = c.header.position.X;
                float cy = c.header.position.Y;
                float cz = c.header.position.Z;

                // Vertical extents from height samples
                float vmin = c.vertices.vertices.Min() + cy;
                float vmax = c.vertices.vertices.Max() + cy;

                float minX = cx - HALF, maxX = cx + HALF;
                float minY = vmin, maxY = vmax;
                float minZ = cz - HALF, maxZ = cz + HALF;

                float ccx = cx;
                float ccy = (vmin + vmax) * 0.5f;
                float ccz = cz;

                // Some WoWFormatLib structs have nLayers; fallback to 0 if unavailable
                int layerCount = 0;
                try { layerCount = (int)c.header.nLayers; } catch { layerCount = 0; }

                var metrics = AnalyzeChunkPattern(c);
                string metricsJson = metrics != null
                    ? "\"metrics\":{\"type\":\"" + metrics.Type + "\",\"heightRange\":" + metrics.HeightRange.ToString("F3", CultureInfo.InvariantCulture) + ",\"avgSlope\":" + metrics.AverageSlope.ToString("F3", CultureInfo.InvariantCulture) + "}"
                    : "\"metrics\":null";

                sb.Append('{');
                sb.Append("\"chunkIndex\":" + chunkIdx);
                sb.Append(",\"indexX\":" + (int)c.header.indexX);
                sb.Append(",\"indexY\":" + (int)c.header.indexY);
                sb.Append(",\"aabb\":[" + minX.ToString("F3", CultureInfo.InvariantCulture) + "," + minY.ToString("F3", CultureInfo.InvariantCulture) + "," + minZ.ToString("F3", CultureInfo.InvariantCulture) + "," + maxX.ToString("F3", CultureInfo.InvariantCulture) + "," + maxY.ToString("F3", CultureInfo.InvariantCulture) + "," + maxZ.ToString("F3", CultureInfo.InvariantCulture) + "]");
                sb.Append(",\"centroid\":[" + ccx.ToString("F3", CultureInfo.InvariantCulture) + "," + ccy.ToString("F3", CultureInfo.InvariantCulture) + "," + ccz.ToString("F3", CultureInfo.InvariantCulture) + "]");
                sb.Append(",\"texture\":{\"layerCount\":" + layerCount + "}");
                sb.Append(',');
                sb.Append(metricsJson);
                sb.Append('}');
                sb.Append(Environment.NewLine);

                // Flush periodically for large tiles
                if (sb.Length > 1_000_000)
                {
                    File.AppendAllText(chunksPathAbs, sb.ToString());
                    sb.Clear();
                }

                chunkIdx++;
            }
            if (sb.Length > 0) File.AppendAllText(chunksPathAbs, sb.ToString());
        }

        private static void WriteTileMetaJson(ADT adt, string adtPath, string metaPathAbs)
        {
            const float TILE_SIZE = 533.333333f;
            const float CHUNK_SIZE = TILE_SIZE / 16f;
            const float HALF = CHUNK_SIZE / 2f;

            // Compute tile AABB and gather per-chunk layer counts
            float minX = float.PositiveInfinity, minY = float.PositiveInfinity, minZ = float.PositiveInfinity;
            float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity, maxZ = float.NegativeInfinity;
            var layerCounts = new List<int>(256);
            foreach (var c in adt.chunks)
            {
                if (c.vertices.vertices == null || c.vertices.vertices.Length == 0)
                {
                    layerCounts.Add(0);
                    continue;
                }
                float cx = c.header.position.X;
                float cy = c.header.position.Y;
                float cz = c.header.position.Z;
                float vmin = c.vertices.vertices.Min() + cy;
                float vmax = c.vertices.vertices.Max() + cy;
                minX = Math.Min(minX, cx - HALF);
                maxX = Math.Max(maxX, cx + HALF);
                minZ = Math.Min(minZ, cz - HALF);
                maxZ = Math.Max(maxZ, cz + HALF);
                minY = Math.Min(minY, vmin);
                maxY = Math.Max(maxY, vmax);
                int lc = 0; try { lc = (int)c.header.nLayers; } catch { lc = 0; }
                layerCounts.Add(lc);
            }

            // Tile-level metadata JSON
            string json = "{\"positionAABB\":[" + minX.ToString("F3", CultureInfo.InvariantCulture) + "," + minY.ToString("F3", CultureInfo.InvariantCulture) + "," + minZ.ToString("F3", CultureInfo.InvariantCulture) + "," + maxX.ToString("F3", CultureInfo.InvariantCulture) + "," + maxY.ToString("F3", CultureInfo.InvariantCulture) + "," + maxZ.ToString("F3", CultureInfo.InvariantCulture) + "],\"layerCounts\":[" + string.Join(',', layerCounts) + "]}";
            File.WriteAllText(metaPathAbs, json);
        }

        // ===== Existing mesh/glb helpers omitted for brevity (unchanged) =====
        private static void AppendADTMesh(ADT adt, StreamWriter writer, ref int totalVertices, ref int chunkIndex, bool noComments)
        {
            const float TILE_SIZE = 533.333333f;
            const float CHUNK_SIZE = TILE_SIZE / 16f;
            const float UNIT_SIZE = CHUNK_SIZE / 8f;
            const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

            foreach (var chunk in adt.chunks)
            {
                if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0)
                {
                    chunkIndex++;
                    continue;
                }

                if (!noComments)
                {
                    writer.WriteLine($"o chunk_{chunkIndex}");
                }

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

                        writer.WriteLine($"v {vx:F6} {vy:F6} {vz:F6}");
                        idx++;
                    }
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
                        int i0 = j;
                        int a = i0, b = i0 - 9, c = i0 + 8;
                        int d = i0 - 8, e = i0 + 9;
                        writer.WriteLine($"f {totalVertices + a + 1} {totalVertices + b + 1} {totalVertices + c + 1}");
                        writer.WriteLine($"f {totalVertices + a + 1} {totalVertices + d + 1} {totalVertices + b + 1}");
                        writer.WriteLine($"f {totalVertices + a + 1} {totalVertices + e + 1} {totalVertices + d + 1}");
                        writer.WriteLine($"f {totalVertices + a + 1} {totalVertices + c + 1} {totalVertices + e + 1}");
                    }

                    if (((j + 1) % (9 + 8)) == 0) j += 9;
                }

                totalVertices += 145;
                chunkIndex++;
            }
        }

        private static void BuildGlbForADT(ADT adt, string outDir, string stem, bool binary, bool computeNormals)
        {
            // Existing GLB builder (unchanged)
            var scene = new SceneBuilder();
            var mat = new MaterialBuilder().WithDoubleSide(true);
            var mesh = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>("terrain");

            const float TILE_SIZE = 533.333333f;
            const float CHUNK_SIZE = TILE_SIZE / 16f;
            const float UNIT_SIZE = CHUNK_SIZE / 8f;
            const float UNIT_SIZE_HALF = UNIT_SIZE / 2f;

            foreach (var chunk in adt.chunks)
            {
                if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0) continue;

                var prim = mesh.UsePrimitive(mat);

                var vlist = new List<(System.Numerics.Vector3 pos, System.Numerics.Vector3 normal)>(145);

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
                        vlist.Add((new System.Numerics.Vector3(vx, vy, vz), System.Numerics.Vector3.Zero));
                        idx++;
                    }
                }

                // Fake normals for now (optional)
                for (int i = 0; i < vlist.Count; i++) vlist[i] = (vlist[i].pos, new System.Numerics.Vector3(0, 1, 0));

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
                        int i0 = j;
                        int a = i0, b = i0 - 9, c = i0 + 8;
                        int d = i0 - 8, e = i0 + 9;

                        var pa = vlist[a].pos; var pb = vlist[b].pos; var pc = vlist[c].pos;
                        var pd = vlist[d].pos; var pe = vlist[e].pos;

                        prim.AddTriangle(new VertexPositionNormal(pa, vlist[a].normal), new VertexPositionNormal(pb, vlist[b].normal), new VertexPositionNormal(pc, vlist[c].normal));
                        prim.AddTriangle(new VertexPositionNormal(pa, vlist[a].normal), new VertexPositionNormal(pd, vlist[d].normal), new VertexPositionNormal(pb, vlist[b].normal));
                        prim.AddTriangle(new VertexPositionNormal(pa, vlist[a].normal), new VertexPositionNormal(pe, vlist[e].normal), new VertexPositionNormal(pd, vlist[d].normal));
                        prim.AddTriangle(new VertexPositionNormal(pa, vlist[a].normal), new VertexPositionNormal(pc, vlist[c].normal), new VertexPositionNormal(pe, vlist[e].normal));
                    }

                    if (((j + 1) % (9 + 8)) == 0) j += 9;
                }
            }

            var sceneBuilder = new SceneBuilder();
            sceneBuilder.AddRigidMesh(mesh, System.Numerics.Matrix4x4.Identity);
            var model = sceneBuilder.ToGltf2();
            string glbPath = Path.Combine(outDir, stem + ".glb");
            if (File.Exists(glbPath)) File.Delete(glbPath);
            model.SaveGLB(glbPath);
        }

        // ===== Pattern analysis (unchanged) =====
        private class PatternInfo
        {
            public string Type { get; set; } = string.Empty;
            public int ChunkX { get; set; }
            public int ChunkY { get; set; }
            public string Description { get; set; } = string.Empty;
            public float HeightRange { get; set; }
            public float AverageSlope { get; set; }
        }

        private static PatternInfo? AnalyzeChunkPattern(MCNK chunk)
        {
            if (chunk.vertices.vertices == null || chunk.vertices.vertices.Length == 0) return null;
            float minH = chunk.vertices.vertices.Min();
            float maxH = chunk.vertices.vertices.Max();
            float range = maxH - minH;
            float totalSlope = 0; int cnt = 0;
            for (int i = 0; i < chunk.vertices.vertices.Length - 1; i++) { totalSlope += Math.Abs(chunk.vertices.vertices[i + 1] - chunk.vertices.vertices[i]); cnt++; }
            float avgSlope = cnt > 0 ? totalSlope / cnt : 0f;
            string type = range > 8f ? "steep" : (avgSlope > 0.5f ? "sloped" : "flat");
            return new PatternInfo
            {
                Type = type,
                ChunkX = (int)chunk.header.indexX,
                ChunkY = (int)chunk.header.indexY,
                Description = $"range={range:F2}, avgSlope={avgSlope:F2}",
                HeightRange = range,
                AverageSlope = avgSlope
            };
        }

        private static void ProcessADTFile(string inputPath, string outputDir, bool writeObjComments, bool writeGlb)
        {
            Directory.CreateDirectory(outputDir);

            var adtReader = new ADTReader();
            using (var stream = File.OpenRead(inputPath))
            {
                adtReader.ReadRootFile(stream, WoWFormatLib.Structs.WDT.MPHDFlags.wdt_has_maid);
            }
            var adt = adtReader.adtfile;

            string stem = Path.GetFileNameWithoutExtension(inputPath);
            string objPath = Path.Combine(outputDir, stem + ".obj");
            using (var fs = new FileStream(objPath, FileMode.Create, FileAccess.Write, FileShare.Read))
            using (var writer = new StreamWriter(fs))
            {
                writer.WriteLine("# ADT Terrain Mesh (Single)");
                writer.WriteLine("# Generated by ADTPreFabTool");
                writer.WriteLine($"# File: {Path.GetFileName(inputPath)}");
                writer.WriteLine();

                int totalVertices = 0;
                int chunkIndex = 0;
                AppendADTMesh(adt, writer, ref totalVertices, ref chunkIndex, noComments: !writeObjComments);
            }

            if (writeGlb)
            {
                BuildGlbForADT(adt, outputDir, stem, binary: true, computeNormals: true);
            }
        }
    }
}
