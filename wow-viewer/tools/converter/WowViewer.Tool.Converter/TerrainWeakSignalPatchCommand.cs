using System.Text.Json;
using System.Text.RegularExpressions;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Tool.Converter;

internal static class TerrainWeakSignalPatchCommand
{
    private const int TileHeightmapSize = 257;

    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapNameArg = GetOption(args, "--map", "--map");
        string? mapPath = GetOption(args, "--map-path", "-m");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string formatArg = GetOption(args, "--format", "-f") ?? "both";
        float maxHeightRange = GetFloatOption(args, "--max-height-range") ?? WeakSignalDetector.DefaultMaxHeightRange;
        float minHeightBand = GetFloatOption(args, "--min-height-band") ?? WeakSignalOptions.DefaultMinHeightBand;
        float maxHeightBand = GetFloatOption(args, "--max-height-band") ?? WeakSignalOptions.DefaultMaxHeightBand;
        float? manualFactor = GetFloatOption(args, "--amplification-factor");
        bool noCopyFamily = HasFlag(args, "--no-copy-family");

        if (!string.IsNullOrWhiteSpace(clientRoot) && !string.IsNullOrWhiteSpace(mapNameArg))
        {
            RunFromClient(clientRoot, mapNameArg, outputDir!, formatArg, maxHeightRange, minHeightBand, maxHeightBand, manualFactor, noCopyFamily);
            return;
        }

        if (string.IsNullOrWhiteSpace(mapPath) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Usage:\n  --client-root <dir> --map <name> --output-dir <dir>\n  --map-path <dir> --output-dir <dir>");
            Console.Error.WriteLine("Options: --format <alpha|lk|both> --max-height-range <f> --min-height-band <f> --max-height-band <f> --amplification-factor <f> --no-copy-family");
            Environment.ExitCode = 1;
            return;
        }

        string mapRoot = Path.GetFullPath(mapPath);
        string outputRoot = Path.GetFullPath(outputDir);
        string mapName = Path.GetFileName(mapRoot);

        if (!Directory.Exists(mapRoot))
        {
            Console.Error.WriteLine($"Error: map path not found: {mapRoot}");
            Environment.ExitCode = 1;
            return;
        }

        bool wantAlpha = formatArg is "alpha" or "both";
        bool wantLk = formatArg is "lk" or "both";
        var opts = MakeOptions(maxHeightRange, minHeightBand, maxHeightBand, manualFactor);
        var report = MakeReport(mapName, formatArg, maxHeightRange, minHeightBand, maxHeightBand, manualFactor);
        string outputMapDir = Path.Combine(outputRoot, "World", "Maps", mapName);
        Directory.CreateDirectory(outputMapDir);

        string? alphaWdtMpq = Path.Combine(mapRoot, $"{mapName}.wdt.MPQ");
        if (File.Exists(alphaWdtMpq))
        {
            string? alphaWdlMpq = Path.Combine(mapRoot, $"{mapName}.wdl.MPQ");
            RunAlphaFromFile(alphaWdtMpq, alphaWdlMpq, mapName, outputMapDir, opts, report);
            FinalizeReport(outputRoot, report);
            return;
        }

        var rootAdts = IndexRootAdts(mapRoot);
        Console.WriteLine($"Found {rootAdts.Count} root ADT tile(s)");
        int patched = 0, skipped = 0;

        foreach (var (tileName, adtPath) in rootAdts)
        {
            var tr = new WeakSignalTileReport { TileName = tileName };
            report.Tiles.Add(tr);
            try
            {
                var td = WorldTerrainTileBuilder.Read(adtPath, applyBaseHeightOffset: true);
                if (td.Heightmap?.Heights.Length != TileHeightmapSize * TileHeightmapSize)
                { tr.Error = "No valid heightmap"; skipped++; continue; }

                if (!DetectAndAmplifyFlat(tr, td.Heightmap!.Heights, td.Heightmap.MinHeight, td.Heightmap.MaxHeight, opts, out float f2, out string s2, out float[] a2))
                { if (wantLk) CopyFamily(adtPath, outputMapDir, tileName, noCopyFamily); skipped++; continue; }

                tr.WasPatched = true;
                if (wantLk)
                {
                    string outPath = Path.Combine(outputMapDir, $"{tileName}.adt");
                    File.Copy(adtPath, outPath, overwrite: true);
                    AdtTerrainWriter.Write(outPath, outPath, a2);
                    if (!noCopyFamily) CopyFamily(adtPath, outputMapDir, tileName, false);
                }
                patched++;
            }
            catch (Exception ex) { tr.Error = ex.Message; skipped++; }
        }

        if (wantLk)
        {
            CopyIfExists(mapRoot, $"{mapName}.wdt", outputMapDir);
            CopyIfExists(mapRoot, $"{mapName}.wdl", outputMapDir);
        }
        report.PatchedCount = patched;
        report.SkippedCount = skipped;
        report.TotalTiles = rootAdts.Count;
        FinalizeReport(outputRoot, report);
    }

    static void RunFromClient(string clientRoot, string mapName, string outputDir, string formatArg,
        float maxHeightRange, float minHeightBand, float maxHeightBand, float? manualFactor, bool noCopyFamily)
    {
        Console.WriteLine($"Loading '{mapName}' from {clientRoot}");
        string outputRoot = Path.GetFullPath(outputDir);
        string outputMapDir = Path.Combine(outputRoot, "World", "Maps", mapName);
        Directory.CreateDirectory(outputMapDir);
        var opts = MakeOptions(maxHeightRange, minHeightBand, maxHeightBand, manualFactor);
        var report = MakeReport(mapName, formatArg, maxHeightRange, minHeightBand, maxHeightBand, manualFactor);
        bool wantLk = formatArg is "lk" or "both";

        try
        {
            using var cat = new NativeMpqService();
            cat.LoadArchives([clientRoot]);

            byte[]? wdt = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdt");
            if (wdt == null)
            {
                wdt = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdt.MPQ");
                if (wdt != null)
                {
                    RunAlphaFromMemory(wdt, mapName, outputMapDir, opts, report);
                    FinalizeReport(outputRoot, report);
                    return;
                }
                Console.Error.WriteLine($"Error: map '{mapName}' not found in client.");
                Environment.ExitCode = 1;
                return;
            }

            if (AlphaWdtReader.IsAlphaWdt(wdt))
            {
                RunAlphaFromMemory(wdt, mapName, outputMapDir, opts, report);
                FinalizeReport(outputRoot, report);
                return;
            }

            int patched = 0, skipped = 0;
            for (int tx = 0; tx < 64; tx++)
                for (int ty = 0; ty < 64; ty++)
                {
                    byte[]? adt = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}.adt");
                    if (adt == null) continue;
                    string tileName = $"{mapName}_{tx}_{ty}";
                    var tr = new WeakSignalTileReport { TileName = tileName, TileX = tx, TileY = ty };
                    report.Tiles.Add(tr);
                    try
                    {
                        using var ms = new MemoryStream(adt);
                        var fs = MapFileSummaryReader.Read(ms, tileName);
                        ms.Position = 0;
                        var td = WorldTerrainTileBuilder.Read(ms, fs, applyBaseHeightOffset: true);
                        if (td.Heightmap?.Heights.Length != TileHeightmapSize * TileHeightmapSize)
                        { tr.Error = "No valid heightmap"; skipped++; continue; }

                if (!DetectAndAmplifyFlat(tr, td.Heightmap!.Heights, td.Heightmap.MinHeight, td.Heightmap.MaxHeight, opts, out float factor, out string src, out float[] amplified))
                        { skipped++; continue; }

                        tr.WasPatched = true;
                        if (wantLk)
                        {
                            string outPath = Path.Combine(outputMapDir, $"{tileName}.adt");
                            File.WriteAllBytes(outPath, adt);
                            AdtTerrainWriter.Write(outPath, outPath, amplified);
                        }
                        patched++;
                    }
                    catch (Exception ex) { tr.Error = ex.Message; skipped++; }
                }

            if (wantLk)
            {
                File.WriteAllBytes(Path.Combine(outputMapDir, $"{mapName}.wdt"), wdt);
                byte[]? wdl = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdl");
                if (wdl != null) File.WriteAllBytes(Path.Combine(outputMapDir, $"{mapName}.wdl"), wdl);
            }
            report.PatchedCount = patched;
            report.SkippedCount = skipped;
            report.TotalTiles = patched + skipped;
            Console.WriteLine($"complete: patched={patched} skipped={skipped} total={report.TotalTiles}");
        }
        catch (Exception ex) { Console.Error.WriteLine($"Error: {ex.Message}"); Environment.ExitCode = 1; return; }
        FinalizeReport(outputRoot, report);
    }

    // --- Alpha MPQ (in-memory) ---

    static void RunAlphaFromMemory(byte[] wdtData, string mapName, string outputMapDir, WeakSignalOptions opts, WeakSignalPatchReport report)
    {
        Console.WriteLine($"Reading Alpha WDT from memory ({wdtData.Length:N0} bytes)");
        var tiles = AlphaWdtReader.ReadExistingTiles(wdtData);
        Console.WriteLine($"Found {tiles.Count} tiles");
        int patched = 0, skipped = 0;

        foreach (var (tx, ty) in tiles)
        {
            if (!AlphaWdtReader.TryReadTile(wdtData, tx, ty, null, out var original) || original == null)
            { skipped++; continue; }

            string tileName = $"{mapName}_{tx}_{ty}";
            var tr = new WeakSignalTileReport { TileName = tileName, TileX = tx, TileY = ty };
            report.Tiles.Add(tr);

            float[,] hm2d = original.Heightmap;
            var mask = WeakSignalDetector.AnalyzeChunks(hm2d);
            if (!DetectAndAmplifyTile(tr, hm2d, mask, opts, out float factor, out string src, out float[] amplified))
            { skipped++; continue; }

            tr.WasPatched = true;
            patched++;
        }

        report.PatchedCount = patched;
        report.SkippedCount = skipped;
        report.TotalTiles = tiles.Count;
        Console.WriteLine($"Alpha complete: patched={patched} skipped={skipped} total={tiles.Count}");
    }

    // --- Alpha MPQ (file-based) ---
    static void RunAlphaFromFile(string wdtMpqPath, string? wdlMpqPath, string mapName, string outputMapDir, WeakSignalOptions opts, WeakSignalPatchReport report)
    {
        byte[]? wdt = AlphaArchiveReader.ReadFromMpq(wdtMpqPath);
        if (wdt == null) { Console.Error.WriteLine("Failed to read Alpha WDT"); return; }
        RunAlphaFromMemory(wdt, mapName, outputMapDir, opts, report);
    }

    // --- Detection & amplification ---

    static float[,] FlatToArray257(float[] flat)
    {
        var a = new float[TileHeightmapSize, TileHeightmapSize];
        for (int y = 0; y < TileHeightmapSize; y++)
            for (int x = 0; x < TileHeightmapSize; x++)
                a[y, x] = flat[y * TileHeightmapSize + x];
        return a;
    }

    static float[] Array257ToFlat(float[,] a)
    {
        var flat = new float[TileHeightmapSize * TileHeightmapSize];
        for (int y = 0; y < TileHeightmapSize; y++)
            for (int x = 0; x < TileHeightmapSize; x++)
                flat[y * TileHeightmapSize + x] = a[y, x];
        return flat;
    }

    static bool DetectAndAmplifyFlat(WeakSignalTileReport tr, float[] heights, float minH, float maxH, WeakSignalOptions opts,
        out float factor, out string factorSource, out float[] amplified)
    {
        factor = 1f; factorSource = "none"; amplified = [];
        float range = maxH - minH;
        tr.HeightRange = range;
        tr.MinHeight = minH;
        tr.MaxHeight = maxH;

        if (range > opts.MaxHeightRange || range < 0.001f)
        { tr.IsWeakSignalCandidate = false; tr.Severity = "none"; return false; }

        float[,] hm2d = FlatToArray257(heights);
        var mask = WeakSignalDetector.AnalyzeChunks(hm2d);
        if (mask.WeakChunkCount == 0)
        { tr.IsWeakSignalCandidate = false; tr.Severity = "none"; return false; }

        tr.IsWeakSignalCandidate = true;
        tr.Severity = mask.WeakChunkCount >= 128 ? "high" : mask.WeakChunkCount >= 32 ? "medium" : "low";

        if (opts.UseAutoFactor)
        {
            float fb = WeakSignalDetector.EstimateFallbackFactor(minH, maxH);
            if (fb >= WeakSignalDetector.MinFactorThreshold)
            { factor = fb; factorSource = "fallback"; }
            else
            { factor = WeakSignalDetector.ClassicCompressionFactor; factorSource = "classic"; }
        }
        else
        { factor = opts.ManualFactor; factorSource = "manual"; }

        factor = WeakSignalDetector.SnapFactor(factor);
        float[,] amp2d = WeakSignalDetector.AmplifyChunks(hm2d, mask, factor);
        amplified = Array257ToFlat(amp2d);

        tr.AmplificationFactor = factor;
        tr.FactorSource = factorSource;
        return true;
    }

    static bool DetectAndAmplifyTile(WeakSignalTileReport tr, float[,] hm2d, WeakSignalChunkMask mask, WeakSignalOptions opts,
        out float factor, out string factorSource, out float[] amplified)
    {
        factor = 1f; factorSource = "none"; amplified = [];
        float minH = float.MaxValue, maxH = float.MinValue;
        for (int y = 0; y < TileHeightmapSize; y++)
            for (int x = 0; x < TileHeightmapSize; x++)
            { float v = hm2d[y, x]; if (v < minH) minH = v; if (v > maxH) maxH = v; }
        float range = maxH - minH;
        tr.HeightRange = range; tr.MinHeight = minH; tr.MaxHeight = maxH;

        if (range > opts.MaxHeightRange || range < 0.001f || mask.WeakChunkCount == 0)
        { tr.IsWeakSignalCandidate = false; tr.Severity = "none"; return false; }

        tr.IsWeakSignalCandidate = true;
        tr.Severity = mask.WeakChunkCount >= 128 ? "high" : mask.WeakChunkCount >= 32 ? "medium" : "low";

        if (opts.UseAutoFactor)
        {
            float fb = WeakSignalDetector.EstimateFallbackFactor(minH, maxH);
            factor = fb >= WeakSignalDetector.MinFactorThreshold ? fb : WeakSignalDetector.ClassicCompressionFactor;
            factorSource = fb >= WeakSignalDetector.MinFactorThreshold ? "fallback" : "classic";
        }
        else { factor = opts.ManualFactor; factorSource = "manual"; }

        factor = WeakSignalDetector.SnapFactor(factor);
        float[,] amp2d = WeakSignalDetector.AmplifyChunks(hm2d, mask, factor);
        amplified = Array257ToFlat(amp2d);
        tr.AmplificationFactor = factor; tr.FactorSource = factorSource;
        return true;
    }

    // --- Helpers ---

    static WeakSignalOptions MakeOptions(float mhr, float minB, float maxB, float? mf) => new()
    {
        MaxHeightRange = mhr, MinHeightBand = minB, MaxHeightBand = maxB,
        UseAutoFactor = mf == null, ManualFactor = mf ?? 16f,
    };

    static WeakSignalPatchReport MakeReport(string map, string fmt, float mhr, float minB, float maxB, float? mf) => new()
    {
        Command = "terrain-weak-signal-patch", MapName = map, Format = fmt,
        MaxHeightRange = mhr, MinHeightBand = minB, MaxHeightBand = maxB, AmplificationFactor = mf,
    };

    static Dictionary<string, string> IndexRootAdts(string dir)
    {
        var result = new Dictionary<string, string>();
        foreach (string f in Directory.GetFiles(dir, "*_*.adt"))
        {
            string name = Path.GetFileNameWithoutExtension(f);
            if (name.EndsWith("_obj0") || name.EndsWith("_tex0") || name.EndsWith("_lod")) continue;
            result[name] = f;
        }
        return result;
    }

    static void CopyFamily(string srcAdt, string outDir, string tileName, bool skip)
    {
        if (skip) return;
        string dir = Path.GetDirectoryName(srcAdt)!;
        string baseName = Path.GetFileNameWithoutExtension(srcAdt);
        // Use tileName without the map prefix to find family files
        string tileBase = tileName;
        foreach (string suf in new[] { "_obj0", "_tex0", "_lod" })
        {
            string src = Path.Combine(dir, $"{tileBase}{suf}.adt");
            if (File.Exists(src))
                File.Copy(src, Path.Combine(outDir, $"{tileBase}{suf}.adt"), overwrite: true);
        }
    }

    static void CopyIfExists(string srcDir, string file, string dstDir)
    {
        string src = Path.Combine(srcDir, file);
        if (File.Exists(src)) File.Copy(src, Path.Combine(dstDir, file), overwrite: true);
    }

    static void FinalizeReport(string outputRoot, WeakSignalPatchReport report)
    {
        string path = Path.Combine(outputRoot, "weak_signal_patch_report.json");
        var jopts = new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull };
        File.WriteAllText(path, JsonSerializer.Serialize(report, jopts));
        string final = report.TotalTiles > 0
            ? $"terrain-weak-signal-patch complete: patched={report.PatchedCount} skipped={report.SkippedCount} total={report.TotalTiles} report={path}"
            : $"terrain-weak-signal-patch complete: report={path}";
        Console.WriteLine(final);
    }

    static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (names.Contains(args[i])) return args[i + 1];
        return null;
    }

    static float? GetFloatOption(string[] args, string name)
    {
        string? v = GetOption(args, name);
        return v != null && float.TryParse(v, out float r) ? r : null;
    }

    static bool HasFlag(string[] args, string name) =>
        args.Any(a => a.Equals(name, StringComparison.OrdinalIgnoreCase));
}

// --- Report types (internal to this command) ---

sealed class WeakSignalPatchReport
{
    public string Command { get; set; } = "";
    public string MapName { get; set; } = "";
    public string Format { get; set; } = "";
    public float MaxHeightRange { get; set; }
    public float MinHeightBand { get; set; }
    public float MaxHeightBand { get; set; }
    public float? AmplificationFactor { get; set; }
    public int PatchedCount { get; set; }
    public int SkippedCount { get; set; }
    public int TotalTiles { get; set; }
    public List<WeakSignalTileReport> Tiles { get; set; } = [];
}

sealed class WeakSignalTileReport
{
    public string TileName { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public float HeightRange { get; set; }
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
    public bool IsWeakSignalCandidate { get; set; }
    public string Severity { get; set; } = "";
    public float? AmplificationFactor { get; set; }
    public string FactorSource { get; set; } = "";
    public float? AmplifiedMinHeight { get; set; }
    public float? AmplifiedMaxHeight { get; set; }
    public bool WasPatched { get; set; }
    public string? Error { get; set; }
}
