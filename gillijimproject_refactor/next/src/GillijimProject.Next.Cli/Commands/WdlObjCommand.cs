using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Export;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Cli.Commands;

public static class WdlObjCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdl-obj] Missing or invalid --in <path-to.wdl>.");
            Console.Error.WriteLine("Usage: wdl-obj --in <path-to.wdl> [--out-root <dir>] [--scale <double>] [--height-scale <double>] [--no-normalize-world] [--no-skip-holes] [--only-merged] [--only-tiles]");
            return 2;
        }

        try
        {
            // Run directory
            string outRoot = opts.TryGetValue("--out-root", out var oroot) && !string.IsNullOrWhiteSpace(oroot) ? oroot : "out";
            string baseName = Path.GetFileNameWithoutExtension(inPath);
            string stamp = DateTime.Now.ToString("MMddyy_HHmmss");
            string runDir = Path.Combine(outRoot, $"{baseName}_{stamp}");
            Directory.CreateDirectory(runDir);

            // Tee logging
            var originalOut = Console.Out; var originalErr = Console.Error;
            var logPath = Path.Combine(runDir, "wdl-obj.log");
            using var logWriter = new StreamWriter(File.Open(logPath, FileMode.Create, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
            using var outTee = new ObjTeeWriter(originalOut, logWriter);
            using var errTee = new ObjTeeWriter(originalErr, logWriter);
            Console.SetOut(outTee); Console.SetError(errTee);
            try
            {
                var wdl = AlphaReader.ParseWdl(inPath);

                double scale = 1.0;
                if (opts.TryGetValue("--scale", out var sRaw) && double.TryParse(sRaw, out var sVal)) scale = sVal;
                double heightScale = 1.0;
                if (opts.TryGetValue("--height-scale", out var hRaw) && double.TryParse(hRaw, out var hVal)) heightScale = hVal;
                bool normalizeWorld = !opts.ContainsKey("--no-normalize-world");
                bool skipHoles = !opts.ContainsKey("--no-skip-holes");
                bool onlyMerged = opts.ContainsKey("--only-merged");
                bool onlyTiles = opts.ContainsKey("--only-tiles");
                if (onlyMerged && onlyTiles)
                {
                    Console.Error.WriteLine("[wdl-obj] Cannot use both --only-merged and --only-tiles.");
                    return 2;
                }

                var options = new WdlObjExporter.ExportOptions(Scale: scale, SkipHoles: skipHoles, NormalizeWorld: normalizeWorld, HeightScale: heightScale);

                // Per-tile
                WdlObjExporter.ExportStats tilesStats = new(0, 0, 0);
                string tilesDir = Path.Combine(runDir, "tiles");
                if (!onlyMerged)
                {
                    Directory.CreateDirectory(tilesDir);
                    tilesStats = WdlObjExporter.ExportPerTile(wdl, tilesDir, options);
                    Console.WriteLine($"[wdl-obj] wrote {tilesStats.TilesExported} tile OBJs to {tilesDir}");
                }

                // Merged
                WdlObjExporter.ExportStats mergedStats = new(0, 0, 0);
                string mergedPath = Path.Combine(runDir, "merged.obj");
                if (!onlyTiles)
                {
                    mergedStats = WdlObjExporter.ExportMerged(wdl, mergedPath, options);
                    Console.WriteLine($"[wdl-obj] wrote merged OBJ to {mergedPath}");
                }

                // Summary
                var summary = new ObjSummary(
                    InputFile: Path.GetFullPath(inPath),
                    OutputRunDir: Path.GetFullPath(runDir),
                    Scale: scale,
                    HeightScale: heightScale,
                    NormalizeWorld: normalizeWorld,
                    SkipHoles: skipHoles,
                    OnlyMerged: onlyMerged,
                    OnlyTiles: onlyTiles,
                    Tiles_Exported: tilesStats.TilesExported,
                    Tiles_Vertices: tilesStats.VerticesWritten,
                    Tiles_Faces: tilesStats.FacesWritten,
                    Merged_Tiles: mergedStats.TilesExported,
                    Merged_VerticesApprox: mergedStats.VerticesWritten,
                    Merged_Faces: mergedStats.FacesWritten,
                    XY_Cell_Scale: (normalizeWorld ? 533.3333333333 / 16.0 : 1.0) * scale
                );
                var summaryJson = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(Path.Combine(runDir, "summary.json"), summaryJson);
            }
            finally
            {
                Console.SetOut(originalOut); Console.SetError(originalErr);
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdl-obj] Failed: {ex.Message}");
            return 3;
        }
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (key.StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    dict[key] = args[++i];
                }
                else
                {
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}

internal sealed record ObjSummary(
    string InputFile,
    string OutputRunDir,
    double Scale,
    double HeightScale,
    bool NormalizeWorld,
    bool SkipHoles,
    bool OnlyMerged,
    bool OnlyTiles,
    int Tiles_Exported,
    int Tiles_Vertices,
    int Tiles_Faces,
    int Merged_Tiles,
    int Merged_VerticesApprox,
    int Merged_Faces,
    double XY_Cell_Scale
);

internal sealed class ObjTeeWriter : TextWriter
{
    private readonly TextWriter _a; private readonly TextWriter _b; private readonly object _lock = new object();
    public ObjTeeWriter(TextWriter a, TextWriter b) { _a = a; _b = b; }
    public override Encoding Encoding => Encoding.UTF8;
    public override void Write(char value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
    public override void Write(string? value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
    public override void WriteLine(string? value) { lock (_lock) { _a.WriteLine(value); _b.WriteLine(value); } }
    public override void Flush() { lock (_lock) { _a.Flush(); _b.Flush(); } }
}
