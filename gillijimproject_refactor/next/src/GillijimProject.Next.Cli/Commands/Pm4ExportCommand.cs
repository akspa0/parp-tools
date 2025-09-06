using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GillijimProject.Next.Core.PM4;
using GillijimProject.Next.Core.Export;
using System.Text;

namespace GillijimProject.Next.Cli.Commands;

public static class Pm4ExportCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);

        if (!opts.TryGetValue("--input", out var input) || string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("[pm4-export] Missing --input <file|dir>.");
            PrintHelp();
            return 2;
        }

        // Output root and run dir
        string outRoot = opts.TryGetValue("--out-root", out var oroot) && !string.IsNullOrWhiteSpace(oroot) ? oroot : "out";
        string baseName = DeriveBaseName(input, opts.ContainsKey("--include-adjacent"));
        string stamp = DateTime.Now.ToString("MMddyy_HHmmss");
        string runDir = Path.Combine(outRoot, $"{baseName}_{stamp}");
        Directory.CreateDirectory(runDir);

        // Subfolders
        string tilesDir = Path.Combine(runDir, "tiles");
        string objectsDir = Path.Combine(runDir, "objects");
        string pointsDir = Path.Combine(runDir, "points");
        Directory.CreateDirectory(tilesDir);
        Directory.CreateDirectory(objectsDir);
        Directory.CreateDirectory(pointsDir);

        var pattern = opts.TryGetValue("--pattern", out var pat) && !string.IsNullOrWhiteSpace(pat) ? pat : "*.pm4";
        bool includeAdjacent = GetFlag(opts, "--include-adjacent", defaultValue: false);
        bool mscnSidecar = GetFlag(opts, "--mscn-sidecar", defaultValue: false);
        bool invertX = GetFlag(opts, "--invert-x", defaultValue: true);
        bool onlyMerged = GetFlag(opts, "--only-merged", defaultValue: false);
        bool onlyTiles = GetFlag(opts, "--only-tiles", defaultValue: false);
        // Default objects ON to align with desired behavior; user can disable with --objects off
        bool objectsOn = GetFlag(opts, "--objects", defaultValue: true);
        if (onlyMerged && onlyTiles)
        {
            Console.Error.WriteLine("[pm4-export] Cannot use both --only-merged and --only-tiles.");
            return 2;
        }

        // Tee logging to file
        var originalOut = Console.Out; var originalErr = Console.Error;
        var logPath = Path.Combine(runDir, "pm4-export.log");
        using var logWriter = new StreamWriter(File.Open(logPath, FileMode.Create, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
        using var outTee = new TeeWriter(originalOut, logWriter);
        using var errTee = new TeeWriter(originalErr, logWriter);
        Console.SetOut(outTee); Console.SetError(errTee);

        try
        {
            Console.WriteLine($"[pm4-export] runDir={runDir}");
            Console.WriteLine($"[pm4-export] options includeAdjacent={includeAdjacent} onlyMerged={onlyMerged} onlyTiles={onlyTiles} objectsOn={objectsOn} invertX={invertX} mscnSidecar={mscnSidecar}");

            int totalVerts = 0, totalTris = 0, totalAnchors = 0;
            int tilesWritten = 0, objectsWritten = 0;

            // Merged
            if (!onlyTiles)
            {
                Console.WriteLine("[pm4-export] Loading merged scene...");
                var mergedScene = Pm4Loader.Load(input, includeAdjacent, applyMscnRemap: true);
                string mergedPath = Path.Combine(runDir, "merged.obj");
                Pm4ObjExporter.Write(mergedScene, mergedPath, invertX: invertX);
                Console.WriteLine($"[pm4-export] Wrote merged OBJ: {mergedPath}");
                totalVerts += mergedScene.Vertices.Count;
                totalTris += mergedScene.Triangles.Count > 0 ? mergedScene.Triangles.Count : mergedScene.Indices.Count / 3;

                if (mscnSidecar)
                {
                    totalAnchors += WriteMscnSidecars(mergedScene, pointsDir, baseName);
                }
            }

            // Tiles (object-first)
            if (!onlyMerged)
            {
                Console.WriteLine("[pm4-export] Composing tiles from objects (object-first)...");
                Pm4TileComposer.ExportTilesFromObjects(input, invertX, tilesDir);
                tilesWritten = Directory.EnumerateFiles(tilesDir, "*.obj", SearchOption.TopDirectoryOnly).Count();
                Console.WriteLine($"[pm4-export] Wrote {tilesWritten} tile OBJs to {tilesDir}");
            }

            // Per-object (hierarchical assembler)
            if (objectsOn)
            {
                Console.WriteLine("[pm4-export] Exporting per-object OBJs (hierarchical)...");
                try
                {
                    objectsWritten = Pm4ObjectAssemblerService.ExportPerObject(input, includeAdjacent, objectsDir, baseName);
                    Console.WriteLine($"[pm4-export] Wrote {objectsWritten} per-object OBJs to {objectsDir}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[pm4-export] Objects export failed: {ex.Message}");
                }
            }

            Console.WriteLine($"[pm4-export] Done. tiles={tilesWritten} objects={objectsWritten} mergedVertsApprox={totalVerts:N0} mergedFacesApprox={totalTris:N0} anchors={totalAnchors:N0}");
            Console.WriteLine($"[pm4-export] Output run: {runDir}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[pm4-export] Fatal error: {ex.Message}");
            return 1;
        }
        finally
        {
            Console.SetOut(originalOut); Console.SetError(originalErr);
        }
    }

    private static List<string> CollectTileFiles(string input, string pattern)
    {
        var files = new List<string>();
        if (File.Exists(input))
        {
            files.Add(input);
        }
        else if (Directory.Exists(input))
        {
            files.AddRange(Directory.EnumerateFiles(input, pattern, SearchOption.TopDirectoryOnly));
        }
        return files;
    }

    private static string DeriveBaseName(string input, bool includeAdjacent)
    {
        if (File.Exists(input))
        {
            if (includeAdjacent)
            {
                return GetRegionBaseName(input);
            }
            return Path.GetFileNameWithoutExtension(input);
        }
        else if (Directory.Exists(input))
        {
            return new DirectoryInfo(input).Name;
        }
        return "pm4";
    }

    private static int WriteMscnSidecars(Pm4Scene scene, string pointsDir, string baseName)
    {
        if (scene.MscnAnchors.Count == 0)
        {
            Console.WriteLine("[pm4-export] MSCN: no anchors present; skipping sidecar.");
            return 0;
        }

        // Points-only OBJ
        var pointsObj = Path.Combine(pointsDir, baseName + "_mscn.obj");
        using (var sw = new StreamWriter(pointsObj, false))
        {
            sw.NewLine = "\n";
            sw.WriteLine("# MSCN anchor points");
            foreach (var p in scene.MscnAnchors)
            {
                sw.WriteLine($"v {Fmt(p.X)} {Fmt(p.Y)} {Fmt(p.Z)}");
            }
        }
        Console.WriteLine($"[pm4-export] Wrote MSCN points OBJ: {pointsObj}");

        // Counts CSV grouped by tile id (if available)
        var csv = Path.Combine(pointsDir, baseName + "_mscn_counts.csv");
        using (var sw = new StreamWriter(csv, false))
        {
            sw.NewLine = "\n";
            sw.WriteLine("tile_id,tile_x,tile_y,count");
            if (scene.MscnTileIds.Count == scene.MscnAnchors.Count && scene.MscnTileIds.Count > 0)
            {
                var groups = new Dictionary<int, int>();
                for (int i = 0; i < scene.MscnTileIds.Count; i++)
                {
                    int id = scene.MscnTileIds[i];
                    groups[id] = groups.TryGetValue(id, out var c) ? c + 1 : 1;
                }
                foreach (var kv in groups.OrderBy(k => k.Key))
                {
                    int id = kv.Key; int count = kv.Value;
                    int x = id % 64; int y = id / 64;
                    sw.WriteLine($"{id},{x},{y},{count}");
                }
            }
            else
            {
                // No tile ids; emit a single total row with id = -1
                sw.WriteLine($"-1,-1,-1,{scene.MscnAnchors.Count}");
            }
        }
        Console.WriteLine($"[pm4-export] Wrote MSCN counts CSV: {csv}");
        return scene.MscnAnchors.Count;
    }

    private static string GetRegionBaseName(string file)
    {
        var name = Path.GetFileNameWithoutExtension(file);
        var parts = name.Split('_');
        if (parts.Length >= 3 && int.TryParse(parts[^2], out _) && int.TryParse(parts[^1], out _))
        {
            // strip trailing _XX_YY
            return string.Join('_', parts.Take(parts.Length - 2));
        }
        return name;
    }

    private static string Fmt(float v) => v.ToString("G9", CultureInfo.InvariantCulture);

    private static void PrintHelp()
    {
        Console.WriteLine("Usage: pm4-export --input <file|dir> [--pattern *.pm4] [--out-root <dir>] [--include-adjacent] [--mscn-sidecar on|off] [--invert-x on|off] [--only-merged] [--only-tiles] [--objects on|off]");
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
                    dict[key] = string.Empty; // flag
                }
            }
        }
        return dict;
    }

    private static bool GetFlag(Dictionary<string, string> opts, string key, bool defaultValue)
    {
        if (!opts.TryGetValue(key, out var val)) return defaultValue;
        if (string.IsNullOrEmpty(val)) return true;
        if (bool.TryParse(val, out var b)) return b;
        if (string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "yes", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "on", StringComparison.OrdinalIgnoreCase)) return true;
        if (string.Equals(val, "0", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "no", StringComparison.OrdinalIgnoreCase) || string.Equals(val, "off", StringComparison.OrdinalIgnoreCase)) return false;
        return defaultValue;
    }

    private sealed class TeeWriter : TextWriter
    {
        private readonly TextWriter _a; private readonly TextWriter _b; private readonly object _lock = new object();
        public TeeWriter(TextWriter a, TextWriter b) { _a = a; _b = b; }
        public override Encoding Encoding => Encoding.UTF8;
        public override void Write(char value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
        public override void Write(string? value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
        public override void WriteLine(string? value) { lock (_lock) { _a.WriteLine(value); _b.WriteLine(value); } }
        public override void Flush() { lock (_lock) { _a.Flush(); _b.Flush(); } }
    }
}
