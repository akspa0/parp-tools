using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

namespace WowViewer.Tool.Converter;

internal static class TerrainStratigraphyPatchCommand
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m") ?? "Azeroth";
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string format = GetOption(args, "--format", "-f") ?? "lk";
        float factor = GetFloatOption(args, "--factor") ?? TemporalStratigraphyOptions.DefaultClassicFactor;
        bool unhideHoles = HasFlag(args, "--unhide-holes");

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.WriteLine("Usage: wowviewer-converter terrain-stratigraphy-patch --client-root <dir> --output-dir <dir> [--map <name>] [--format lk|alpha|both] [--factor <f>] [--unhide-holes]");
            return;
        }

        Console.WriteLine($"=== Stratigraphy Patcher ===");
        Console.WriteLine($"Client Root:   {clientRoot}");
        Console.WriteLine($"Map:           {mapName}");
        Console.WriteLine($"Output:        {outputDir}");
        Console.WriteLine($"Scale Factor:  {factor:0.###}x");
        Console.WriteLine($"Unhide Holes:  {unhideHoles}");

        string outputMapDir = Path.Combine(outputDir, "World", "Maps", mapName);
        Directory.CreateDirectory(outputMapDir);

        using var cat = new NativeMpqService();
        cat.LoadArchives([clientRoot]);

        int patchedCount = 0;
        int skippedCount = 0;

        for (int tx = 0; tx < 64; tx++)
        {
            for (int ty = 0; ty < 64; ty++)
            {
                byte[]? adt = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}.adt");
                if (adt == null)
                    continue;

                try
                {
                    using var ms = new MemoryStream(adt);
                    var fs = MapFileSummaryReader.Read(ms, $"{mapName}_{tx}_{ty}");
                    ms.Position = 0;
                    var td = WorldTerrainTileBuilder.Read(ms, fs, applyBaseHeightOffset: true);

                    if (td.Heightmap?.Heights.Length != 257 * 257)
                    {
                        skippedCount++;
                        continue;
                    }

                    float[,] lattice257 = StratigraphyTileExporter.ExpandHeights257(td.Heightmap.Heights);
                    var holeMasks = new ushort[256];
                    for (int i = 0; i < Math.Min(td.Chunks.Count, 256); i++)
                        holeMasks[i] = td.Chunks[i].HoleMask;

                    var analysis = StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tx, ty, $"{mapName}_{tx}_{ty}");

                    if (!analysis.IsWeakSignalCandidate && (!unhideHoles || !analysis.HasDevMeshes))
                    {
                        skippedCount++;
                        continue;
                    }

                    float effectiveFactor = factor;
                    float[,] restored = TemporalMeshRestorer.RestoreLattice(lattice257, effectiveFactor);
                    float[] flatRestored = StratigraphyTileExporter.FlattenHeights257(restored);

                    string outPath = Path.Combine(outputMapDir, $"{mapName}_{tx}_{ty}.adt");
                    File.WriteAllBytes(outPath, adt);
                    AdtTerrainWriter.Write(outPath, outPath, flatRestored);

                    patchedCount++;
                    Console.WriteLine($"  Patched ({tx:D2}, {ty:D2}): {analysis.DominantStratum} -> {effectiveFactor:0.#}x");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error ({tx}, {ty}): {ex.Message}");
                    skippedCount++;
                }
            }
        }

        // Copy Map WDT & WDL if present
        byte[]? wdt = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdt");
        if (wdt != null) File.WriteAllBytes(Path.Combine(outputMapDir, $"{mapName}.wdt"), wdt);

        byte[]? wdl = cat.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdl");
        if (wdl != null) File.WriteAllBytes(Path.Combine(outputMapDir, $"{mapName}.wdl"), wdl);

        Console.WriteLine($"\nPatching Complete: {patchedCount} tiles restored, {skippedCount} unchanged.");
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (names.Contains(args[i])) return args[i + 1];
        return null;
    }

    private static float? GetFloatOption(string[] args, string name)
    {
        string? v = GetOption(args, name);
        return v != null && float.TryParse(v, out float r) ? r : null;
    }

    private static bool HasFlag(string[] args, string name) =>
        args.Any(a => a.Equals(name, StringComparison.OrdinalIgnoreCase));
}
