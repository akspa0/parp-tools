using System.Text.Json;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

namespace WowViewer.Tool.Inspect;

internal static class TerrainStratigraphyScanCommand
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m") ?? "Azeroth";
        string? outputDir = GetOption(args, "--output", "-o") ?? Directory.GetCurrentDirectory();

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.WriteLine("Usage: wowviewer-inspect terrain-stratigraphy-scan --client-root <dir> [--map <name>] [--output <dir>]");
            return;
        }

        Console.WriteLine($"=== Stratigraphy Scanner ===");
        Console.WriteLine($"Client Root: {clientRoot}");
        Console.WriteLine($"Map:         {mapName}");
        Console.WriteLine($"Output:      {outputDir}");

        Directory.CreateDirectory(outputDir);

        using var cat = new NativeMpqService();
        cat.LoadArchives([clientRoot]);

        var scannedTiles = new List<StratigraphyTileAnalysis>();
        int scannedCount = 0;
        int candidateCount = 0;
        int devMeshCount = 0;

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
                        continue;

                    float[,] lattice257 = StratigraphyTileExporter.ExpandHeights257(td.Heightmap.Heights);
                    var holeMasks = new ushort[256];
                    for (int i = 0; i < Math.Min(td.Chunks.Count, 256); i++)
                        holeMasks[i] = td.Chunks[i].HoleMask;

                    string tileName = $"{mapName}_{tx}_{ty}";
                    var analysis = StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tx, ty, tileName);
                    scannedTiles.Add(analysis);
                    scannedCount++;

                    if (analysis.IsWeakSignalCandidate) candidateCount++;
                    if (analysis.HasDevMeshes) devMeshCount++;

                    if (analysis.IsWeakSignalCandidate || analysis.HasDevMeshes)
                    {
                        Console.WriteLine($"  Tile ({tx:D2}, {ty:D2}): {analysis.DominantStratum} | {analysis.TotalSurvivingLevels} levels | {analysis.SqueezedChunkCount} squeezed | {analysis.HoledChunkCount} dev mesh");
                    }
                }
                catch
                {
                    // Skip corrupted or unreadable tiles
                }
            }
        }

        string jsonPath = Path.Combine(outputDir, $"{mapName}_stratigraphy_manifest.json");
        var jopts = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(scannedTiles, jopts));

        string csvPath = Path.Combine(outputDir, $"{mapName}_stratigraphy_summary.csv");
        using (var sw = new StreamWriter(csvPath))
        {
            sw.WriteLine("TileX,TileY,TileName,DominantStratum,TotalLevels,MinZ,MaxZ,HeightRange,ActiveChunks,SqueezedChunks,HoledChunks,FlatChunks,InferredMergeOrigin");
            foreach (var t in scannedTiles)
            {
                sw.WriteLine($"{t.TileX},{t.TileY},{t.TileName},{t.DominantStratum},{t.TotalSurvivingLevels},{t.MinHeight:F2},{t.MaxHeight:F2},{t.HeightRange:F4},{t.ActiveChunkCount},{t.SqueezedChunkCount},{t.HoledChunkCount},{t.FlatChunkCount},\"{t.SeamProfile?.InferredMergeOrigin}\"");
            }
        }

        Console.WriteLine($"\nScan Complete: {scannedCount} tiles scanned ({candidateCount} weak signal candidates, {devMeshCount} tiles with dev meshes).");
        Console.WriteLine($"Manifest: {jsonPath}");
        Console.WriteLine($"CSV:      {csvPath}");
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (names.Contains(args[i])) return args[i + 1];
        return null;
    }
}
