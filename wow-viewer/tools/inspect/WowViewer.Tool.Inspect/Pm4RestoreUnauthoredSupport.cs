using System.Globalization;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Tool.Inspect;

/// <summary>
/// CLI command handler for restoring authentic object placements on unauthored PM4 tiles
/// (Spec 270 Phase 2). Emits standard split _obj0.adt files and validates round-trip reading.
/// </summary>
internal static class Pm4RestoreUnauthoredSupport
{
    private const string DefaultDevPath = @"I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development";
    private const string DefaultOutputPath = @"I:\parp\parp-tools\wow-viewer\output\restored_development\World\Maps\development";
    private const string DefaultLedgerPath = @"I:\parp\parp-tools\wow-viewer\output\datasets\pm4_object_association_ledger.json";

    public static void Run(string[] args)
    {
        string pm4Dir = DefaultDevPath;
        string outputDir = DefaultOutputPath;
        string? targetTilesFilter = null;
        string ledgerPath = DefaultLedgerPath;
        string? jsonOutPath = null;
        bool validate = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--pm4":
                case "--map":
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) pm4Dir = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--tiles":
                case "-t":
                    if (i + 1 < args.Length) targetTilesFilter = args[++i];
                    break;
                case "--ledger":
                    if (i + 1 < args.Length) ledgerPath = args[++i];
                    break;
                case "--json":
                    if (i + 1 < args.Length) jsonOutPath = args[++i];
                    break;
                case "--no-validate":
                    validate = false;
                    break;
            }
        }

        string resolvedDir = Pm4CoordinateService.ResolveMapDirectory(pm4Dir);
        string nestedMap = Path.Combine(resolvedDir, "World", "Maps", "development");
        if (Directory.Exists(nestedMap))
            resolvedDir = nestedMap;

        if (!Directory.Exists(resolvedDir))
        {
            Console.Error.WriteLine($"Error: PM4 directory '{resolvedDir}' does not exist.");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);

        Console.WriteLine("=================================================================");
        Console.WriteLine(" Spec 270: PM4 Unauthored Tile Placement Restoration Pipeline");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Source Directory: {resolvedDir}");
        Console.WriteLine($"Output Directory: {outputDir}");

        // Build shape library from authored development tiles and/or ledger
        Dictionary<string, List<Vector3>> shapeLibrary = BuildShapeLibrary(resolvedDir, ledgerPath);
        Console.WriteLine($"Shape Library: Loaded {shapeLibrary.Count} unique asset shape profiles.");

        // Find candidate unauthored PM4 tiles
        var pm4Files = Directory.EnumerateFiles(resolvedDir, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .ToList();

        HashSet<string>? filterSet = null;
        if (!string.IsNullOrWhiteSpace(targetTilesFilter) && !targetTilesFilter.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            filterSet = new HashSet<string>(
                targetTilesFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(s => s.Replace(".pm4", "", StringComparison.OrdinalIgnoreCase)),
                StringComparer.OrdinalIgnoreCase);
        }

        var results = new List<Pm4TileResolutionResult>();
        int totalWmos = 0;
        int totalDoodads = 0;
        int roundTripPassed = 0;
        int roundTripFailed = 0;
        int processedCount = 0;

        foreach (string pm4Path in pm4Files)
        {
            string fileName = Path.GetFileName(pm4Path);
            string tileStem = Path.GetFileNameWithoutExtension(pm4Path);

            if (filterSet != null)
            {
                bool matches = filterSet.Contains(tileStem) ||
                               filterSet.Any(f => tileStem.EndsWith(f, StringComparison.OrdinalIgnoreCase));
                if (!matches)
                    continue;
            }
            else
            {
                // Check if tile is unauthored: PM4 exists, but no companion _obj0.adt exists in source directory
                string? existingObj0 = Pm4CoordinateService.TryGetObj0PathForPm4(pm4Path);
                if (existingObj0 is not null && File.Exists(existingObj0))
                {
                    // Skip authored tiles when running full batch
                    continue;
                }
            }

            // Quick check if PM4 contains geometry
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            processedCount++;

            try
            {
                Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
                    pm4Path,
                    resolvedDir,
                    companionPath => AdtPlacementReader.Read(companionPath),
                    shapeLibrary);

                // Write split _obj0.adt
                string outObj0Name = $"{tileStem}_obj0.adt";
                string outObj0Path = Path.Combine(outputDir, outObj0Name);

                SplitAdtObjWriter.Write(outObj0Path, result.Catalog);

                bool rtOk = true;
                if (validate)
                {
                    try
                    {
                        AdtPlacementCatalog readBack = AdtPlacementReader.Read(outObj0Path);
                        if (readBack.WorldModelPlacements.Count != result.Catalog.WorldModelPlacements.Count ||
                            readBack.ModelPlacements.Count != result.Catalog.ModelPlacements.Count)
                        {
                            rtOk = false;
                        }
                    }
                    catch (Exception ex)
                    {
                        rtOk = false;
                        Console.Error.WriteLine($"[FAIL] Round-trip error on {outObj0Name}: {ex.Message}");
                    }
                }

                if (rtOk) roundTripPassed++;
                else roundTripFailed++;

                totalWmos += result.ResolvedWmos.Count;
                totalDoodads += result.ResolvedDoodads.Count;
                results.Add(result);

                // Console logging for tile
                Console.WriteLine($"\n[{processedCount}] Restored: {tileStem} (Grid: [{result.TileX}, {result.TileY}])");
                Console.WriteLine($"    WMO Placements:    {result.ResolvedWmos.Count}");
                Console.WriteLine($"    Doodad Placements: {result.ResolvedDoodads.Count}");
                Console.WriteLine($"    Round-Trip Check:  {(rtOk ? "PASS (100% Chunk Match)" : "FAIL")}");

                foreach (Pm4ResolvedObjectRecord wmo in result.ResolvedWmos)
                {
                    string shortMethod = wmo.Method switch
                    {
                        Pm4PlacementResolutionMethod.AdjacentSpillover => "Spillover",
                        Pm4PlacementResolutionMethod.LedgerExactZ => "Ledger-Z",
                        Pm4PlacementResolutionMethod.NeighborThematic => "Neighbor-Theme",
                        Pm4PlacementResolutionMethod.ShapeScore => "Shape-Score",
                        _ => "Fallback"
                    };
                    Console.WriteLine($"    * WMO [{shortMethod}] {Path.GetFileName(wmo.ModelPath)} @ Z={wmo.PlacementZ:F2} (Conf: {wmo.Confidence:P0}) - {wmo.SourceNote}");
                }

                if (result.ResolvedDoodads.Count > 0)
                {
                    var doodadSummary = result.ResolvedDoodads
                        .GroupBy(d => Path.GetFileName(d.ModelPath))
                        .OrderByDescending(g => g.Count())
                        .Take(5)
                        .Select(g => $"{g.Key} ({g.Count()})");
                    Console.WriteLine($"    * Top Doodads: {string.Join(", ", doodadSummary)}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Failed to restore tile '{tileStem}': {ex.Message}");
            }
        }

        Console.WriteLine("\n=================================================================");
        Console.WriteLine(" Restoration Summary");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"Tiles Restored:      {results.Count}");
        Console.WriteLine($"Total WMOs Restored: {totalWmos}");
        Console.WriteLine($"Total M2s Restored:  {totalDoodads}");
        Console.WriteLine($"Round-Trip Verified: {roundTripPassed} PASS / {roundTripFailed} FAIL");
        Console.WriteLine($"Output Location:     {outputDir}");
        Console.WriteLine("=================================================================\n");

        if (!string.IsNullOrWhiteSpace(jsonOutPath))
        {
            var summaryObj = new
            {
                SourceDirectory = resolvedDir,
                OutputDirectory = outputDir,
                TilesRestoredCount = results.Count,
                TotalWmosRestored = totalWmos,
                TotalDoodadsRestored = totalDoodads,
                RoundTripPassed = roundTripPassed,
                RoundTripFailed = roundTripFailed,
                Tiles = results.Select(r => new
                {
                    r.TileName,
                    r.TileX,
                    r.TileY,
                    WmoCount = r.ResolvedWmos.Count,
                    DoodadCount = r.ResolvedDoodads.Count,
                    Wmos = r.ResolvedWmos.Select(w => new
                    {
                        w.ModelPath,
                        w.PlacementZ,
                        Method = w.Method.ToString(),
                        w.Confidence,
                        w.SourceNote,
                        w.Position,
                        w.Rotation
                    })
                })
            };

            File.WriteAllText(jsonOutPath, JsonSerializer.Serialize(summaryObj, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"Wrote detailed report to: {jsonOutPath}");
        }
    }

    private static Dictionary<string, List<Vector3>> BuildShapeLibrary(string mapDir, string ledgerPath)
    {
        var library = new Dictionary<string, List<Vector3>>(StringComparer.OrdinalIgnoreCase);

        // 1. Try loading from persistent ledger
        if (File.Exists(ledgerPath))
        {
            try
            {
                using JsonDocument json = JsonDocument.Parse(File.ReadAllBytes(ledgerPath));
                if (json.RootElement.TryGetProperty("Entries", out JsonElement entries))
                {
                    foreach (JsonElement rec in entries.EnumerateArray())
                    {
                        if (rec.TryGetProperty("Status", out JsonElement statusElem) &&
                            statusElem.GetString() == "matched" &&
                            rec.TryGetProperty("AssetPath", out JsonElement modelElem) &&
                            rec.TryGetProperty("MinX", out JsonElement minX) &&
                            rec.TryGetProperty("MinY", out JsonElement minY) &&
                            rec.TryGetProperty("MinZ", out JsonElement minZ) &&
                            rec.TryGetProperty("MaxX", out JsonElement maxX) &&
                            rec.TryGetProperty("MaxY", out JsonElement maxY) &&
                            rec.TryGetProperty("MaxZ", out JsonElement maxZ))
                        {
                            string? model = modelElem.GetString();
                            if (!string.IsNullOrWhiteSpace(model))
                            {
                                Vector3 shape = new(
                                    MathF.Abs(maxX.GetSingle() - minX.GetSingle()),
                                    MathF.Abs(maxY.GetSingle() - minY.GetSingle()),
                                    MathF.Abs(maxZ.GetSingle() - minZ.GetSingle()));
                                if (!library.TryGetValue(model, out List<Vector3>? shapes))
                                {
                                    shapes = [];
                                    library[model] = shapes;
                                }
                                shapes.Add(shape);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: Failed to load shape library from ledger: {ex.Message}");
            }
        }

        // 2. Supplement from authored tiles in mapDir
        foreach (string pm4Path in Directory.EnumerateFiles(mapDir, "*.pm4", SearchOption.TopDirectoryOnly))
        {
            string? companion = Pm4CoordinateService.TryGetObj0PathForPm4(pm4Path);
            if (companion is null || !File.Exists(companion))
                continue;

            try
            {
                AdtPlacementCatalog cat = AdtPlacementReader.Read(companion);
                Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
                if (chunks.Msur.Count == 0)
                    continue;

                // Group surfaces by PackedParams / Key
                var groups = new Dictionary<uint, List<Pm4MsurEntry>>();
                for (int i = 0; i < chunks.Msur.Count; i++)
                {
                    uint key = chunks.Msur[i].PackedParams;
                    if (key == 0) continue;
                    if (!groups.TryGetValue(key, out var list))
                    {
                        list = [];
                        groups[key] = list;
                    }
                    list.Add(chunks.Msur[i]);
                }

                var boundsMap = new Dictionary<uint, (Vector3 Min, Vector3 Max)>();
                foreach (var (key, surfaces) in groups)
                {
                    List<Vector3> verts = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
                    if (verts.Count == 0) continue;
                    Vector3 min = new(float.MaxValue), max = new(float.MinValue);
                    foreach (Vector3 v in verts)
                    {
                        Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(v);
                        min = Vector3.Min(min, p);
                        max = Vector3.Max(max, p);
                    }
                    boundsMap[key] = (min, max);
                }

                foreach (AdtWorldModelPlacement placed in cat.WorldModelPlacements)
                {
                    float targetZ = placed.Position.Z;
                    foreach (var (key, (min, max)) in boundsMap)
                    {
                        float z = BitConverter.UInt32BitsToSingle(key);
                        if (MathF.Abs(z - targetZ) <= 0.05f)
                        {
                            Vector3 span = max - min;
                            Vector3 shape = new(MathF.Abs(span.X), MathF.Abs(span.Y), MathF.Abs(span.Z));
                            if (!library.TryGetValue(placed.ModelPath, out List<Vector3>? shapes))
                            {
                                shapes = [];
                                library[placed.ModelPath] = shapes;
                            }
                            shapes.Add(shape);
                        }
                    }
                }
            }
            catch
            {
                // Ignore individual file parsing errors
            }
        }

        return library;
    }
}
