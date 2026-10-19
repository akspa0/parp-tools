using System.Numerics;
using System.Text.Json;
using WowViewer.Core.Archaeology;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Population;

namespace WowViewer.Tool.Inspect;

internal static class MapStratigraphySupport
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        IncludeFields = true
    };

    public static void Run(string[] args)
    {
        string? inputPath = GetOption(args, "--input", "-i");
        string? popPath = GetOption(args, "--population", "-p");
        string? outputPath = GetOption(args, "--output", "-o");
        string? buildLabel = GetOption(args, "--build", "-b") ?? "unknown";
        int gapThreshold = int.TryParse(GetOption(args, "--gap", "-g"), out int g) ? g : UniqueIdStratigraphyEngine.DefaultLayerGapThreshold;

        bool isDivergence = args.Any(static a => string.Equals(a, "--divergence", StringComparison.OrdinalIgnoreCase) || string.Equals(a, "-div", StringComparison.OrdinalIgnoreCase));
        int targetMapId = int.TryParse(GetOption(args, "--target"), out int tId) ? tId : -1;
        int refMapId = int.TryParse(GetOption(args, "--ref"), out int rId) ? rId : -1;

        if (string.IsNullOrWhiteSpace(inputPath) && string.IsNullOrWhiteSpace(popPath))
        {
            Console.Error.WriteLine("Error: at least one of --input <adt/wdt/dir> or --population <dataset-dir> is required.");
            Console.WriteLine("Usage: inspect map stratigraphy [--input <path>] [--population <dir>] [--output <report.json>]");
            Console.WriteLine("       inspect map stratigraphy --population <dir> --divergence --target <mapId> --ref <refMapId> [--output <report.json>]");
            Environment.ExitCode = 1;
            return;
        }

        var mapPlacements = new Dictionary<int, List<UniqueIdRecord>>();

        // 1. Ingest population dataset if provided
        if (!string.IsNullOrWhiteSpace(popPath))
        {
            string resolvedPop = Path.GetFullPath(popPath);
            if (File.Exists(resolvedPop) && resolvedPop.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                IngestPopulationFile(resolvedPop, mapPlacements);
            }
            else if (Directory.Exists(resolvedPop))
            {
                foreach (string file in Directory.EnumerateFiles(resolvedPop, "map_*.json"))
                {
                    IngestPopulationFile(file, mapPlacements);
                }
            }
            else
            {
                Console.Error.WriteLine($"Warning: Population path '{resolvedPop}' not found.");
            }
        }

        // 2. Ingest ADT / WDT input if provided
        if (!string.IsNullOrWhiteSpace(inputPath))
        {
            string resolvedInput = Path.GetFullPath(inputPath);
            IngestAdtPlacements(resolvedInput, mapPlacements);
        }

        if (mapPlacements.Count == 0)
        {
            Console.Error.WriteLine("Error: no placements found in provided inputs.");
            Environment.ExitCode = 1;
            return;
        }

        // 3. Handle Cross-Map Divergence Mode
        if (isDivergence)
        {
            if (targetMapId < 0 || refMapId < 0)
            {
                Console.Error.WriteLine("Error: --divergence requires both --target <mapId> and --ref <refMapId>.");
                Environment.ExitCode = 1;
                return;
            }

            if (!mapPlacements.TryGetValue(targetMapId, out var targetList))
            {
                Console.Error.WriteLine($"Error: Target map {targetMapId} has no placements in loaded inputs.");
                Environment.ExitCode = 1;
                return;
            }

            if (!mapPlacements.TryGetValue(refMapId, out var refList))
            {
                Console.Error.WriteLine($"Error: Reference map {refMapId} has no placements in loaded inputs.");
                Environment.ExitCode = 1;
                return;
            }

            string targetName = $"Map_{targetMapId}";
            string refName = $"Map_{refMapId}";

            MapDivergenceReport divergenceReport = CrossMapDivergenceEngine.AnalyzeDivergence(
                targetMapId,
                targetName,
                targetList,
                refMapId,
                refName,
                refList);

            PrintDivergenceSummary(divergenceReport);

            if (!string.IsNullOrWhiteSpace(outputPath))
            {
                string fullOutput = Path.GetFullPath(outputPath);
                string? dir = Path.GetDirectoryName(fullOutput);
                if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(fullOutput, JsonSerializer.Serialize(divergenceReport, JsonOptions));
                Console.WriteLine($"Wrote divergence report: {fullOutput}");
            }

            return;
        }

        // 4. Handle Standard Stratigraphy Profile Mode
        var profiles = new List<MapStratigraphyProfile>();
        foreach (var (mid, placements) in mapPlacements.OrderBy(static kvp => kvp.Key))
        {
            string mapName = $"Map_{mid}";
            var profile = UniqueIdStratigraphyEngine.ComputeProfile(
                mid,
                mapName,
                placements,
                buildVersion: buildLabel,
                layerGapThreshold: gapThreshold);

            profiles.Add(profile);
            PrintProfileSummary(profile);
        }

        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            string fullOutput = Path.GetFullPath(outputPath);
            string? dir = Path.GetDirectoryName(fullOutput);
            if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(fullOutput, JsonSerializer.Serialize(profiles, JsonOptions));
            Console.WriteLine();
            Console.WriteLine($"Wrote {profiles.Count} map stratigraphy profiles to: {fullOutput}");
        }
    }

    private static void IngestPopulationFile(string jsonPath, Dictionary<int, List<UniqueIdRecord>> mapPlacements)
    {
        try
        {
            string json = File.ReadAllText(jsonPath);
            var mapData = JsonSerializer.Deserialize<PopulationMapData>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (mapData == null || mapData.Spawns.Count == 0)
                return;

            int mapId = mapData.MapId;
            if (!mapPlacements.TryGetValue(mapId, out var list))
            {
                list = [];
                mapPlacements[mapId] = list;
            }

            foreach (var s in mapData.Spawns)
            {
                if (string.IsNullOrWhiteSpace(s.ModelPath))
                    continue;

                list.Add(new UniqueIdRecord
                {
                    UniqueId = s.SpawnId,
                    ModelPath = s.ModelPath,
                    DisplayName = s.Name,
                    Position = s.Position,
                    Rotation = new Vector3(0f, 0f, s.Orientation),
                    Scale = s.EffectiveScale > 0 ? s.EffectiveScale : 1.0f,
                    Kind = s.SpawnType == PopulationSpawnType.Creature ? PlacementOriginKind.Creature : PlacementOriginKind.GameObject,
                    MapId = mapId,
                    TileX = 0,
                    TileY = 0,
                    HasTileCoordinates = false
                });
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: Failed to parse population file '{jsonPath}': {ex.Message}");
        }
    }

    private static void IngestAdtPlacements(string inputPath, Dictionary<int, List<UniqueIdRecord>> mapPlacements)
    {
        var placementFiles = new List<string>();
        if (File.Exists(inputPath))
        {
            string ext = Path.GetExtension(inputPath);
            if (ext.Equals(".adt", StringComparison.OrdinalIgnoreCase))
                placementFiles.Add(inputPath);
        }
        else if (Directory.Exists(inputPath))
        {
            placementFiles.AddRange(Directory.EnumerateFiles(inputPath, "*.adt", SearchOption.AllDirectories)
                .Where(static p => !p.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase) && !p.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase)));
        }

        int defaultMapId = 0;
        if (!mapPlacements.TryGetValue(defaultMapId, out var list))
        {
            list = [];
            mapPlacements[defaultMapId] = list;
        }

        foreach (string file in placementFiles)
        {
            try
            {
                AdtPlacementCatalog catalog = AdtPlacementReader.Read(file);
                foreach (var m in catalog.ModelPlacements)
                {
                    list.Add(new UniqueIdRecord
                    {
                        UniqueId = m.UniqueId,
                        ModelPath = m.ModelPath,
                        Position = m.Position,
                        Rotation = m.Rotation,
                        Scale = m.Scale,
                        Kind = PlacementOriginKind.M2,
                        MapId = defaultMapId,
                        TileX = 0,
                        TileY = 0,
                        HasTileCoordinates = false
                    });
                }

                foreach (var w in catalog.WorldModelPlacements)
                {
                    list.Add(new UniqueIdRecord
                    {
                        UniqueId = w.UniqueId,
                        ModelPath = w.ModelPath,
                        Position = w.Position,
                        Rotation = w.Rotation,
                        Scale = 1.0f,
                        Kind = PlacementOriginKind.Wmo,
                        MapId = defaultMapId,
                        TileX = 0,
                        TileY = 0,
                        HasTileCoordinates = false
                    });
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: Failed to read ADT placement file '{file}': {ex.Message}");
            }
        }
    }

    private static void PrintProfileSummary(MapStratigraphyProfile profile)
    {
        Console.WriteLine("-------------------------------------------------------------------");
        Console.WriteLine($"Map {profile.MapId} ({profile.MapName}) — UniqueId Stratigraphy Profile");
        Console.WriteLine($"  Placements: {profile.TotalPlacements} (Distinct IDs: {profile.DistinctUniqueIds})");
        Console.WriteLine($"  Range: {profile.MinUniqueId} .. {profile.MaxUniqueId}");
        Console.WriteLine($"  Quantiles: p10={profile.P10:F0}  p25={profile.P25:F0}  median={profile.Median:F0}  p75={profile.P75:F0}  p90={profile.P90:F0}");
        Console.WriteLine($"  Detected Layers (gap threshold): {profile.Layers.Count}");

        int displayLayers = Math.Min(profile.Layers.Count, 5);
        for (int i = 0; i < displayLayers; i++)
        {
            var l = profile.Layers[i];
            Console.WriteLine($"    Layer #{l.LayerNumber}: {l.StartId}..{l.EndId} (Count={l.TotalCount}: M2={l.M2Count}, WMO={l.WmoCount}, NPC={l.CreatureCount}, GO={l.GameObjectCount})");
        }
        if (profile.Layers.Count > 5)
            Console.WriteLine($"    ... and {profile.Layers.Count - 5} more layers.");
    }

    private static void PrintDivergenceSummary(MapDivergenceReport report)
    {
        Console.WriteLine("===================================================================");
        Console.WriteLine($"Cross-Map Divergence Report: Map {report.TargetMapId} vs Reference Map {report.ReferenceMapId}");
        Console.WriteLine("===================================================================");
        Console.WriteLine($"  Target Total Placements:    {report.TargetTotalPlacements}");
        Console.WriteLine($"  Matched Common Placements:  {report.MatchedPlacementsCount} ({report.SharedPercentage:F1}%)");
        Console.WriteLine($"  Divergence Cutoff UniqueId: {report.DivergenceCutoffUniqueId}");
        Console.WriteLine();
        foreach (string note in report.Notes)
            Console.WriteLine($"  * {note}");
        Console.WriteLine("===================================================================");
    }

    private static string? GetOption(string[] args, string longName, string? shortName = null)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase) ||
                (shortName != null && string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase)))
            {
                if (i + 1 < args.Length)
                    return args[i + 1];
            }
        }
        return null;
    }
}
