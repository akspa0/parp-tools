using System.Text.Json;
using System.Linq;
using System.IO;
using AlphaWdtAnalyzer.Core;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Generates per-tile overlay JSONs for viewer plugin architecture.
/// </summary>
public sealed class OverlayGenerator
{
    /// <summary>
    /// Generates overlay JSONs from AnalysisIndex data.
    /// </summary>
    /// <param name="analysisIndex">Analysis index with placement and terrain data</param>
    /// <param name="viewerDir">Viewer output directory</param>
    /// <param name="mapName">Map name</param>
    /// <param name="version">Version string</param>
    /// <returns>Result with tile counts</returns>
    public OverlayGenerationResult GenerateFromIndex(
        AnalysisIndex analysisIndex,
        string analysisOutputDir,
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            if (analysisIndex.Placements.Count == 0)
            {
                return new OverlayGenerationResult(
                    0, 0, 0, 0,
                    Success: false,
                    ErrorMessage: $"No placements found in analysis index for {mapName}");
            }

            // Convert AnalysisIndex placements to AssetTimelineDetailedEntry for OverlayBuilder
            var entries = ConvertToTimelineEntries(analysisIndex, version);

            // Use OverlayBuilder for proper coordinate transformation
            var overlayBuilder = new OverlayBuilder();
            var options = ViewerOptions.CreateDefault();

            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");
            Directory.CreateDirectory(objectsDir);

            int objectOverlays = 0;
            
            // Group by tile
            var tileGroups = entries
                .GroupBy(e => (e.TileRow, e.TileCol))
                .OrderBy(g => g.Key.TileRow)
                .ThenBy(g => g.Key.TileCol);

            foreach (var tileGroup in tileGroups)
            {
                var (tileRow, tileCol) = tileGroup.Key;
                
                // Use OverlayBuilder to generate JSON with proper world→pixel transformation
                var json = overlayBuilder.BuildOverlayJson(
                    mapName,
                    tileRow,
                    tileCol,
                    tileGroup,
                    options
                );

                var jsonPath = Path.Combine(objectsDir, $"tile_r{tileRow}_c{tileCol}.json");
                File.WriteAllText(jsonPath, json);
                objectOverlays++;
            }

            return new OverlayGenerationResult(
                TilesProcessed: tileGroups.Count(),
                TerrainOverlays: 0,
                ObjectOverlays: objectOverlays,
                ShadowOverlays: 0,
                Success: true);
        }
        catch (Exception ex)
        {
            return new OverlayGenerationResult(
                0, 0, 0, 0,
                Success: false,
                ErrorMessage: $"Overlay generation failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Generates objects overlays from placements.csv copied to 04_analysis/<ver>/objects/{map}_placements.csv
    /// </summary>
    public OverlayGenerationResult GenerateObjectsFromPlacementsCsv(
        string placementsCsvPath,
        string analysisOutputDir,
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            if (!File.Exists(placementsCsvPath))
            {
                return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Missing placements CSV at {placementsCsvPath}");
            }

            MapMasterIndexDocument? master = null;
            var masterDir = Path.Combine(analysisOutputDir, "master");
            var masterPath = Path.Combine(masterDir, $"{mapName}_master_index.json");
            if (File.Exists(masterPath))
            {
                master = JsonSerializer.Deserialize<MapMasterIndexDocument>(File.ReadAllText(masterPath));
            }

            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");
            Directory.CreateDirectory(objectsDir);

            if (master is not null)
            {
                int generated = WritePlacementsFromMaster(master, objectsDir);
                return new OverlayGenerationResult(
                    TilesProcessed: master.Tiles.Count,
                    TerrainOverlays: 0,
                    ObjectOverlays: generated,
                    ShadowOverlays: 0,
                    Success: true);
            }

            // Fallback: derive overlays from CSV directly
            var rows = ReadCsvRows(placementsCsvPath);
            var byTile = rows
                .Where(r => r.Length >= 16)
                .GroupBy(r => (tileX: ParseInt(r[1]), tileY: ParseInt(r[2])))
                .ToList();

            int objectOverlays = 0;
            foreach (var g in byTile)
            {
                var tileX = g.Key.tileX;
                var tileY = g.Key.tileY;

                var placements = g
                    .Select(r => new PlacementOverlayJson
                    {
                        Kind = r[3] ?? string.Empty,
                        UniqueId = TryParseUIntNullable(r[5]),
                        AssetPath = r[4] ?? string.Empty,
                        World = new[] { TryParseFloat(r[6]), TryParseFloat(r[7]), TryParseFloat(r[8]) },
                        TileOffset = Array.Empty<float>(),
                        Chunk = Array.Empty<int>(),
                        Rotation = new[] { TryParseFloat(r[9]), TryParseFloat(r[10]), TryParseFloat(r[11]) },
                        Scale = TryParseFloat(r[12]),
                        Flags = 0,
                        DoodadSet = (ushort)TryParseInt(r[13]),
                        NameSet = (ushort)TryParseInt(r[14])
                    })
                    .ToList();

                var overlay = new TileOverlayJson
                {
                    TileX = tileX,
                    TileY = tileY,
                    Placements = placements
                };

                var jsonPath = Path.Combine(objectsDir, $"tile_{tileX}_{tileY}.json");
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, new JsonSerializerOptions { WriteIndented = true }));
                objectOverlays++;
            }

            return new OverlayGenerationResult(
                TilesProcessed: byTile.Count,
                TerrainOverlays: 0,
                ObjectOverlays: objectOverlays,
                ShadowOverlays: 0,
                Success: true);
        }
        catch (Exception ex)
        {
            return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Objects overlay (placements.csv) failed: {ex.Message}");
        }

        static int TryParseInt(string s) => int.TryParse(s, out var v) ? v : 0;
        static float TryParseFloat(string s) => float.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : 0f;
        static uint? TryParseUIntNullable(string s)
        {
            if (uint.TryParse(s, out var u)) return u;
            if (int.TryParse(s, out var i) && i >= 0) return (uint)i;
            return null;
        }
    }

    /// <summary>
    /// Generates terrain overlay JSONs from terrain.csv emitted by the ADT stage.
    /// </summary>
    public OverlayGenerationResult GenerateTerrainOverlaysFromCsv(
        string adtOutputDir,
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            var terrainCsv = Path.Combine(adtOutputDir, "csv", "maps", mapName, "terrain.csv");
            if (!File.Exists(terrainCsv))
            {
                return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Missing terrain.csv at {terrainCsv}");
            }

            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var terrainDir = Path.Combine(overlaysRoot, "terrain_complete");
            Directory.CreateDirectory(terrainDir);

            // group rows by tile (tile_row, tile_col)
            var groups = new Dictionary<(int row, int col), List<string[]>>();
            using (var reader = new StreamReader(terrainCsv))
            {
                string? line;
                // header
                line = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = SplitCsv(line);
                    if (parts.Length < 23) continue;
                    if (!int.TryParse(parts[1], out var tileRow)) continue;
                    if (!int.TryParse(parts[2], out var tileCol)) continue;
                    var key = (tileRow, tileCol);
                    if (!groups.TryGetValue(key, out var list))
                    {
                        list = new List<string[]>();
                        groups[key] = list;
                    }
                    list.Add(parts);
                }
            }

            int written = 0;
            foreach (var kvp in groups)
            {
                var (tileRow, tileCol) = kvp.Key; // row = Y, col = X
                var rows = kvp.Value;

                // Aggregate properties
                bool hasLiquids = rows.Any(r => ParseBool(r[8]) || ParseBool(r[9]) || ParseBool(r[10]) || ParseBool(r[11]));
                bool hasHoles = rows.Any(r => ParseBool(r[16]));
                int maxLayers = rows.Select(r => ParseInt(r[15])).DefaultIfEmpty(0).Max();
                // Choose representative areaId as mode
                var areaId = rows
                    .Select(r => ParseInt(r[14]))
                    .GroupBy(x => x)
                    .OrderByDescending(g => g.Count())
                    .ThenBy(g => g.Key)
                    .Select(g => g.Key)
                    .FirstOrDefault();

                var overlay = new
                {
                    tileX = tileCol,
                    tileY = tileRow,
                    areaId = areaId,
                    properties = new
                    {
                        hasLiquids = hasLiquids,
                        hasHoles = hasHoles,
                        layers = maxLayers
                    }
                };

                var jsonPath = Path.Combine(terrainDir, $"tile_{tileCol}_{tileRow}.json");
                var options = new JsonSerializerOptions { WriteIndented = true };
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));
                written++;
            }

            return new OverlayGenerationResult(
                TilesProcessed: groups.Count,
                TerrainOverlays: written,
                ObjectOverlays: 0,
                ShadowOverlays: 0,
                Success: true);
        }
        catch (Exception ex)
        {
            return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Terrain overlay generation failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Generates shadow overlay JSONs from shadow.csv emitted by the ADT stage.
    /// </summary>
    public OverlayGenerationResult GenerateShadowOverlaysFromCsv(
        string adtOutputDir,
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            var shadowCsv = Path.Combine(adtOutputDir, "csv", "maps", mapName, "shadow.csv");
            if (!File.Exists(shadowCsv))
            {
                return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Missing shadow.csv at {shadowCsv}");
            }

            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var shadowDir = Path.Combine(overlaysRoot, "shadow_map");
            Directory.CreateDirectory(shadowDir);

            // group rows by tile (tile_row, tile_col), keep only chunks with has_shadow==true
            var groups = new Dictionary<(int row, int col), List<string[]>>();
            using (var reader = new StreamReader(shadowCsv))
            {
                string? line;
                // header
                line = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = SplitCsv(line);
                    if (parts.Length < 8) continue;
                    if (!int.TryParse(parts[1], out var tileRow)) continue;
                    if (!int.TryParse(parts[2], out var tileCol)) continue;
                    if (!ParseBool(parts[5])) continue; // has_shadow false -> skip
                    var key = (tileRow, tileCol);
                    if (!groups.TryGetValue(key, out var list))
                    {
                        list = new List<string[]>();
                        groups[key] = list;
                    }
                    list.Add(parts);
                }
            }

            int written = 0;
            foreach (var kvp in groups)
            {
                var (tileRow, tileCol) = kvp.Key;
                var rows = kvp.Value;
                var chunks = rows.Select(r => new
                {
                    chunkRow = ParseInt(r[3]),
                    chunkCol = ParseInt(r[4]),
                    shadowSize = ParseInt(r[6])
                    // shadow_bitmap_base64 is r[7] (too large to inline per-chunk here)
                }).ToList();

                var overlay = new
                {
                    tileX = tileCol,
                    tileY = tileRow,
                    chunks = chunks,
                    chunkCount = chunks.Count
                };

                var jsonPath = Path.Combine(shadowDir, $"tile_{tileCol}_{tileRow}.json");
                var options = new JsonSerializerOptions { WriteIndented = true };
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));
                written++;
            }

            return new OverlayGenerationResult(
                TilesProcessed: groups.Count,
                TerrainOverlays: 0,
                ObjectOverlays: 0,
                ShadowOverlays: written,
                Success: true);
        }
        catch (Exception ex)
        {
            return new OverlayGenerationResult(0, 0, 0, 0, Success: false, ErrorMessage: $"Shadow overlay generation failed: {ex.Message}");
        }
    }

    // Simple CSV splitter supporting quoted values
    private static string[] SplitCsv(string line)
    {
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    current.Append(c);
                }
            }
            else
            {
                if (c == ',')
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                else if (c == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    current.Append(c);
                }
            }
        }
        result.Add(current.ToString());
        return result.ToArray();
    }

    private MapMasterIndexDocument LoadOrCreateMasterIndex(
        AnalysisIndex analysisIndex,
        string analysisOutputDir,
        string mapName,
        string version,
        out string masterPath)
    {
        var masterDir = Path.Combine(analysisOutputDir, "master");
        masterPath = Path.Combine(masterDir, $"{mapName}_master_index.json");
        
        if (File.Exists(masterPath))
        {
            var json = File.ReadAllText(masterPath);
            return JsonSerializer.Deserialize<MapMasterIndexDocument>(json)!;
        }
        
        // Create from AnalysisIndex - master index should already exist from MapMasterIndexWriter
        // but if not, create minimal structure
        return new MapMasterIndexDocument
        {
            Map = mapName,
            Version = version,
            GeneratedAtUtc = DateTime.UtcNow,
            Tiles = new List<MapTileRecord>()
        };
    }

    private int WritePlacementsFromMaster(MapMasterIndexDocument master, string objectsDir)
    {
        int count = 0;
        foreach (var tile in master.Tiles)
        {
            if (tile.Placements.Count == 0) continue;
            
            var overlay = new TileOverlayJson
            {
                TileX = tile.TileX,
                TileY = tile.TileY,
                Placements = tile.Placements.Select(ToPlacementJson).ToList()
            };
            
            var jsonPath = Path.Combine(objectsDir, $"tile_{tile.TileX}_{tile.TileY}.json");
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, 
                new JsonSerializerOptions { WriteIndented = true }));
            count++;
        }
        return count;
    }

    private List<string[]> ReadCsvRows(string csvPath)
    {
        var rows = new List<string[]>();
        using var reader = new StreamReader(csvPath);
        string? line = reader.ReadLine(); // skip header
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            rows.Add(SplitCsv(line));
        }
        return rows;
    }

    private static int ParseInt(string s) => int.TryParse(s, out var v) ? v : 0;
    
    private static bool ParseBool(string s) => bool.TryParse(s, out var v) && v;

    /// <summary>
    /// Converts AnalysisIndex placements to AssetTimelineDetailedEntry for use with OverlayBuilder.
    /// </summary>
    private List<AssetTimelineDetailedEntry> ConvertToTimelineEntries(AnalysisIndex analysisIndex, string version)
    {
        var entries = new List<AssetTimelineDetailedEntry>();
        
        foreach (var placement in analysisIndex.Placements)
        {
            var fileName = ExtractFileName(placement.AssetPath);
            var fileStem = Path.GetFileNameWithoutExtension(placement.AssetPath);
            var extension = Path.GetExtension(placement.AssetPath);
            
            entries.Add(new AssetTimelineDetailedEntry(
                Version: version,
                Map: placement.MapName,
                TileRow: placement.TileX,
                TileCol: placement.TileY,
                Kind: ConvertAssetTypeToPlacementKind(placement.Type),
                UniqueId: (uint)(placement.UniqueId ?? 0),
                AssetPath: placement.AssetPath,
                Folder: string.Empty,  // Not available in PlacementRecord
                Category: string.Empty,
                Subcategory: string.Empty,
                DesignKit: string.Empty,
                SourceRule: string.Empty,
                KitRoot: string.Empty,
                SubkitPath: string.Empty,
                SubkitTop: string.Empty,
                SubkitDepth: 0,
                FileName: fileName,
                FileStem: fileStem,
                Extension: extension,
                WorldX: placement.WorldX,
                WorldY: placement.WorldY,
                WorldZ: placement.WorldZ,
                RotationX: placement.RotationX,
                RotationY: placement.RotationY,
                RotationZ: placement.RotationZ,
                Scale: placement.Scale,
                Flags: placement.Flags,
                DoodadSet: placement.DoodadSet,
                NameSet: placement.NameSet
            ));
        }
        
        return entries;
    }

    private static PlacementKind ConvertAssetTypeToPlacementKind(AssetType type)
    {
        return type switch
        {
            AssetType.Wmo => PlacementKind.WMO,
            AssetType.MdxOrM2 => PlacementKind.M2,
            _ => PlacementKind.M2
        };
    }

    private static string ExtractFileName(string path)
    {
        if (string.IsNullOrEmpty(path)) return "Unknown";
        return Path.GetFileName(path);
    }

    private static PlacementOverlayJson ToPlacementJson(MapPlacement placement)
    {
        return new PlacementOverlayJson
        {
            Kind = placement.Kind,
            UniqueId = placement.UniqueId,
            AssetPath = placement.AssetPath,
            World = new float[] { placement.WorldNorth, placement.WorldWest, placement.WorldUp },
            TileOffset = new float[] { placement.TileOffsetNorth, placement.TileOffsetWest },
            Chunk = new int[] { placement.ChunkX, placement.ChunkY },
            Rotation = new float[] { placement.RotationX, placement.RotationY, placement.RotationZ },
            Scale = placement.Scale,
            Flags = placement.Flags,
            DoodadSet = placement.DoodadSet,
            NameSet = placement.NameSet
        };
    }

    private sealed record MapMasterIndexDocument
    {
        public required string Map { get; init; }
        public required string Version { get; init; }
        public required DateTime GeneratedAtUtc { get; init; }
        public required IReadOnlyList<MapTileRecord> Tiles { get; init; }
    }

    private sealed record MapTileRecord
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IReadOnlyList<MapPlacement> Placements { get; init; }
    }

    private sealed record MapPlacement
    {
        public required string Kind { get; init; }
        public uint? UniqueId { get; init; }
        public string? AssetPath { get; init; }
        public float RawNorth { get; init; }
        public float RawUp { get; init; }
        public float RawWest { get; init; }
        public float WorldNorth { get; init; }
        public float WorldWest { get; init; }
        public float WorldUp { get; init; }
        public float TileOffsetNorth { get; init; }
        public float TileOffsetWest { get; init; }
        public int ChunkX { get; init; }
        public int ChunkY { get; init; }
        public float RotationX { get; init; }
        public float RotationY { get; init; }
        public float RotationZ { get; init; }
        public float Scale { get; init; }
        public ushort Flags { get; init; }
        public ushort DoodadSet { get; init; }
        public ushort NameSet { get; init; }
    }

    private sealed record TileOverlayJson
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IReadOnlyList<PlacementOverlayJson> Placements { get; init; }
    }

    private sealed record PlacementOverlayJson
    {
        public required string Kind { get; init; }
        public uint? UniqueId { get; init; }
        public string? AssetPath { get; init; }
        public required float[] World { get; init; }
        public required float[] TileOffset { get; init; }
        public required int[] Chunk { get; init; }
        public required float[] Rotation { get; init; }
        public float Scale { get; init; }
        public ushort Flags { get; init; }
        public ushort DoodadSet { get; init; }
        public ushort NameSet { get; init; }
    }

    /// <summary>
    /// Generates overlay JSONs for all tiles in a map (legacy method - reads ADTs directly).
    /// </summary>
    /// <param name="adtMapDir">Directory containing ADT files</param>
    /// <param name="viewerDir">Viewer output directory</param>
    /// <param name="mapName">Map name</param>
    /// <param name="version">Version string</param>
    /// <returns>Result with tile counts</returns>
    public OverlayGenerationResult Generate(
        string adtMapDir,
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            var adtFiles = Directory.GetFiles(adtMapDir, "*.adt", SearchOption.TopDirectoryOnly);
            if (adtFiles.Length == 0)
            {
                return new OverlayGenerationResult(
                    0, 0, 0, 0,
                    Success: false,
                    ErrorMessage: $"No ADT files found in {adtMapDir}");
            }

            // Create overlay directories
            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var terrainDir = Path.Combine(overlaysRoot, "terrain_complete");
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");
            var shadowDir = Path.Combine(overlaysRoot, "shadow_map");

            Directory.CreateDirectory(terrainDir);
            Directory.CreateDirectory(objectsDir);
            Directory.CreateDirectory(shadowDir);

            int terrainOverlays = 0;
            int objectOverlays = 0;
            int shadowOverlays = 0;

            // Generate overlays for each tile
            foreach (var adtPath in adtFiles)
            {
                // Parse tile coordinates
                var fileName = Path.GetFileNameWithoutExtension(adtPath);
                var parts = fileName.Split('_');
                if (parts.Length < 3 || !int.TryParse(parts[^2], out var tileX) || !int.TryParse(parts[^1], out var tileY))
                {
                    continue;
                }

                // Generate terrain overlay
                if (GenerateTerrainOverlay(adtPath, terrainDir, tileX, tileY))
                    terrainOverlays++;

                // Generate objects overlay
                if (GenerateObjectsOverlay(adtPath, objectsDir, tileX, tileY))
                    objectOverlays++;

                // Generate shadow overlay
                if (GenerateShadowOverlay(adtPath, shadowDir, tileX, tileY))
                    shadowOverlays++;
            }

            return new OverlayGenerationResult(
                TilesProcessed: adtFiles.Length,
                TerrainOverlays: terrainOverlays,
                ObjectOverlays: objectOverlays,
                ShadowOverlays: shadowOverlays,
                Success: true);
        }
        catch (Exception ex)
        {
            return new OverlayGenerationResult(
                0, 0, 0, 0,
                Success: false,
                ErrorMessage: $"Overlay generation failed: {ex.Message}");
        }
    }

    private bool GenerateObjectsOverlayFromPlacements(
        List<PlacementRecord> placements,
        string outputDir,
        int tileX,
        int tileY)
    {
        try
        {
            var m2Placements = placements
                .Where(p => p.Type == AssetType.MdxOrM2)
                .Select(p => new
                {
                    uniqueId = p.UniqueId,
                    path = p.AssetPath,
                    x = p.WorldX,
                    y = p.WorldY,
                    z = p.WorldZ,
                    rotX = p.RotationX,
                    rotY = p.RotationY,
                    rotZ = p.RotationZ,
                    scale = p.Scale
                })
                .ToList();

            var wmoPlacements = placements
                .Where(p => p.Type == AssetType.Wmo)
                .Select(p => new
                {
                    uniqueId = p.UniqueId,
                    path = p.AssetPath,
                    x = p.WorldX,
                    y = p.WorldY,
                    z = p.WorldZ,
                    rotX = p.RotationX,
                    rotY = p.RotationY,
                    rotZ = p.RotationZ,
                    doodadSet = p.DoodadSet,
                    nameSet = p.NameSet
                })
                .ToList();

            var overlay = new
            {
                tileX,
                tileY,
                m2Placements,
                wmoPlacements
            };

            var jsonPath = Path.Combine(outputDir, $"tile_{tileX}_{tileY}.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool GenerateTerrainOverlay(string adtPath, string outputDir, int tileX, int tileY)
    {
        try
        {
            // TODO: Read LK ADT and extract MCNK terrain data
            // For now, generate placeholder JSON

            var overlay = new
            {
                tileX,
                tileY,
                areaId = 0, // Placeholder
                properties = new
                {
                    hasLiquids = false,
                    hasHoles = false,
                    layers = 0
                },
                liquids = Array.Empty<object>()
            };

            var jsonPath = Path.Combine(outputDir, $"tile_{tileX}_{tileY}.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool GenerateObjectsOverlay(string adtPath, string outputDir, int tileX, int tileY)
    {
        try
        {
            // TODO: Read LK ADT and extract MDDF/MODF placement data
            // For now, generate placeholder JSON

            var overlay = new
            {
                tileX,
                tileY,
                m2Placements = Array.Empty<object>(),
                wmoplacements = Array.Empty<object>()
            };

            var jsonPath = Path.Combine(outputDir, $"tile_{tileX}_{tileY}.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool GenerateShadowOverlay(string adtPath, string outputDir, int tileX, int tileY)
    {
        try
        {
            // TODO: Read shadow data if available
            // For now, skip shadow overlays (sparse coverage)

            return false; // Not implemented yet
        }
        catch
        {
            return false;
        }
    }
}
