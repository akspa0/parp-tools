using System.Text.Json;
using System.Linq;
using System.IO;
using AlphaWdtAnalyzer.Core;

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
        string viewerDir,
        string mapName,
        string version)
    {
        try
        {
            if (analysisIndex.Tiles.Count == 0)
            {
                return new OverlayGenerationResult(
                    0, 0, 0, 0,
                    Success: false,
                    ErrorMessage: $"No tiles found in analysis index for {mapName}");
            }

            // Create overlay directories
            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");

            Directory.CreateDirectory(objectsDir);

            int objectOverlays = 0;

            // Group placements by tile
            var placementsByTile = analysisIndex.Placements
                .GroupBy(p => (p.TileX, p.TileY))
                .ToDictionary(g => g.Key, g => g.ToList());

            // Generate overlays for each tile
            foreach (var tile in analysisIndex.Tiles)
            {
                // Generate objects overlay if this tile has placements
                if (placementsByTile.TryGetValue((tile.X, tile.Y), out var placements))
                {
                    if (GenerateObjectsOverlayFromPlacements(placements, objectsDir, tile.X, tile.Y))
                        objectOverlays++;
                }
            }

            return new OverlayGenerationResult(
                TilesProcessed: analysisIndex.Tiles.Count,
                TerrainOverlays: 0, // Terrain overlays come from CSVs, not generated here
                ObjectOverlays: objectOverlays,
                ShadowOverlays: 0, // Shadow overlays not implemented yet
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

            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");
            Directory.CreateDirectory(objectsDir);

            // Read CSV rows (skip header)
            var rows = new List<string[]>();
            using (var reader = new StreamReader(placementsCsvPath))
            {
                string? line = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    rows.Add(SplitCsv(line));
                }
            }

            // Group rows by tile
            var byTile = rows
                .Where(r => r.Length >= 16)
                .GroupBy(r => (tileX: ParseInt(r[1]), tileY: ParseInt(r[2])))
                .ToList();

            int objectOverlays = 0;
            foreach (var g in byTile)
            {
                var tileX = g.Key.tileX;
                var tileY = g.Key.tileY;

                var m2Placements = g
                    .Where(r => string.Equals(r[3], "M2", System.StringComparison.OrdinalIgnoreCase))
                    .Select(r => new
                    {
                        uniqueId = TryParseIntNullable(r[5]),
                        fileId = r[4] ?? string.Empty,
                        position = new[] { TryParseFloat(r[6]), TryParseFloat(r[7]), TryParseFloat(r[8]) },
                        rotation = new[] { TryParseFloat(r[9]), TryParseFloat(r[10]), TryParseFloat(r[11]) },
                        scale = TryParseFloat(r[12])
                    })
                    .ToList();

                var wmoPlacements = g
                    .Where(r => string.Equals(r[3], "WMO", System.StringComparison.OrdinalIgnoreCase))
                    .Select(r => new
                    {
                        uniqueId = TryParseIntNullable(r[5]),
                        fileId = r[4] ?? string.Empty,
                        position = new[] { TryParseFloat(r[6]), TryParseFloat(r[7]), TryParseFloat(r[8]) },
                        rotation = new[] { TryParseFloat(r[9]), TryParseFloat(r[10]), TryParseFloat(r[11]) },
                        doodadSet = TryParseInt(r[13]),
                        nameSet = TryParseInt(r[14])
                    })
                    .ToList();

                var overlay = new
                {
                    tileX,
                    tileY,
                    m2Placements,
                    wmoplacements = wmoPlacements
                };

                var jsonPath = Path.Combine(objectsDir, $"tile_{tileX}_{tileY}.json");
                var options = new JsonSerializerOptions { WriteIndented = true };
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(overlay, options));
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

        static int? TryParseIntNullable(string s) => int.TryParse(s, out var v) ? v : (int?)null;
        static int TryParseInt(string s) => int.TryParse(s, out var v) ? v : 0;
        static float TryParseFloat(string s) => float.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : 0f;
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

    private static bool ParseBool(string s) => string.Equals(s, "true", StringComparison.OrdinalIgnoreCase);
    private static int ParseInt(string s) => int.TryParse(s, out var v) ? v : 0;

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
