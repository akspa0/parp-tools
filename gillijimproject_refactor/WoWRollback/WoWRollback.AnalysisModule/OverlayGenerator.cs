using System.Text.Json;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Generates per-tile overlay JSONs for viewer plugin architecture.
/// </summary>
public sealed class OverlayGenerator
{
    /// <summary>
    /// Generates overlay JSONs for all tiles in a map.
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
