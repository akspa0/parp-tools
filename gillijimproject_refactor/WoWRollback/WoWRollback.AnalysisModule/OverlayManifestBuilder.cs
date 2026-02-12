using System.Text.Json;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Builds overlay manifest for viewer plugin architecture.
/// </summary>
public sealed class OverlayManifestBuilder
{
    /// <summary>
    /// Builds overlay manifest JSON for a map/version.
    /// </summary>
    /// <param name="viewerDir">Viewer output directory</param>
    /// <param name="mapName">Map name</param>
    /// <param name="version">Version string</param>
    /// <returns>Result with manifest path</returns>
    public ManifestBuildResult Build(string viewerDir, string mapName, string version)
    {
        try
        {
            var overlaysRoot = Path.Combine(viewerDir, "overlays", version, mapName);
            var manifestPath = Path.Combine(viewerDir, "overlays", "metadata.json");

            // Check which overlay types exist
            var terrainDir = Path.Combine(overlaysRoot, "terrain_complete");
            var objectsDir = Path.Combine(overlaysRoot, "objects_combined");
            var shadowDir = Path.Combine(overlaysRoot, "shadow_map");

            var hasTerrain = Directory.Exists(terrainDir) && Directory.EnumerateFiles(terrainDir, "*.json").Any();
            var hasObjects = Directory.Exists(objectsDir) && Directory.EnumerateFiles(objectsDir, "*.json").Any();
            var hasShadow = Directory.Exists(shadowDir) && Directory.EnumerateFiles(shadowDir, "*.json").Any();

            var overlays = new List<object>();

            if (hasTerrain)
            {
                overlays.Add(new
                {
                    id = "terrain.properties",
                    plugin = "terrain",
                    title = "Terrain Properties",
                    tiles = "complete",
                    resources = new
                    {
                        tilePattern = $"overlays/{version}/{mapName}/terrain_complete/tile_{{col}}_{{row}}.json"
                    }
                });
            }

            if (hasObjects)
            {
                overlays.Add(new
                {
                    id = "objects.combined",
                    plugin = "objects",
                    subtype = "combined",
                    title = "Object Placements",
                    tiles = "complete",
                    resources = new
                    {
                        tilePattern = $"overlays/{version}/{mapName}/objects_combined/tile_{{col}}_{{row}}.json"
                    }
                });
            }

            if (hasShadow)
            {
                overlays.Add(new
                {
                    id = "shadow.overview",
                    plugin = "shadow",
                    title = "Shadow Map",
                    tiles = "sparse",
                    resources = new
                    {
                        metadataPattern = $"overlays/{version}/{mapName}/shadow_map/tile_{{col}}_{{row}}.json",
                        imagePattern = $"overlays/{version}/{mapName}/shadow_map/{{filename}}"
                    }
                });
            }

            var manifest = new
            {
                version,
                map = mapName,
                overlays
            };

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, options));

            return new ManifestBuildResult(
                ManifestPath: manifestPath,
                OverlayCount: overlays.Count,
                Success: true);
        }
        catch (Exception ex)
        {
            return new ManifestBuildResult(
                string.Empty,
                0,
                Success: false,
                ErrorMessage: $"Manifest building failed: {ex.Message}");
        }
    }
}
