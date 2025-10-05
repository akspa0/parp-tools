using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds overlay_manifest.json files for plugin-driven viewer (Phase 2).
/// Manifests describe available overlays without viewer needing to probe files.
/// </summary>
public sealed class OverlayManifestBuilder
{
    /// <summary>
    /// Generates an overlay manifest for a specific version/map combination.
    /// </summary>
    /// <param name="version">Game version (e.g., "0.5.3.3368")</param>
    /// <param name="mapName">Map name (e.g., "DeadminesInstance")</param>
    /// <param name="tiles">List of available tiles (row, col)</param>
    /// <param name="overlayDirectory">Root overlay directory for this version/map</param>
    /// <param name="hasTerrainData">Whether terrain overlays exist</param>
    /// <param name="hasShadowData">Whether shadow overlays exist</param>
    /// <returns>JSON string representing the manifest</returns>
    public string BuildManifest(
        string version,
        string mapName,
        IEnumerable<(int Row, int Col)> tiles,
        string overlayDirectory,
        bool hasTerrainData = false,
        bool hasShadowData = false)
    {
        ArgumentNullException.ThrowIfNull(version);
        ArgumentNullException.ThrowIfNull(mapName);
        ArgumentNullException.ThrowIfNull(tiles);

        var tileList = tiles.OrderBy(t => t.Row).ThenBy(t => t.Col).ToList();
        
        var manifest = new OverlayManifest
        {
            Version = version,
            Map = mapName,
            GeneratedAt = DateTime.UtcNow,
            Overlays = BuildOverlayList(mapName, tileList, hasTerrainData, hasShadowData),
            Tiles = BuildTileList(tileList)
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        return JsonSerializer.Serialize(manifest, options);
    }

    private static List<OverlayDefinition> BuildOverlayList(
        string mapName,
        List<(int Row, int Col)> tiles,
        bool hasTerrainData,
        bool hasShadowData)
    {
        var overlays = new List<OverlayDefinition>();

        // Object overlays (always present)
        overlays.Add(new OverlayDefinition
        {
            Id = "objects.combined",
            Plugin = "objects",
            Title = "All Objects (M2 + WMO)",
            Enabled = true,
            TilePattern = "combined/tile_r{row}_c{col}.json",
            TileCoverage = "complete" // All tiles have objects
        });

        overlays.Add(new OverlayDefinition
        {
            Id = "objects.m2",
            Plugin = "objects",
            Title = "M2 Models Only",
            Enabled = false,
            TilePattern = "m2/tile_r{row}_c{col}.json",
            TileCoverage = "complete"
        });

        overlays.Add(new OverlayDefinition
        {
            Id = "objects.wmo",
            Plugin = "objects",
            Title = "WMO Objects Only",
            Enabled = false,
            TilePattern = "wmo/tile_r{row}_c{col}.json",
            TileCoverage = "complete"
        });

        // Terrain overlays (if available)
        if (hasTerrainData)
        {
            overlays.Add(new OverlayDefinition
            {
                Id = "terrain.properties",
                Plugin = "terrain",
                Title = "Terrain Properties",
                Enabled = true,
                TilePattern = "terrain_complete/tile_{col}_{row}.json",
                TileCoverage = "sparse", // Not all tiles have terrain data
                Description = "Height maps, flags, liquids, area IDs"
            });
        }

        // Shadow overlays (if available)
        if (hasShadowData)
        {
            overlays.Add(new OverlayDefinition
            {
                Id = "shadow.overview",
                Plugin = "shadow",
                Title = "Shadow Maps",
                Enabled = true,
                Resources = new Dictionary<string, string>
                {
                    ["metadataPattern"] = "shadow_map/tile_{col}_{row}.json",
                    ["imagePattern"] = "shadow_map/{filename}"
                },
                TileCoverage = "sparse",
                Description = "Shadow map overlays from MCSH data"
            });
        }

        return overlays;
    }

    private static TileManifest BuildTileList(List<(int Row, int Col)> tiles)
    {
        return new TileManifest
        {
            Count = tiles.Count,
            Bounds = new TileBounds
            {
                MinRow = tiles.Min(t => t.Row),
                MaxRow = tiles.Max(t => t.Row),
                MinCol = tiles.Min(t => t.Col),
                MaxCol = tiles.Max(t => t.Col)
            },
            Tiles = tiles.Select(t => new TileEntry
            {
                Row = t.Row,
                Col = t.Col
            }).ToList()
        };
    }
}

// ============================================================================
// Manifest Schema Classes
// ============================================================================

/// <summary>
/// Root manifest structure for a version/map combination.
/// </summary>
public sealed class OverlayManifest
{
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;

    [JsonPropertyName("map")]
    public string Map { get; set; } = string.Empty;

    [JsonPropertyName("generatedAt")]
    public DateTime GeneratedAt { get; set; }

    [JsonPropertyName("overlays")]
    public List<OverlayDefinition> Overlays { get; set; } = new();

    [JsonPropertyName("tiles")]
    public TileManifest Tiles { get; set; } = new();
}

/// <summary>
/// Defines a single overlay layer.
/// </summary>
public sealed class OverlayDefinition
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("plugin")]
    public string Plugin { get; set; } = string.Empty;

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("tilePattern")]
    public string? TilePattern { get; set; }

    [JsonPropertyName("tileCoverage")]
    public string TileCoverage { get; set; } = "complete"; // "complete" or "sparse"

    [JsonPropertyName("description")]
    public string? Description { get; set; }

    [JsonPropertyName("resources")]
    public Dictionary<string, string>? Resources { get; set; }
}

/// <summary>
/// Tile coverage information.
/// </summary>
public sealed class TileManifest
{
    [JsonPropertyName("count")]
    public int Count { get; set; }

    [JsonPropertyName("bounds")]
    public TileBounds Bounds { get; set; } = new();

    [JsonPropertyName("tiles")]
    public List<TileEntry> Tiles { get; set; } = new();
}

/// <summary>
/// Tile grid bounds.
/// </summary>
public sealed class TileBounds
{
    [JsonPropertyName("minRow")]
    public int MinRow { get; set; }

    [JsonPropertyName("maxRow")]
    public int MaxRow { get; set; }

    [JsonPropertyName("minCol")]
    public int MinCol { get; set; }

    [JsonPropertyName("maxCol")]
    public int MaxCol { get; set; }
}

/// <summary>
/// Individual tile entry.
/// </summary>
public sealed class TileEntry
{
    [JsonPropertyName("row")]
    public int Row { get; set; }

    [JsonPropertyName("col")]
    public int Col { get; set; }
}
