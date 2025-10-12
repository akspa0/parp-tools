using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds per-tile overlay JSON grouping placements by version and kit/subkit metadata.
/// </summary>
public sealed class OverlayBuilder
{
    /// <summary>
    /// Generates overlay JSON payload for a given map/tile.
    /// </summary>
    /// <param name="map">Map name.</param>
    /// <param name="tileRow">Row index (0-63).</param>
    /// <param name="tileCol">Column index (0-63).</param>
    /// <param name="entries">Placement entries enriched with world coordinates.</param>
    /// <param name="options">Viewer configuration.</param>
    /// <returns>Serialized JSON text.</returns>
    public string BuildOverlayJson(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<AssetTimelineDetailedEntry> entries,
        ViewerOptions options)
        => BuildOverlayJsonInternal(map, tileRow, tileCol, entries, options, allVersions: null);

    /// <summary>
    /// Generates overlay JSON, forcing a layer entry for every version in <paramref name="allVersions"/>.
    /// Versions without objects will have an empty kinds array.
    /// </summary>
    public string BuildOverlayJson(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<AssetTimelineDetailedEntry> entries,
        IEnumerable<string> allVersions,
        ViewerOptions options)
        => BuildOverlayJsonInternal(map, tileRow, tileCol, entries, options, allVersions);

    public string BuildOverlayJsonByKind(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<AssetTimelineDetailedEntry> entries,
        IEnumerable<string> allVersions,
        ViewerOptions options,
        PlacementKind kind)
    {
        ArgumentNullException.ThrowIfNull(entries);
        return BuildOverlayJsonInternal(
            map,
            tileRow,
            tileCol,
            entries.Where(e => e.Kind == kind),
            options,
            allVersions);
    }

    private static string BuildOverlayJsonInternal(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<AssetTimelineDetailedEntry> entries,
        ViewerOptions? options,
        IEnumerable<string>? allVersions)
    {
        ArgumentNullException.ThrowIfNull(map);
        ArgumentNullException.ThrowIfNull(entries);

        options ??= ViewerOptions.CreateDefault();

        var entriesList = entries.ToList();
        Console.WriteLine($"[OverlayBuilder] BuildOverlay for {map} tile ({tileRow},{tileCol}): Received {entriesList.Count} total entries");

        // COORDINATE-BASED FILTERING: Use object coordinates to determine tile membership
        // This ensures objects appear on the correct tile regardless of entry.TileRow/TileCol
        var filtered = entriesList
            .Where(e => e.Map.Equals(map, StringComparison.OrdinalIgnoreCase))
            .Select(e => new { Entry = e, TileIndices = ComputeTileFromCoordinates(e) })
            .Where(x => x.TileIndices.Row == tileRow && x.TileIndices.Col == tileCol)
            .Select(x => x.Entry)
            .ToList();
        
        Console.WriteLine($"[OverlayBuilder] After tile-based filter: {filtered.Count} entries for tile ({tileRow},{tileCol})");
        
        if (filtered.Count == 0 && entriesList.Count > 0)
        {
            // Debug: Show sample and what tiles we have
            var sample = entriesList.FirstOrDefault();
            if (sample != null)
            {
                Console.WriteLine($"[OverlayBuilder] Sample entry: Map='{sample.Map}', TileRow={sample.TileRow}, TileCol={sample.TileCol}, WorldX={sample.WorldX:F1}, WorldZ={sample.WorldZ:F1}");
                Console.WriteLine($"[OverlayBuilder] Looking for tile: ({tileRow},{tileCol})");
                
                // Show distribution of tiles in the input
                var tileDistribution = entriesList
                    .GroupBy(e => (e.TileRow, e.TileCol))
                    .OrderBy(g => g.Key.TileRow)
                    .ThenBy(g => g.Key.TileCol)
                    .Take(10)
                    .Select(g => $"({g.Key.TileRow},{g.Key.TileCol}):{g.Count()}")
                    .ToList();
                Console.WriteLine($"[OverlayBuilder] Tile distribution (first 10): {string.Join(", ", tileDistribution)}");
            }
        }

        // Deduplicate by UniqueID (in case same object appears with slightly different coords)
        var deduplicated = DeduplicateByUniqueId(filtered, tileRow, tileCol);

        var layers = (allVersions ?? deduplicated.Select(e => e.Version))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .Select(version => new
            {
                version,
                kinds = deduplicated
                    .Where(e => e.Version.Equals(version, StringComparison.OrdinalIgnoreCase))
                    .GroupBy(e => e.Kind)
                    .Select(kindGroup => new
                    {
                        kind = kindGroup.Key.ToString(),
                        points = kindGroup
                            .Select(e => BuildPoint(e, tileRow, tileCol, options))
                            .Where(p => p != null)
                            .Cast<object>()
                            .ToList()
                    })
                    .ToList()
            })
            .Cast<object>()
            .ToList();

        var payload = new
        {
            map,
            tile = new { row = tileRow, col = tileCol },
            minimap = new { width = options.MinimapWidth, height = options.MinimapHeight },
            layers
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        });
    }

    /// <summary>
    /// Returns which tile an entry belongs to.
    /// Uses the tile coordinates from the CSV (which tile the ADT file was from).
    /// </summary>
    private static (int Row, int Col) ComputeTileFromCoordinates(AssetTimelineDetailedEntry entry)
    {
        // IMPORTANT: entry.WorldX/WorldZ are TILE-LOCAL coordinates (0-533 range)
        // They come from ADT MDDF/MODF Position fields which are relative to the tile
        // The CSV already tells us which tile via entry.TileRow/TileCol
        // So just use those directly!
        return (entry.TileRow, entry.TileCol);
    }

    private static object? BuildPoint(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        // Filter dummy marker entries
        if (entry.UniqueId == 0 && entry.AssetPath == "_dummy_tile_marker")
        {
            return null;
        }
        
        // At this point, entry is already filtered to this tile by ComputeTileFromCoordinates
        // Just compute rendering position
        
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE;
        
        double worldX = MAP_CENTER - entry.WorldX;
        double worldY = MAP_CENTER - entry.WorldZ;
        
        var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(worldX, worldY, tileRow, tileCol);
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

        return new
        {
            uniqueId = entry.UniqueId,
            __kind = entry.Kind.ToString(),
            assetPath = entry.AssetPath,
            folder = entry.Folder,
            category = entry.Category,
            subcategory = entry.Subcategory,
            designKit = entry.DesignKit,
            sourceRule = entry.SourceRule,
            kitRoot = entry.KitRoot,
            subkitPath = entry.SubkitPath,
            subkitTop = entry.SubkitTop,
            subkitDepth = entry.SubkitDepth,
            fileName = entry.FileName,
            fileStem = entry.FileStem,
            extension = entry.Extension,
            placement = new
            {
                // Raw ADT placement coordinates (MDDF/MODF Position fields)
                x = Math.Round(entry.WorldX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(entry.WorldY, 2, MidpointRounding.AwayFromZero),
                z = Math.Round(entry.WorldZ, 2, MidpointRounding.AwayFromZero)
            },
            world = new
            {
                // Transformed world coordinates (used for rendering position)
                x = Math.Round(worldX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(worldY, 2, MidpointRounding.AwayFromZero),
                z = Math.Round(entry.WorldY, 2, MidpointRounding.AwayFromZero)  // Height unchanged
            },
            rotation = new
            {
                // Rotation in degrees
                x = Math.Round(entry.RotationX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(entry.RotationY, 2, MidpointRounding.AwayFromZero),
                z = Math.Round(entry.RotationZ, 2, MidpointRounding.AwayFromZero)
            },
            scale = Math.Round(entry.Scale, 4, MidpointRounding.AwayFromZero),
            local = new
            {
                // 6 decimal places for normalized 0-1 coordinates
                x = Math.Round(localX, 6, MidpointRounding.AwayFromZero),
                y = Math.Round(localY, 6, MidpointRounding.AwayFromZero)
            },
            pixel = new
            {
                // 3 decimal places for pixel coordinates
                x = Math.Round(pixelX, 3, MidpointRounding.AwayFromZero),
                y = Math.Round(pixelY, 3, MidpointRounding.AwayFromZero)
            }
        };
    }

    /// <summary>
    /// Deduplicates entries by UniqueID, keeping only ONE instance per ID.
    /// Prefers the instance where coordinates best match the expected tile location.
    /// </summary>
    private static List<AssetTimelineDetailedEntry> DeduplicateByUniqueId(
        List<AssetTimelineDetailedEntry> entries,
        int tileRow,
        int tileCol)
    {
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE;
        
        var byUniqueId = entries.GroupBy(e => e.UniqueId);
        var deduplicated = new List<AssetTimelineDetailedEntry>();
        
        foreach (var group in byUniqueId)
        {
            if (group.Count() == 1)
            {
                // No duplicates - keep it
                deduplicated.Add(group.First());
                continue;
            }
            
            // Multiple instances of same UniqueID - pick the best one
            // Prefer the one where coordinates match this tile
            AssetTimelineDetailedEntry? best = null;
            double bestScore = double.MaxValue;
            
            foreach (var entry in group)
            {
                double worldX = MAP_CENTER - entry.WorldX;
                double worldY = MAP_CENTER - entry.WorldZ;
                var (computedRow, computedCol) = CoordinateTransformer.ComputeTileIndices(worldX, worldY);
                
                // Calculate "distance" from expected tile
                double score = Math.Abs(computedRow - tileRow) + Math.Abs(computedCol - tileCol);
                
                if (score < bestScore)
                {
                    bestScore = score;
                    best = entry;
                }
            }
            
            if (best != null)
            {
                deduplicated.Add(best);
            }
        }
        
        int duplicatesRemoved = entries.Count - deduplicated.Count;
        if (duplicatesRemoved > 0)
        {
            Console.WriteLine($"[OverlayBuilder] Removed {duplicatesRemoved} duplicate UniqueIDs for tile ({tileRow},{tileCol})");
        }
        
        return deduplicated;
    }
}
