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

        var filtered = entriesList
            .Where(e => e.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && e.TileRow == tileRow && e.TileCol == tileCol)
            .ToList();
        
        Console.WriteLine($"[OverlayBuilder] After filter: {filtered.Count} entries for tile ({tileRow},{tileCol})");
        
        if (filtered.Count == 0 && entriesList.Count > 0)
        {
            // Debug: Show what we're NOT matching
            var sample = entriesList.FirstOrDefault();
            if (sample != null)
            {
                Console.WriteLine($"[OverlayBuilder] Sample entry: Map='{sample.Map}', TileRow={sample.TileRow}, TileCol={sample.TileCol}");
                Console.WriteLine($"[OverlayBuilder] Looking for: Map='{map}', TileRow={tileRow}, TileCol={tileCol}");
            }
        }

        // Deduplicate by UniqueID before building overlays
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

    private static object? BuildPoint(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        // Filter dummy marker entries used only for tile indexing
        if (entry.UniqueId == 0 && entry.AssetPath == "_dummy_tile_marker")
        {
            return null;
        }
        
        // Trust source tile data - filter objects that don't belong to this tile
        if (entry.TileRow != tileRow || entry.TileCol != tileCol)
        {
            Console.WriteLine($"[OverlayBuilder] Skipped UID {entry.UniqueId}: tile mismatch (entry={entry.TileRow},{entry.TileCol} vs expected={tileRow},{tileCol})");
            return null;
        }
        
        // CRITICAL: Cross-tile duplicate detection
        // Objects spanning tile boundaries appear in multiple ADT files for culling
        // Strategy: Keep ONE instance - prefer the tile where coordinates place it
        // If coordinates are wrong (common in alpha data), keep it on ANY tile to avoid data loss
        
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE;
        
        // Transform placement coordinates to world coordinates
        double worldX = MAP_CENTER - entry.WorldX;  // placement X → worldX (col axis)
        double worldY = MAP_CENTER - entry.WorldZ;  // placement Z → worldY (row axis)
        
        // Compute which tile these coords SHOULD place object on
        var (computedRow, computedCol) = CoordinateTransformer.ComputeTileIndices(worldX, worldY);
        
        // Check if coordinates match this tile (within small tolerance for boundary objects)
        bool coordsMatchTile = (computedRow == tileRow && computedCol == tileCol);
        
        // Check if this is likely a duplicate by being on an adjacent tile
        int rowDiff = Math.Abs(computedRow - tileRow);
        int colDiff = Math.Abs(computedCol - tileCol);
        bool isAdjacentTile = (rowDiff <= 1 && colDiff <= 1) && !coordsMatchTile;
        
        // KEEP if: coordinates place it here OR it's far from computed tile (bad data - keep anyway)
        // FILTER if: on adjacent tile and coords place it elsewhere (clear duplicate)
        if (isAdjacentTile)
        {
            // This is a clear cross-tile duplicate - skip it
            // Primary instance (on correct tile) will be rendered
            return null;
        }
        
        // Keep this instance - either coords match or we don't have better info
        // This ensures at least ONE instance renders even if coords are wrong

        // DISABLED: World coordinates in ADT files often don't match the tile filename
        // This is a data integrity issue in the source files, not our code
        // For loose ADT analysis, we trust the tile filename over the world coordinates
        /*
        var (computedRow, computedCol) = CoordinateTransformer.ComputeTileIndices(entry.WorldX, entry.WorldY);
        if (computedRow != tileRow || computedCol != tileCol)
        {
            Console.WriteLine($"[OverlayBuilder] Filtered UID {entry.UniqueId}: world coords ({entry.WorldX:F1}, {entry.WorldY:F1}) compute to tile ({computedRow},{computedCol}) but stored as ({tileRow},{tileCol})");
            return null;
        }
        */

        // Transform MDDF/MODF placement coordinates to world coordinates
        // worldX and worldY are already computed above for duplicate detection - reuse them
        var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(worldX, worldY, tileRow, tileCol);
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);
        
        // DISABLED: Bounds validation - ADT placement coordinates are unreliable
        // Many objects legitimately span tile boundaries or have coordinate mismatches
        // Since we already filter by entry.TileRow/TileCol, trust the tile assignment
        /*
        if (localX < -0.1 || localX > 1.1 || localY < -0.1 || localY > 1.1)
        {
            if (entry.UniqueId % 500 == 0) // Log occasionally
            {
                Console.WriteLine($"[OverlayBuilder] Filtered UID {entry.UniqueId} from tile ({tileRow},{tileCol}): " +
                    $"computed position ({localX:F3}, {localY:F3}) outside tile bounds");
            }
            return null;
        }
        */

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
