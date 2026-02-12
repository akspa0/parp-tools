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

        // ===== FILTERING PHILOSOPHY =====
        // 
        // COMPUTE ACTUAL TILE FROM COORDINATES:
        // ADT placement coords are ABSOLUTE offsets from map NW corner (0,0).
        // The ADT filename may not match the object's actual tile location!
        // 
        // Example: Dark Portal at coords (1086, 1099):
        // - Stored in ADT files development_1_1.adt and development_2_1.adt
        // - But world coords (15980, 15966) place it in tile (2,2)!
        // 
        // We MUST compute the actual owning tile from coordinates, not trust filename.
        //
        // See AdtPlacementsExtractor.cs for details on coordinate system issues.
        
        var filtered = entriesList
            .Where(e => e.Map.Equals(map, StringComparison.OrdinalIgnoreCase))
            .Where(e => !(e.UniqueId == 0 && e.AssetPath == "_dummy_tile_marker")) // Skip dummy markers early
            .Select(e => new { Entry = e, ActualTile = ComputeActualTile(e) })
            .Where(x => x.ActualTile.Row == tileRow && x.ActualTile.Col == tileCol)
            .Select(x => x.Entry)
            .ToList();
        
        Console.WriteLine($"[OverlayBuilder] After tile filter: {filtered.Count} entries for tile ({tileRow},{tileCol})");

        // Deduplicate by UniqueID to handle cross-tile object spanning
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
    /// Computes which tile actually owns this entry based on its coordinates.
    /// ADT placement coords are absolute offsets from (0,0), so we convert to world
    /// coords and compute the tile indices.
    /// </summary>
    private static (int Row, int Col) ComputeActualTile(AssetTimelineDetailedEntry entry)
    {
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE; // 17066.67
        
        // Convert ADT placement coords to world coords
        double worldX = MAP_CENTER - entry.WorldX;
        double worldY = MAP_CENTER - entry.WorldZ;
        
        // Compute tile indices from world coords
        var (tileRow, tileCol) = CoordinateTransformer.ComputeTileIndices(worldX, worldY);
        
        // DEBUG: Log samples to verify computation (skip dummy markers)
        if (entry.UniqueId > 0 && entry.UniqueId % 100000 == 0) // Sample every 100k
        {
            Console.WriteLine($"[ComputeActualTile] UID={entry.UniqueId}: ADT coords ({entry.WorldX:F2}, {entry.WorldZ:F2}) → world ({worldX:F2}, {worldY:F2}) → tile ({tileRow},{tileCol}) vs CSV tile ({entry.TileRow},{entry.TileCol})");
        }
        
        return (tileRow, tileCol);
    }

    private static object? BuildPoint(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        // Filter dummy marker entries
        if (entry.UniqueId == 0 && entry.AssetPath == "_dummy_tile_marker")
        {
            return null;
        }
        
        // Entry is already filtered by actual tile computed from coordinates
        // Compute rendering position from raw ADT coordinates
        
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE; // 17066.67
        
        // CRITICAL: ADT MDDF/MODF coordinate system transformation!
        // 
        // From ADT wiki - MDDF/MODF (Placement) coordinate system:
        // - Coords are stored as ABSOLUTE offsets from NW corner (tile 0,0)
        // - x' = 32 * TILESIZE - x (INVERTED from our viewer coords)
        // - z' = 32 * TILESIZE - z (INVERTED from our viewer coords)
        //
        // ADT coords appear to be consistent: absolute offsets from map NW corner.
        // We ALWAYS transform: worldCoord = MAP_CENTER - adtCoord
        //
        // Example: Dark Portal at (1086, 1099) in tile (1,1):
        // - These are offsets from (0,0), NOT tile-local coords within tile (1,1)
        // - Transform: worldX = 17066 - 1086 = 15980 (places it ~2 tiles from corner)
        
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
                // Negate X to match terrain orientation (mirrored along X axis)
                x = Math.Round(-entry.WorldX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(entry.WorldY, 2, MidpointRounding.AwayFromZero),
                z = Math.Round(entry.WorldZ, 2, MidpointRounding.AwayFromZero)
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
