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

        var layers = (allVersions ?? filtered.Select(e => e.Version))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .Select(version => new
            {
                version,
                kinds = filtered
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
        
        // CRITICAL: Detect cross-tile duplicates
        // Objects that span tile boundaries are duplicated in ADT files for culling/LOD
        // Only show them on their "primary" tile (where coordinates actually place them)
        const double TILESIZE = 533.33333;
        const double MAP_CENTER = 32.0 * TILESIZE;
        
        // Transform placement coordinates to world coordinates (same as used for rendering)
        double worldX = MAP_CENTER - entry.WorldX;  // placement X → worldX (col axis)
        double worldY = MAP_CENTER - entry.WorldZ;  // placement Z → worldY (row axis)
        
        // Use CoordinateTransformer to compute which tile these world coords belong to
        var (computedRow, computedCol) = CoordinateTransformer.ComputeTileIndices(worldX, worldY);
        
        // If coordinates place object on different tile, this is a cross-tile duplicate
        bool isCrossTileDuplicate = (computedRow != tileRow || computedCol != tileCol);
        
        // TODO: Future enhancement - include cross-tile duplicates with special flag for UI toggle
        // For now, completely filter them out to avoid duplicate rendering
        if (isCrossTileDuplicate)
        {
            return null;
        }

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
            world = new
            {
                // Preserve full precision for world coordinates
                x = Math.Round(entry.WorldX, 6, MidpointRounding.AwayFromZero),
                y = Math.Round(entry.WorldY, 6, MidpointRounding.AwayFromZero),
                z = Math.Round(entry.WorldZ, 6, MidpointRounding.AwayFromZero)
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
}
