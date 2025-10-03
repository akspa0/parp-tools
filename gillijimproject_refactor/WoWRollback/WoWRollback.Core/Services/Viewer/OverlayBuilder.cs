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

        var filtered = entries
            .Where(e => e.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && e.TileRow == tileRow && e.TileCol == tileCol)
            .ToList();

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
        // Trust source tile data - filter objects that don't belong to this tile
        if (entry.TileRow != tileRow || entry.TileCol != tileCol)
        {
            Console.WriteLine($"[OverlayBuilder] Skipped UID {entry.UniqueId}: tile mismatch (entry={entry.TileRow},{entry.TileCol} vs expected={tileRow},{tileCol})");
            return null;
        }

        // Validate world coordinates match the tile before computing pixels
        var (computedRow, computedCol) = CoordinateTransformer.ComputeTileIndices(entry.WorldX, entry.WorldY);
        if (computedRow != tileRow || computedCol != tileCol)
        {
            Console.WriteLine($"[OverlayBuilder] Filtered UID {entry.UniqueId}: world coords ({entry.WorldX:F1}, {entry.WorldY:F1}) compute to tile ({computedRow},{computedCol}) but stored as ({tileRow},{tileCol})");
            return null;
        }

        var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(entry.WorldX, entry.WorldY, tileRow, tileCol);
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

        // Final safety validation: reject coordinates outside reasonable tile bounds
        const double safetyMargin = 25.0; // Tighter margin
        if (pixelX < -safetyMargin || pixelX > options.MinimapWidth + safetyMargin ||
            pixelY < -safetyMargin || pixelY > options.MinimapHeight + safetyMargin)
        {
            Console.WriteLine($"[OverlayBuilder] Filtered UID {entry.UniqueId}: pixel ({pixelX:F1}, {pixelY:F1}) outside tile {tileRow},{tileCol} bounds (world: {entry.WorldX:F1}, {entry.WorldY:F1})");
            return null;
        }

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
