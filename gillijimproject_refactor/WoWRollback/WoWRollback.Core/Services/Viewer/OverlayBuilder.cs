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

        List<object> layers;

        if (allVersions is null)
        {
            layers = filtered
                .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
                .Select(versionGroup => new
                {
                    version = versionGroup.Key,
                    kinds = versionGroup
                        .GroupBy(e => e.Kind)
                        .Select(kindGroup => new
                        {
                            kind = kindGroup.Key.ToString(),
                            points = kindGroup.Select(e => BuildPoint(e, tileRow, tileCol, options)).ToList()
                        })
                        .ToList()
                })
                .OrderBy(layer => layer.version, StringComparer.OrdinalIgnoreCase)
                .Cast<object>()
                .ToList();
        }
        else
        {
            var byVersion = filtered
                .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
                .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

            layers = allVersions
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
                .Select(version => new
                {
                    version,
                    kinds = (byVersion.TryGetValue(version, out var list)
                            ? (IEnumerable<AssetTimelineDetailedEntry>)list
                            : Array.Empty<AssetTimelineDetailedEntry>())
                        .GroupBy(e => e.Kind)
                        .Select(kindGroup => new
                        {
                            kind = kindGroup.Key.ToString(),
                            points = kindGroup.Select(e => BuildPoint(e, tileRow, tileCol, options)).ToList()
                        })
                        .ToList()
                })
                .Cast<object>()
                .ToList();
        }

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

    private static object BuildPoint(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(entry.WorldX, entry.WorldY, tileRow, tileCol);
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

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
                x = entry.WorldX,
                y = entry.WorldY,
                z = entry.WorldZ
            },
            local = new
            {
                x = Math.Round(localX, 6, MidpointRounding.AwayFromZero),
                y = Math.Round(localY, 6, MidpointRounding.AwayFromZero)
            },
            pixel = new
            {
                x = Math.Round(pixelX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(pixelY, 2, MidpointRounding.AwayFromZero)
            }
        };
    }
}
