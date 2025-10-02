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
    {
        ArgumentNullException.ThrowIfNull(map);
        ArgumentNullException.ThrowIfNull(entries);
        options ??= ViewerOptions.CreateDefault();

        static bool IsInTile(AssetTimelineDetailedEntry e, string targetMap, int r, int c)
        {
            if (!e.Map.Equals(targetMap, StringComparison.OrdinalIgnoreCase)) return false;
            var tr = e.TileRow;
            var tc = e.TileCol;
            var hasTile = tr >= 0 && tr <= 63 && tc >= 0 && tc <= 63;
            if (hasTile) return tr == r && tc == c;
            var computed = CoordinateTransformer.ComputeTileIndices(e.WorldX, e.WorldY);
            return computed.TileRow == r && computed.TileCol == c;
        }

        var filtered = entries
            .Where(e => IsInTile(e, map, tileRow, tileCol))
            .ToList();

        var layers = filtered
            .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .Select(versionGroup => new
            {
                version = versionGroup.Key,
                kinds = versionGroup
                    .GroupBy(e => e.Kind)
                    .Select(kindGroup => new
                    {
                        kind = kindGroup.Key.ToString(),
                        points = kindGroup
                            .Select(e => BuildPoint(e, tileRow, tileCol, options))
                            .Where(p => p is not null)
                            .ToList()
                    })
                    .ToList()
            })
            .OrderBy(layer => layer.version, StringComparer.OrdinalIgnoreCase)
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
            Converters = { new JsonStringEnumConverter() },
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        });
    }

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
    {
        ArgumentNullException.ThrowIfNull(map);
        ArgumentNullException.ThrowIfNull(entries);
        ArgumentNullException.ThrowIfNull(allVersions);
        options ??= ViewerOptions.CreateDefault();

        var filtered = entries
            .Where(e =>
            {
                if (!e.Map.Equals(map, StringComparison.OrdinalIgnoreCase)) return false;
                var tr = e.TileRow; var tc = e.TileCol;
                var hasTile = tr >= 0 && tr <= 63 && tc >= 0 && tc <= 63;
                if (hasTile) return tr == tileRow && tc == tileCol;
                var computed = CoordinateTransformer.ComputeTileIndices(e.WorldX, e.WorldY);
                return computed.TileRow == tileRow && computed.TileCol == tileCol;
            })
            .ToList();

        var byVersion = filtered
            .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

        var layers = allVersions
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
                        points = kindGroup
                            .Select(e => BuildPoint(e, tileRow, tileCol, options))
                            .Where(p => p is not null)
                            .ToList()
                    })
                    .ToList()
            })
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
            Converters = { new JsonStringEnumConverter() },
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        });
    }

    private static object BuildPoint(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        static double Sanitize(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) return 0.0;
            return value;
        }

        var safeWorldX = Sanitize(entry.WorldX);
        var safeWorldY = Sanitize(entry.WorldY);
        var safeWorldZ = Sanitize(entry.WorldZ);

        // Compute RAW local and clamp to [0,1]; do not drop outright to avoid losing MDX/M2 on early alphas
        var (localRawX, localRawY) = CoordinateTransformer.ComputeLocalCoordinatesRaw(safeWorldX, safeWorldY, tileRow, tileCol);
        var (localX, localY) = (Math.Clamp(localRawX, 0.0, 1.0), Math.Clamp(localRawY, 0.0, 1.0));
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

        static bool IsWorldReliable(double x, double y) => !(double.IsNaN(x) || double.IsNaN(y) || (x == 0.0 && y == 0.0));
        var worldReliable = IsWorldReliable(safeWorldX, safeWorldY);
        int worldRow = -1, worldCol = -1;
        bool isAdjacentRef = false; // disable adjacency marking until transforms validated
        if (worldReliable)
        {
            var idx = CoordinateTransformer.ComputeTileIndices(safeWorldX, safeWorldY);
            worldRow = idx.TileRow;
            worldCol = idx.TileCol;
            // keep isAdjacentRef=false for now to avoid hiding WMOs when toggle is off
        }

        static string ClassifyType(AssetTimelineDetailedEntry e)
        {
            var kind = e.Kind.ToString();
            if (!string.IsNullOrWhiteSpace(kind))
            {
                var k = kind.Trim().ToLowerInvariant();
                if (k.Contains("wmo")) return "wmo";
                if (k.Contains("m2") || k.Contains("mdx")) return "m2";
            }
            var ext = (e.Extension ?? string.Empty).Trim().TrimStart('.').ToLowerInvariant();
            return ext switch
            {
                "wmo" => "wmo",
                "m2" => "m2",
                "mdx" => "m2",
                _ => "other"
            };
        }

        // Determine type and apply permanent WMO-specific Y flip at generation time
        var typeLabel = ClassifyType(entry);
        if (typeLabel == "wmo")
        {
            pixelY = options.MinimapHeight - pixelY;
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
            type = typeLabel,
            // Use raw world coordinates from source (MDDF/MODF) for display
            world = new { x = safeWorldX, y = safeWorldY, z = safeWorldZ },
            local = new
            {
                x = Math.Round(localX, 6, MidpointRounding.AwayFromZero),
                y = Math.Round(localY, 6, MidpointRounding.AwayFromZero)
            },
            pixel = new
            {
                x = Math.Round(pixelX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(pixelY, 2, MidpointRounding.AwayFromZero)
            },
            tileOfWorld = new { row = worldRow, col = worldCol },
            isAdjacentRef
        };
    }
}
