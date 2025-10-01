using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Orchestrates viewer asset generation (minimaps, overlays, diffs, index/config).
/// </summary>
public sealed class ViewerReportWriter
{
    private readonly MinimapComposer _minimapComposer;
    private readonly OverlayBuilder _overlayBuilder;
    private readonly OverlayDiffBuilder _diffBuilder;

    public ViewerReportWriter()
        : this(new MinimapComposer(), new OverlayBuilder(), new OverlayDiffBuilder())
    {
    }

    private static string ResolveMinimapExtension(ViewerOptions options)
    {
        var fmt = (options.MinimapFormat ?? "png").Trim().ToLowerInvariant();
#if HAS_WEBP
        if (fmt == "webp") return "webp";
#else
        if (fmt == "webp") return "jpg";
#endif
        return fmt switch
        {
            "jpeg" => "jpg",
            "jpg" => "jpg",
            _ => "png"
        };
    }

    public ViewerReportWriter(
        MinimapComposer minimapComposer,
        OverlayBuilder overlayBuilder,
        OverlayDiffBuilder diffBuilder)
    {
        _minimapComposer = minimapComposer ?? throw new ArgumentNullException(nameof(minimapComposer));
        _overlayBuilder = overlayBuilder ?? throw new ArgumentNullException(nameof(overlayBuilder));
        _diffBuilder = diffBuilder ?? throw new ArgumentNullException(nameof(diffBuilder));
    }

    /// <summary>
    /// Generates viewer artifacts for the supplied comparison result.
    /// </summary>
    /// <returns>Absolute path to the viewer root (or empty string when skipped).</returns>
    public string Generate(
        string comparisonDirectory,
        VersionComparisonResult result,
        ViewerOptions? options,
        (string Baseline, string Comparison)? diffPair = null)
    {
        ArgumentNullException.ThrowIfNull(comparisonDirectory);
        ArgumentNullException.ThrowIfNull(result);
        var resolvedOptions = options ?? ViewerOptions.CreateDefault();

        if (result.AssetTimelineDetailed is null || result.AssetTimelineDetailed.Count == 0)
            return string.Empty;

        var viewerRoot = Path.Combine(comparisonDirectory, "viewer");
        Directory.CreateDirectory(viewerRoot);

        var overlaysRoot = Path.Combine(viewerRoot, "overlays");
        var diffsRoot = Path.Combine(viewerRoot, "diffs");
        Directory.CreateDirectory(overlaysRoot);
        Directory.CreateDirectory(diffsRoot);

        var entries = result.AssetTimelineDetailed;
        var comparer = StringComparer.OrdinalIgnoreCase;
        var minimapLocator = MinimapLocator.Build(result.RootDirectory, result.Versions);

        // Collect map names from both placements and minimap sources
        var mapNames = new HashSet<string>(comparer);
        foreach (var m in entries.Select(e => e.Map)) mapNames.Add(m);
        foreach (var v in result.Versions)
        {
            try
            {
                foreach (var m in minimapLocator.EnumerateMaps(v)) mapNames.Add(m);
            }
            catch { /* ignore per-version lookup errors */ }
        }

        // Build EntryTiles with fallback to world->tile indices when TileRow/TileCol are missing or out of range
        var entryTilesByMap = new Dictionary<string, HashSet<(int Row, int Col)>>(comparer);
        foreach (var e in entries)
        {
            if (string.IsNullOrWhiteSpace(e.Map)) continue;
            if (!entryTilesByMap.TryGetValue(e.Map, out var set))
            {
                set = new HashSet<(int, int)>();
                entryTilesByMap[e.Map] = set;
            }
            int r = e.TileRow;
            int c = e.TileCol;
            bool valid = r >= 0 && r <= 63 && c >= 0 && c <= 63;
            if (!valid)
            {
                var (tr, tc) = CoordinateTransformer.ComputeTileIndices(e.WorldX, e.WorldY);
                r = tr; c = tc;
                valid = r >= 0 && r <= 63 && c >= 0 && c <= 63;
            }
            if (valid) set.Add((r, c));
        }

        var maps = mapNames
            .OrderBy(n => n, comparer)
            .Select(name => new
            {
                Name = name,
                EntryTiles = entryTilesByMap.TryGetValue(name, out var set)
                    ? set
                    : new HashSet<(int Row, int Col)>(),
                MinimapTiles = result.Versions
                    .SelectMany(v =>
                    {
                        try { return minimapLocator.EnumerateTiles(v, name); }
                        catch { return Array.Empty<(int Row, int Col)>(); }
                    })
                    .ToHashSet()
            })
            .ToList();

        var entriesByVersion = entries
            .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

        var chosenDefaultVersion = SelectDefaultVersion(resolvedOptions, result);
        var effectiveDiffPair = ResolveDiffPair(diffPair, resolvedOptions, result);

        var mapTileCatalog = new Dictionary<string, List<TileDescriptor>>(StringComparer.OrdinalIgnoreCase);

        // Resolve effective minimap file extension (webp may fall back to jpg if encoder is unavailable)
        var effectiveExt = ResolveMinimapExtension(resolvedOptions);

        foreach (var mapGroup in maps)
        {
            var mapName = mapGroup.Name;
            var safeMap = Sanitize(mapName);
            var mapOverlayDir = Path.Combine(overlaysRoot, safeMap);
            var mapDiffDir = Path.Combine(diffsRoot, safeMap);
            Directory.CreateDirectory(mapOverlayDir);
            Directory.CreateDirectory(mapDiffDir);

            // Compute union of tiles from placements and minimap sources
            var tileSet = new HashSet<(int Row, int Col)>();
            foreach (var t in mapGroup.EntryTiles) tileSet.Add(t);
            foreach (var t in mapGroup.MinimapTiles) tileSet.Add(t);

            // Late fallback: if tile set is empty but we have entries for this map, derive from world coords
            if (tileSet.Count == 0)
            {
                var fallbackEntries = entries.Where(e => string.Equals(e.Map, mapName, StringComparison.OrdinalIgnoreCase)).ToList();
                Console.WriteLine($"[viewer] Map={mapName} initial tiles=0; fallbackEntries={fallbackEntries.Count}");
                foreach (var e in fallbackEntries)
                {
                    var (tr, tc) = CoordinateTransformer.ComputeTileIndices(e.WorldX, e.WorldY);
                    if (tr >= 0 && tr <= 63 && tc >= 0 && tc <= 63)
                        tileSet.Add((tr, tc));
                }
            }

            var tiles = tileSet.OrderBy(t => t.Row).ThenBy(t => t.Col).ToList();

            Console.WriteLine($"[viewer] Map={mapName} tiles={tiles.Count}");

            foreach (var tileGroup in tiles)
            {
                var (row, col) = tileGroup;

                // Export minimap for every version (write placeholder if unavailable)
                var versionsForTile = result.Versions.OrderBy(v => v, comparer).ToList();

                foreach (var version in versionsForTile)
                {
                    var safeVersion = Sanitize(version);
                    var versionMinimapRoot = Path.Combine(viewerRoot, "minimap", safeVersion);
                    var versionMapMinimapDir = Path.Combine(versionMinimapRoot, safeMap);
                    Directory.CreateDirectory(versionMapMinimapDir);

                    var minimapFile = $"{mapName}_{col}_{row}.{effectiveExt}";
                    var minimapPath = Path.Combine(versionMapMinimapDir, minimapFile);

                    try
                    {
                        if (minimapLocator.TryGetTile(version, mapName, row, col, out var tileDescriptor))
                        {
                            using var tileStream = tileDescriptor.Open();
                            _minimapComposer.ComposeAsync(tileStream, minimapPath, resolvedOptions).GetAwaiter().GetResult();
                            Console.WriteLine($"[minimap] {version}/{mapName} r{row} c{col} -> {Path.GetFileName(minimapPath)}");
                        }
                        else
                        {
                            // No source tile found → emit placeholder so tile pages still work
                            _minimapComposer.WritePlaceholderAsync(minimapPath, resolvedOptions).GetAwaiter().GetResult();
                            Console.WriteLine($"[minimap] placeholder {version}/{mapName} r{row} c{col} -> {Path.GetFileName(minimapPath)}");
                        }
                    }
                    catch
                    {
                        // Any decode or IO failure → write placeholder and continue
                        _minimapComposer.WritePlaceholderAsync(minimapPath, resolvedOptions).GetAwaiter().GetResult();
                        Console.WriteLine($"[minimap] error->placeholder {version}/{mapName} r{row} c{col}");
                    }
                }

                var overlayPath = Path.Combine(mapOverlayDir, $"tile_r{row}_c{col}.json");
                // Always write overlay JSON, and include all versions (empty kinds when no objects)
                var overlayJson = _overlayBuilder.BuildOverlayJson(mapName, row, col, entries, result.Versions, resolvedOptions);
                File.WriteAllText(overlayPath, overlayJson);
                Console.WriteLine($"[overlay] {mapName} r{row} c{col} -> {Path.GetFileName(overlayPath)}");

                if (effectiveDiffPair is { } pair &&
                    entriesByVersion.TryGetValue(pair.Baseline, out var baselineEntries) &&
                    entriesByVersion.TryGetValue(pair.Comparison, out var comparisonEntries))
                {
                    var diffJson = _diffBuilder.BuildDiffJson(mapName, row, col, baselineEntries, comparisonEntries, resolvedOptions);
                    var diffPath = Path.Combine(mapDiffDir, $"tile_r{row}_c{col}.json");
                    File.WriteAllText(diffPath, diffJson);
                }

                if (!mapTileCatalog.TryGetValue(mapName, out var tileList))
                {
                    tileList = new List<TileDescriptor>();
                    mapTileCatalog[mapName] = tileList;
                }

                tileList.Add(new TileDescriptor(row, col, versionsForTile));
            }
        }

        WriteIndexJson(viewerRoot, result, mapTileCatalog, chosenDefaultVersion, effectiveDiffPair);
        WriteConfigJson(viewerRoot, resolvedOptions, chosenDefaultVersion, effectiveDiffPair, effectiveExt);
        CopyViewerAssets(viewerRoot);

        return viewerRoot;
    }

    private static string SelectDefaultVersion(ViewerOptions options, VersionComparisonResult result)
    {
        var requested = options.DefaultVersion;
        if (!string.IsNullOrWhiteSpace(requested))
        {
            var match = result.Versions.FirstOrDefault(v => v.Equals(requested, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrWhiteSpace(match))
                return match;
        }

        return result.Versions
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault() ?? result.Versions.First();
    }

    private static (string Baseline, string Comparison)? ResolveDiffPair(
        (string Baseline, string Comparison)? explicitPair,
        ViewerOptions options,
        VersionComparisonResult result)
    {
        static (string Baseline, string Comparison)? Normalize(
            (string Baseline, string Comparison)? pair,
            IReadOnlyList<string> versions)
        {
            if (pair is null) return null;
            var baseMatch = versions.FirstOrDefault(v => v.Equals(pair.Value.Baseline, StringComparison.OrdinalIgnoreCase));
            var compMatch = versions.FirstOrDefault(v => v.Equals(pair.Value.Comparison, StringComparison.OrdinalIgnoreCase));
            if (string.IsNullOrWhiteSpace(baseMatch) || string.IsNullOrWhiteSpace(compMatch)) return null;
            return (baseMatch, compMatch);
        }

        var normalized = Normalize(explicitPair, result.Versions)
            ?? Normalize(options.DiffPair, result.Versions);

        if (normalized is not null)
            return normalized;

        if (result.Versions.Count >= 2)
        {
            var ordered = result.Versions.OrderBy(v => v, StringComparer.OrdinalIgnoreCase).ToList();
            return (ordered[0], ordered[^1]);
        }

        return null;
    }

    private static void WriteIndexJson(
        string viewerRoot,
        VersionComparisonResult result,
        Dictionary<string, List<TileDescriptor>> mapTiles,
        string defaultVersion,
        (string Baseline, string Comparison)? diffPair)
    {
        var maps = mapTiles
            .OrderBy(kvp => kvp.Key, StringComparer.OrdinalIgnoreCase)
            .Select(kvp => new
            {
                map = kvp.Key,
                tiles = kvp.Value
                    .OrderBy(t => t.Row)
                    .ThenBy(t => t.Col)
                    .Select(t => new
                    {
                        row = t.Row,
                        col = t.Col,
                        versions = t.Versions
                    })
                    .ToList()
            })
            .ToList();

        var index = new
        {
            comparisonKey = result.ComparisonKey,
            defaultVersion,
            diff = diffPair is null ? null : new { baseline = diffPair.Value.Baseline, comparison = diffPair.Value.Comparison },
            versions = result.Versions,
            maps
        };

        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };

        var indexPath = Path.Combine(viewerRoot, "index.json");
        File.WriteAllText(indexPath, JsonSerializer.Serialize(index, options));
    }

    private static void WriteConfigJson(
        string viewerRoot,
        ViewerOptions options,
        string defaultVersion,
        (string Baseline, string Comparison)? diffPair,
        string effectiveExt)
    {
        var config = new
        {
            defaultVersion,
            diff = diffPair is null ? null : new { baseline = diffPair.Value.Baseline, comparison = diffPair.Value.Comparison },
            minimap = new { width = options.MinimapWidth, height = options.MinimapHeight, ext = effectiveExt },
            thresholds = new
            {
                distance = options.DiffDistanceThreshold,
                moveEpsilonRatio = options.MoveEpsilonRatio
            },
            coordMode = "wowtools",
            debugOverlayCorners = false,
            wmoSwapXY = false,
            swapPixelXY = false
        };

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        };

        var configPath = Path.Combine(viewerRoot, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(config, jsonOptions));
    }

    private static void CopyViewerAssets(string viewerRoot)
    {
        // Find ViewerAssets directory relative to this assembly
        var assemblyDir = Path.GetDirectoryName(typeof(ViewerReportWriter).Assembly.Location);
        if (assemblyDir is null) return;

        // Search upward from assembly location for ViewerAssets
        var currentDir = assemblyDir;
        string? assetsPath = null;

        for (int i = 0; i < 6; i++) // Search up to 6 levels
        {
            var candidate = Path.Combine(currentDir, "ViewerAssets");
            if (Directory.Exists(candidate))
            {
                assetsPath = candidate;
                break;
            }

            var parent = Directory.GetParent(currentDir);
            if (parent is null) break;
            currentDir = parent.FullName;
        }

        if (assetsPath is null || !Directory.Exists(assetsPath)) return;

        try
        {
            // Copy HTML files
            CopyIfExists(Path.Combine(assetsPath, "index.html"), Path.Combine(viewerRoot, "index.html"));
            CopyIfExists(Path.Combine(assetsPath, "tile.html"), Path.Combine(viewerRoot, "tile.html"));
            CopyIfExists(Path.Combine(assetsPath, "test.html"), Path.Combine(viewerRoot, "test.html"));
            CopyIfExists(Path.Combine(assetsPath, "styles.css"), Path.Combine(viewerRoot, "styles.css"));
            CopyIfExists(Path.Combine(assetsPath, "README.md"), Path.Combine(viewerRoot, "README.md"));
            CopyIfExists(Path.Combine(assetsPath, "start-viewer.ps1"), Path.Combine(viewerRoot, "start-viewer.ps1"));

            // Copy JS directory
            var jsSource = Path.Combine(assetsPath, "js");
            var jsTarget = Path.Combine(viewerRoot, "js");

            if (Directory.Exists(jsSource))
            {
                Directory.CreateDirectory(jsTarget);
                foreach (var file in Directory.GetFiles(jsSource, "*.js"))
                {
                    var fileName = Path.GetFileName(file);
                    File.Copy(file, Path.Combine(jsTarget, fileName), overwrite: true);
                }
            }
        }
        catch
        {
            // Fail silently - viewer will just be missing interactive assets
        }
    }

    private static void CopyIfExists(string source, string destination)
    {
        if (File.Exists(source))
        {
            File.Copy(source, destination, overwrite: true);
        }
    }

    private static string Sanitize(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "unnamed";
        var invalid = Path.GetInvalidFileNameChars();
        var chars = value.Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        var sanitized = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(sanitized) ? "unnamed" : sanitized;
    }

    private sealed record TileDescriptor(int Row, int Col, IReadOnlyList<string> Versions);
}
