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

        var maps = mapNames
            .OrderBy(n => n, comparer)
            .Select(name => new
            {
                Name = name,
                EntryTiles = entries
                    .Where(e => string.Equals(e.Map, name, StringComparison.OrdinalIgnoreCase))
                    .GroupBy(e => (e.TileRow, e.TileCol))
                    .Select(g => g.Key)
                    .ToHashSet(),
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

        foreach (var mapGroup in maps)
        {
            var mapName = mapGroup.Name;
            var safeMap = Sanitize(mapName);
            var mapDiffDir = Path.Combine(diffsRoot, safeMap);
            Directory.CreateDirectory(mapDiffDir);

            // Compute union of tiles from placements and minimap sources
            var tileSet = new HashSet<(int Row, int Col)>();
            foreach (var t in mapGroup.EntryTiles) tileSet.Add(t);
            foreach (var t in mapGroup.MinimapTiles) tileSet.Add(t);
            var tiles = tileSet.OrderBy(t => t.Row).ThenBy(t => t.Col).ToList();
            
            // Generate terrain overlays once per map per version (outside tile loop)
            var terrainOverlaysGenerated = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

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

                    var minimapFile = $"{mapName}_{col}_{row}.jpg";
                    var minimapPath = Path.Combine(versionMapMinimapDir, minimapFile);

                    try
                    {
                        // First check if JPG already exists in minimaps/ directory (from MPQ extraction)
                        var sourceJpgPath = Path.Combine(Path.GetDirectoryName(viewerRoot)!, "minimaps", minimapFile);
                        
                        if (File.Exists(sourceJpgPath))
                        {
                            // Copy existing JPG file
                            File.Copy(sourceJpgPath, minimapPath, overwrite: true);
                        }
                        else if (minimapLocator.TryGetTile(version, mapName, row, col, out var tileDescriptor))
                        {
                            // Fallback: decode from BLP in archive
                            using var tileStream = tileDescriptor.Open();
                            _minimapComposer.ComposeAsync(tileStream, minimapPath, resolvedOptions).GetAwaiter().GetResult();
                        }
                        else
                        {
                            // No source tile found → emit placeholder so tile pages still work
                            _minimapComposer.WritePlaceholderAsync(minimapPath, resolvedOptions).GetAwaiter().GetResult();
                        }
                    }
                    catch
                    {
                        // Any decode or IO failure → write placeholder and continue
                        _minimapComposer.WritePlaceholderAsync(minimapPath, resolvedOptions).GetAwaiter().GetResult();
                    }
                }

                foreach (var version in versionsForTile)
                {
                    var safeVersion = Sanitize(version);
                    var versionRoot = Path.Combine(overlaysRoot, safeVersion);
                    var mapOverlayDir = Path.Combine(versionRoot, safeMap);
                    Directory.CreateDirectory(mapOverlayDir);

                    var combinedDir = Path.Combine(mapOverlayDir, "combined");
                    var modelsDir = Path.Combine(mapOverlayDir, "m2");
                    var wmoDir = Path.Combine(mapOverlayDir, "wmo");
                    Directory.CreateDirectory(combinedDir);
                    Directory.CreateDirectory(modelsDir);
                    Directory.CreateDirectory(wmoDir);

                    // Get ALL entries for this version/map
                    // OverlayBuilder will filter by computed actual tile from coordinates
                    entriesByVersion.TryGetValue(version, out var versionEntries);
                    versionEntries ??= new List<AssetTimelineDetailedEntry>();

                    var perVersion = _overlayBuilder.BuildOverlayJson(
                        mapName,
                        row,
                        col,
                        versionEntries,
                        new[] { version },
                        resolvedOptions);
                    var combinedPath = Path.Combine(combinedDir, $"tile_r{row}_c{col}.json");
                    File.WriteAllText(combinedPath, perVersion);

                    var m2Json = _overlayBuilder.BuildOverlayJsonByKind(
                        mapName,
                        row,
                        col,
                        versionEntries,
                        new[] { version },
                        resolvedOptions,
                        PlacementKind.M2);
                    var m2Path = Path.Combine(modelsDir, $"tile_r{row}_c{col}.json");
                    File.WriteAllText(m2Path, m2Json);

                    var wmoJson = _overlayBuilder.BuildOverlayJsonByKind(
                        mapName,
                        row,
                        col,
                        versionEntries,
                        new[] { version },
                        resolvedOptions,
                        PlacementKind.WMO);
                    var wmoPath = Path.Combine(wmoDir, $"tile_r{row}_c{col}.json");
                    File.WriteAllText(wmoPath, wmoJson);
                    
                    // Generate terrain overlays once per map per version
                    var terrainKey = $"{version}_{mapName}";
                    if (!terrainOverlaysGenerated.Contains(terrainKey))
                    {
                        GenerateTerrainOverlays(result.RootDirectory, version, mapName, versionRoot, safeMap);
                        GenerateClusterOverlays(result.RootDirectory, version, mapName, versionRoot, safeMap, resolvedOptions);
                        terrainOverlaysGenerated.Add(terrainKey);
                    }
                }

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
        WriteConfigJson(viewerRoot, resolvedOptions, chosenDefaultVersion, effectiveDiffPair);
        
        // Phase 2: Generate overlay manifests for each version/map
        GenerateOverlayManifests(overlaysRoot, result, mapTileCatalog);
        
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
        (string Baseline, string Comparison)? diffPair)
    {
        var config = new
        {
            defaultVersion,
            diff = diffPair is null ? null : new { baseline = diffPair.Value.Baseline, comparison = diffPair.Value.Comparison },
            minimap = new { width = options.MinimapWidth, height = options.MinimapHeight },
            thresholds = new
            {
                distance = options.DiffDistanceThreshold,
                moveEpsilonRatio = options.MoveEpsilonRatio
            },
            coordMode = "wowtools",
            debugOverlayCorners = false
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
            CopyIfExists(Path.Combine(assetsPath, "viewer3d.html"), Path.Combine(viewerRoot, "viewer3d.html")); // 3D viewer
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
                
                // Copy overlays subdirectory
                var overlaysSource = Path.Combine(jsSource, "overlays");
                var overlaysTarget = Path.Combine(jsTarget, "overlays");
                if (Directory.Exists(overlaysSource))
                {
                    Directory.CreateDirectory(overlaysTarget);
                    foreach (var file in Directory.GetFiles(overlaysSource, "*.js"))
                    {
                        var fileName = Path.GetFileName(file);
                        File.Copy(file, Path.Combine(overlaysTarget, fileName), overwrite: true);
                    }
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

    private static void GenerateClusterOverlays(
        string rootDirectory,
        string version,
        string mapName,
        string versionRoot,
        string safeMap,
        ViewerOptions options)
    {
        try
        {
            // Look for cluster JSON in root directory
            var clusterJsonPath = Path.Combine(rootDirectory, $"{mapName}_spatial_clusters.json");
            
            if (!File.Exists(clusterJsonPath))
            {
                Console.WriteLine($"[ViewerReportWriter] Cluster JSON not found: {clusterJsonPath}");
                return;
            }

            var overlayVersionRoot = Path.Combine(versionRoot, "overlays", version);
            ClusterOverlayBuilder.BuildClusterOverlays(clusterJsonPath, mapName, overlayVersionRoot, options);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ViewerReportWriter] Error generating cluster overlays: {ex.Message}");
        }
    }

    private static void GenerateTerrainOverlays(
        string rootDirectory,
        string version,
        string mapName,
        string versionRoot,
        string safeMap)
    {
        // Look for MCNK terrain CSV in the output directory structure
        // Path: rollback_outputs/{version}/csv/{map}/{map}_mcnk_terrain.csv
        var csvMapDir = Path.Combine(rootDirectory, version, "csv", mapName);
        var terrainCsvPath = Path.Combine(csvMapDir, $"{mapName}_mcnk_terrain.csv");
        
        if (!File.Exists(terrainCsvPath))
        {
            // CSV not found - terrain overlays were not extracted, skip silently
            return;
        }
        
        var overlayVersionRoot = Path.Combine(versionRoot, "overlays", version);
        
        // Generate shadow map overlays if available
        McnkShadowOverlayBuilder.BuildOverlaysForMap(mapName, csvMapDir, overlayVersionRoot, version);

        try
        {
            // Load AreaTable lookup if available
            // AreaTable CSVs should be in the version directory: rollback_outputs/{version}/AreaTable_*.csv
            AreaTableLookup? areaLookup = null;
            try
            {
                var versionDir = Path.Combine(rootDirectory, version);
                areaLookup = AreaTableReader.LoadForVersion(versionDir);
            }
            catch
            {
                // AreaTable not available, terrain overlay will work without area names
            }

            // Generate terrain overlays
            // csvMapDir parameter is the directory containing the terrain CSV
            McnkTerrainOverlayBuilder.BuildOverlaysForMap(
                mapName,
                csvMapDir,
                overlayVersionRoot,
                version,
                areaLookup
            );
            
            Console.WriteLine($"[info] Generated terrain overlays for {mapName} ({version})");
        }
        catch (Exception ex)
        {
            // Log but don't fail - terrain overlays are optional
            Console.WriteLine($"[warn] Failed to generate terrain overlays for {mapName} ({version}): {ex.Message}");
        }
    }

    /// <summary>
    /// Phase 2: Generates overlay_manifest.json for each version/map combination.
    /// </summary>
    private static void GenerateOverlayManifests(
        string overlaysRoot,
        VersionComparisonResult result,
        Dictionary<string, List<TileDescriptor>> mapTileCatalog)
    {
        var manifestBuilder = new OverlayManifestBuilder();

        foreach (var (mapName, tiles) in mapTileCatalog)
        {
            var safeMap = Sanitize(mapName);
            var tileCoordsForMap = tiles.Select(t => (t.Row, t.Col)).Distinct().ToList();

            foreach (var version in result.Versions)
            {
                var safeVersion = Sanitize(version);
                var versionRoot = Path.Combine(overlaysRoot, safeVersion);
                var mapOverlayDir = Path.Combine(versionRoot, safeMap);

                // Check if terrain/shadow/cluster data exists
                var terrainDir = Path.Combine(mapOverlayDir, "terrain_complete");
                var shadowDir = Path.Combine(mapOverlayDir, "shadow_map");
                var clusterDir = Path.Combine(mapOverlayDir, "clusters");
                var hasTerrainData = Directory.Exists(terrainDir) && Directory.GetFiles(terrainDir, "*.json").Length > 0;
                var hasShadowData = Directory.Exists(shadowDir) && Directory.GetFiles(shadowDir, "*.json").Length > 0;
                var hasClusterData = Directory.Exists(clusterDir) && Directory.GetFiles(clusterDir, "*.json").Length > 0;

                // Generate manifest
                var manifestJson = manifestBuilder.BuildManifest(
                    version,
                    mapName,
                    tileCoordsForMap,
                    mapOverlayDir,
                    hasTerrainData,
                    hasShadowData,
                    hasClusterData);

                var manifestPath = Path.Combine(mapOverlayDir, "overlay_manifest.json");
                File.WriteAllText(manifestPath, manifestJson);

                Console.WriteLine($"[Phase 2] Generated overlay_manifest.json for {mapName} ({version})");
            }
        }
    }

    private sealed record TileDescriptor(int Row, int Col, IReadOnlyList<string> Versions);
}
