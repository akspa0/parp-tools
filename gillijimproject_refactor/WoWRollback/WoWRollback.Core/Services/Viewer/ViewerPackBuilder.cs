using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds a static viewer pack: tiles WebP + overlays scaffolds + index.json.
/// </summary>
public sealed class ViewerPackBuilder
{
    public sealed record BuildResult(int TilesWritten, int MapsWritten, string DefaultMap);

    /// <summary>
    /// Build a pack from a session root that contains versioned Alpha folders and a minimap root.
    /// </summary>
    public BuildResult Build(
        string sessionRoot,
        string minimapRoot,
        string outRoot,
        IReadOnlyCollection<string>? versionsFilter,
        ISet<string>? mapsFilter,
        string label,
        Func<string, IMinimapProvider?>? providerFactory = null)
    {
        if (string.IsNullOrWhiteSpace(sessionRoot) || !Directory.Exists(sessionRoot))
            throw new DirectoryNotFoundException(sessionRoot);
        if (string.IsNullOrWhiteSpace(minimapRoot) || !Directory.Exists(minimapRoot))
            throw new DirectoryNotFoundException(minimapRoot);
        Directory.CreateDirectory(outRoot);
        var dataRoot = Path.Combine(outRoot, "data");
        Directory.CreateDirectory(dataRoot);

        // Simplified: treat sessionRoot (test_data) as the authoritative root and use
        // provided versions directly. Do NOT require minimap indicators at the version folder.
        var rootBase = sessionRoot;
        Console.WriteLine($"[Pack] rootBase (sessionRoot): {rootBase}");

        List<string> versions;
        if (versionsFilter is not null && versionsFilter.Count > 0)
        {
            versions = versionsFilter.Select(v => v.Trim()).Where(v => v.Length > 0).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        }
        else
        {
            // Fallback: use immediate subfolder names as versions
            try
            {
                versions = Directory.EnumerateDirectories(rootBase, "*", SearchOption.TopDirectoryOnly)
                    .Select(d => Path.GetFileName(Path.TrimEndingDirectorySeparator(d)))
                    .Where(n => !string.IsNullOrWhiteSpace(n))
                    .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                    .ToList();
            }
            catch
            {
                versions = new List<string>();
            }
        }

        Console.WriteLine($"[Pack] Versions to scan: {(versions.Count == 0 ? "(none)" : string.Join(", ", versions))}");
        if (versions.Count == 0)
            throw new InvalidOperationException($"No versions resolved under '{rootBase}'. Provide --versions <list> matching folder names under test_data.");

        var locator = providerFactory is null
            ? MinimapLocator.Build(rootBase, versions)
            : MinimapLocator.Build(rootBase, versions, providerFactory);

        // Collect maps
        var allMaps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var ver in versions)
        {
            foreach (var m in locator.EnumerateMaps(ver)) allMaps.Add(m);
        }
        var maps = allMaps.OrderBy(s => s).ToList();
        if (mapsFilter is not null && mapsFilter.Count > 0)
            maps = maps.Where(m => mapsFilter.Contains(m)).ToList();
        var defaultMap = maps.FirstOrDefault() ?? string.Empty;

        // Compose tiles
        int written = 0;
        var composer = new MinimapComposer();
        foreach (var ver in versions)
        foreach (var map in maps)
        {
            var mapDir = Path.Combine(dataRoot, "tiles", map);
            Directory.CreateDirectory(mapDir);

            // overlays scaffold per map (create before tile loop so placeholders can be written)
            var overlaysDir = Path.Combine(dataRoot, "overlays", map);
            var coordsDir = Path.Combine(overlaysDir, "coords");
            var m2Dir = Path.Combine(overlaysDir, "m2");
            var wmoDir = Path.Combine(overlaysDir, "wmo");
            var areaDir = Path.Combine(overlaysDir, "area");
            Directory.CreateDirectory(overlaysDir);
            Directory.CreateDirectory(coordsDir);
            Directory.CreateDirectory(m2Dir);
            Directory.CreateDirectory(wmoDir);
            Directory.CreateDirectory(areaDir);

            foreach (var (row, col) in locator.EnumerateTiles(ver, map))
            {
                if (!locator.TryGetTile(ver, map, row, col, out var tile)) continue;
                var x = col; var y = row; // x=col, y=row
                var dst = Path.Combine(mapDir, $"{x}_{y}.webp");
                if (!File.Exists(dst))
                {
                    try
                    {
                        using var fsSrc = tile.Open();
                        System.Threading.Tasks.Task.Run(async () => await composer.ComposeAsync(fsSrc, dst, ViewerOptions.CreateDefault())).Wait();
                        written++;
                    }
                    catch
                    {
                        // skip failures
                    }
                }

                // Write placeholder overlay tiles so UI never 404s when analysis is missing
                WriteIfMissing(Path.Combine(coordsDir, $"{x}_{y}.json"), new { tile = new { x, y }, worldRect = (object?)null });
                WriteIfMissing(Path.Combine(m2Dir, $"{x}_{y}.json"), new { features = Array.Empty<object>() });
                WriteIfMissing(Path.Combine(wmoDir, $"{x}_{y}.json"), new { features = Array.Empty<object>() });
                WriteIfMissing(Path.Combine(areaDir, $"{x}_{y}.json"), new { features = Array.Empty<object>() });
            }

            // Write manifest per map
            var manifestPath = Path.Combine(overlaysDir, "manifest.json");
            var manifestObj = new
            {
                layers = new object[]
                {
                    new { id = "coords", title = "Game Coords", tilePattern = "coords/{x}_{y}.json", coord = "world", enabledByDefault = true },
                    new { id = "m2", title = "M2", tilePattern = "m2/{x}_{y}.json", coord = "world" },
                    new { id = "wmo", title = "WMO", tilePattern = "wmo/{x}_{y}.json", coord = "world" },
                    new { id = "area", title = "Area IDs", tilePattern = "area/{x}_{y}.json", coord = "tile" }
                }
            };
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifestObj));
        }

        // index.json under data/
        var indexPath = Path.Combine(dataRoot, "index.json");
        var indexObj = new { version = label, defaultMap, maps = maps.Select(n => new { name = n, size = 64 }).ToArray() };
        File.WriteAllText(indexPath, JsonSerializer.Serialize(indexObj));

        return new BuildResult(written, maps.Count, defaultMap);
    }

    /// <summary>
    /// Discover converted ADT tiles produced earlier in the pipeline and emit minimal per-tile
    /// overlay JSONs (coords/m2/wmo) so the viewer layer system can load. This is a scaffold; a
    /// subsequent change will parse MDDF/MODF and world rects via Warcraft.NET.
    /// </summary>
    public void HarvestFromConvertedAdts(
        IEnumerable<(string Version, string Map, string MapDir)> inputs,
        string? communityListfile,
        string? lkListfile,
        string overlaysRoot)
    {
        if (inputs is null) return;
        Directory.CreateDirectory(overlaysRoot);

        foreach (var (version, map, mapDir) in inputs)
        {
            if (string.IsNullOrWhiteSpace(map) || string.IsNullOrWhiteSpace(mapDir) || !Directory.Exists(mapDir))
            {
                // Try fallback to version/analysis/<map>/index.json even if World/Maps/<map> is empty
                // We'll still create per-tile outputs based on analysis index if available.
            }

            var mapOverlayDir = Path.Combine(overlaysRoot, map);
            var coordsDir = Path.Combine(mapOverlayDir, "coords");
            var m2Dir = Path.Combine(mapOverlayDir, "m2");
            var wmoDir = Path.Combine(mapOverlayDir, "wmo");
            Directory.CreateDirectory(coordsDir);
            Directory.CreateDirectory(m2Dir);
            Directory.CreateDirectory(wmoDir);

            // Preferred: parse version/analysis/<map>/index.json (conversion-derived, LK-correct)
            var versionRoot = TryFindVersionRootFromMapDir(mapDir, version) ?? mapDir;
            var analysisIndex = Path.Combine(versionRoot, "analysis", map, "index.json");
            if (File.Exists(analysisIndex))
            {
                TryHarvestFromAnalysisIndex(analysisIndex, coordsDir, m2Dir, wmoDir);
                continue;
            }

            // Fallback: if ADT files exist, at least lay down placeholders per tile
            if (Directory.Exists(mapDir))
            {
                var adtFiles = Directory.EnumerateFiles(mapDir, map + "_*.adt", SearchOption.TopDirectoryOnly);
                foreach (var adt in adtFiles)
                {
                    var stem = Path.GetFileNameWithoutExtension(adt);
                    var parts = stem.Split('_');
                    if (parts.Length < 3) continue;
                    if (!int.TryParse(parts[^2], out var x)) continue;
                    if (!int.TryParse(parts[^1], out var y)) continue;

                    WriteIfMissing(Path.Combine(coordsDir, $"{x}_{y}.json"), new { tile = new { x, y }, worldRect = (object?)null });
                    WriteIfMissing(Path.Combine(m2Dir, $"{x}_{y}.json"), new { features = Array.Empty<object>() });
                    WriteIfMissing(Path.Combine(wmoDir, $"{x}_{y}.json"), new { features = Array.Empty<object>() });
                }
            }
        }
    }

    private static string? TryFindVersionRootFromMapDir(string mapDir, string version)
    {
        try
        {
            var d = new DirectoryInfo(mapDir);
            for (int i = 0; i < 5 && d?.Parent != null; i++)
            {
                if (string.Equals(d.Name, version, StringComparison.OrdinalIgnoreCase))
                    return d.FullName;
                d = d.Parent;
            }
        }
        catch { }
        return null;
    }

    private static void TryHarvestFromAnalysisIndex(string indexPath, string coordsDir, string m2Dir, string wmoDir)
    {
        using var fs = File.OpenRead(indexPath);
        using var doc = JsonDocument.Parse(fs, new JsonDocumentOptions { AllowTrailingCommas = true });
        var root = doc.RootElement;

        // Find tiles array under common keys
        var tiles = TryGetProperty(root, "tiles") ?? TryGetProperty(root, "Tiles");
        if (tiles is null || tiles.Value.ValueKind != JsonValueKind.Array) return;

        foreach (var tile in tiles.Value.EnumerateArray())
        {
            int x = GetInt(tile, "x") ?? GetInt(tile, "tileX") ?? 0;
            int y = GetInt(tile, "y") ?? GetInt(tile, "tileY") ?? 0;
            if (x < 0 || y < 0) continue;

            // coords
            var wr = TryGetProperty(tile, "worldRect") ?? TryGetProperty(tile, "WorldRect");
            object? worldRectObj = null;
            if (wr is not null && wr.Value.ValueKind == JsonValueKind.Object)
            {
                double? minX = GetDouble(wr.Value, "minX") ?? GetDouble(wr.Value, "MinX");
                double? maxX = GetDouble(wr.Value, "maxX") ?? GetDouble(wr.Value, "MaxX");
                double? minY = GetDouble(wr.Value, "minY") ?? GetDouble(wr.Value, "MinY");
                double? maxY = GetDouble(wr.Value, "maxY") ?? GetDouble(wr.Value, "MaxY");
                if (minX.HasValue && maxX.HasValue && minY.HasValue && maxY.HasValue)
                {
                    worldRectObj = new { minX = minX.Value, maxX = maxX.Value, minY = minY.Value, maxY = maxY.Value };
                }
            }
            WriteOrReplace(Path.Combine(coordsDir, $"{x}_{y}.json"), new { tile = new { x, y }, worldRect = worldRectObj });

            // placements: M2 and WMO
            var m2Arr = TryGetProperty(tile, "m2") ?? TryGetProperty(tile, "M2") ?? TryGetProperty(tile, "mddf") ?? TryGetProperty(tile, "MDDF");
            var wmoArr = TryGetProperty(tile, "wmo") ?? TryGetProperty(tile, "WMO") ?? TryGetProperty(tile, "modf") ?? TryGetProperty(tile, "MODF");

            if (m2Arr is not null && m2Arr.Value.ValueKind == JsonValueKind.Array)
            {
                var features = new List<object>();
                foreach (var f in m2Arr.Value.EnumerateArray())
                {
                    var feature = BuildFeature(f, type: "m2");
                    if (feature is not null) features.Add(feature);
                }
                WriteOrReplace(Path.Combine(m2Dir, $"{x}_{y}.json"), new { features });
            }
            if (wmoArr is not null && wmoArr.Value.ValueKind == JsonValueKind.Array)
            {
                var features = new List<object>();
                foreach (var f in wmoArr.Value.EnumerateArray())
                {
                    var feature = BuildFeature(f, type: "wmo");
                    if (feature is not null) features.Add(feature);
                }
                WriteOrReplace(Path.Combine(wmoDir, $"{x}_{y}.json"), new { features });
            }
        }
    }

    private static object? BuildFeature(JsonElement f, string type)
    {
        // Normalize common fields; keep raw blob
        var fileDataId = GetInt(f, "FileDataID") ?? GetInt(f, "fileDataId") ?? GetInt(f, "fileId");
        var uniqueId = GetUInt(f, "UniqueID") ?? GetUInt(f, "uniqueId") ?? GetUInt(f, "UniqueId");
        var flags = GetUInt(f, "Flags") ?? GetUInt(f, "flags");

        var posObj = TryGetProperty(f, "Position") ?? TryGetProperty(f, "position") ?? TryGetProperty(f, "pos");
        var rotObj = TryGetProperty(f, "Rotation") ?? TryGetProperty(f, "rotation") ?? TryGetProperty(f, "rot");
        var scale = GetDouble(f, "Scale") ?? GetDouble(f, "scale");
        var doodadSet = GetInt(f, "DoodadSet") ?? GetInt(f, "doodadSet");
        var nameSet = GetInt(f, "NameSet") ?? GetInt(f, "nameSet");

        object? position = null, rotation = null;
        if (posObj is not null && posObj.Value.ValueKind == JsonValueKind.Object)
        {
            position = new { x = GetDouble(posObj.Value, "x") ?? GetDouble(posObj.Value, "X") ?? 0,
                             y = GetDouble(posObj.Value, "y") ?? GetDouble(posObj.Value, "Y") ?? 0,
                             z = GetDouble(posObj.Value, "z") ?? GetDouble(posObj.Value, "Z") ?? 0 };
        }
        if (rotObj is not null && rotObj.Value.ValueKind == JsonValueKind.Object)
        {
            rotation = new { x = GetDouble(rotObj.Value, "x") ?? GetDouble(rotObj.Value, "X") ?? 0,
                             y = GetDouble(rotObj.Value, "y") ?? GetDouble(rotObj.Value, "Y") ?? 0,
                             z = GetDouble(rotObj.Value, "z") ?? GetDouble(rotObj.Value, "Z") ?? 0 };
        }

        // Serialize raw element back to JSON for embedding
        string rawJson = f.GetRawText();

        return new
        {
            Type = type,
            FileDataID = fileDataId,
            UniqueID = uniqueId,
            Flags = flags,
            Position = position,
            Rotation = rotation,
            Scale = scale,
            DoodadSet = doodadSet,
            NameSet = nameSet,
            raw = JsonSerializer.Deserialize<object>(rawJson)
        };
    }

    private static JsonElement? TryGetProperty(JsonElement obj, string name)
    {
        if (obj.ValueKind != JsonValueKind.Object) return null;
        if (obj.TryGetProperty(name, out var value)) return value;
        return null;
    }

    private static int? GetInt(JsonElement obj, string name)
    {
        if (obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(name, out var el))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var v)) return v;
        }
        return null;
    }
    private static uint? GetUInt(JsonElement obj, string name)
    {
        if (obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(name, out var el))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetUInt32(out var v)) return v;
        }
        return null;
    }
    private static double? GetDouble(JsonElement obj, string name)
    {
        if (obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(name, out var el))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetDouble(out var v)) return v;
        }
        return null;
    }

    private static void WriteIfMissing(string path, object obj)
    {
        if (!File.Exists(path)) File.WriteAllText(path, JsonSerializer.Serialize(obj));
    }
    private static void WriteOrReplace(string path, object obj)
    {
        File.WriteAllText(path, JsonSerializer.Serialize(obj));
    }

    private static void TryDiscoverVersionsUnder(string baseDir, List<string> versions)
    {
        versions.Clear();
        try
        {
            // Case 1: baseDir itself is a version folder (contains minimap indicators)
            if (ContainsMinimapIndicators(baseDir))
            {
                var name = Path.GetFileName(Path.TrimEndingDirectorySeparator(baseDir));
                if (!string.IsNullOrWhiteSpace(name)) versions.Add(name);
                return;
            }

            // Case 2: immediate subdirs represent versions
            foreach (var sub in Directory.EnumerateDirectories(baseDir, "*", SearchOption.TopDirectoryOnly))
            {
                if (ContainsMinimapIndicators(sub))
                {
                    var nm = Path.GetFileName(sub);
                    if (!string.IsNullOrWhiteSpace(nm)) versions.Add(nm);
                }
            }
        }
        catch
        {
            // ignore IO errors
        }
    }

    private static bool TryInferRootBaseFromMinimap(string minimapRoot, out string rootBase, out string? singleVersion)
    {
        rootBase = minimapRoot;
        singleVersion = null;
        try
        {
            var dir = new DirectoryInfo(minimapRoot);
            for (int i = 0; i < 5 && dir != null; i++)
            {
                if (ContainsMinimapIndicators(dir.FullName))
                {
                    var parent = dir.Parent;
                    if (parent != null)
                    {
                        rootBase = parent.FullName; // parent contains version directories
                        singleVersion = dir.Name;
                        return true;
                    }
                }
                dir = dir.Parent;
            }
        }
        catch { }
        return false;
    }

    private static bool ContainsMinimapIndicators(string dir)
    {
        try
        {
            if (Directory.Exists(Path.Combine(dir, "tree"))) return true;
            if (Directory.EnumerateFiles(dir, "md5translate.*", SearchOption.AllDirectories).Any()) return true;
            var candidates = new[]
            {
                Path.Combine(dir, "tree", "World", "Textures", "Minimap"),
                Path.Combine(dir, "tree", "world", "textures", "minimap"),
                Path.Combine(dir, "World", "Textures", "Minimap"),
                Path.Combine(dir, "world", "textures", "minimap"),
                Path.Combine(dir, "Textures", "Minimap"),
                Path.Combine(dir, "textures", "minimap")
            };
            return candidates.Any(Directory.Exists);
        }
        catch { return false; }
    }
}
