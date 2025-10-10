using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using WoWFormatLib.FileReaders;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<MapHostConfig>(builder.Configuration.GetSection("MapHost"));

var app = builder.Build();

// Load config (viewer.config.json at repo root if present)
var config = new MapHostConfig();
var configPath = Path.Combine(builder.Environment.ContentRootPath, "viewer.config.json");
if (File.Exists(configPath))
{
    var loaded = JsonSerializer.Deserialize<MapHostConfig>(await File.ReadAllTextAsync(configPath));
    if (loaded != null) config = loaded;
}

if (string.IsNullOrWhiteSpace(config.MinimapRoot))
{
    Console.WriteLine("[error] MapHost.MinimapRoot not configured. Create viewer.config.json with MinimapRoot.");
}
else
{
    config.MinimapRoot = Path.GetFullPath(config.MinimapRoot);
}

// Build TRS index on startup
var trsIndex = TryBuildTrsIndex(config.MinimapRoot!, config.TrsPath);
Console.WriteLine(trsIndex != null
    ? $"[ok] TRS index ready: {trsIndex.Count} entries"
    : "[warn] TRS not found or empty. /api/minimap will return 404.");

// Simple in-memory cache of PNG bytes
var tileCache = new ConcurrentDictionary<(string map,int x,int y), byte[]>();

app.MapGet("/api/health", () => Results.Ok(new { ok = true }));

app.MapGet("/api/index", () =>
{
    var mapNames = trsIndex == null
        ? Array.Empty<string>()
        : trsIndex.Keys.Select(k => k.map).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(s => s).ToArray();

    // DefaultMap fallback: use configured if present; else first available
    string? defaultMap = config.DefaultMap;
    if (string.IsNullOrWhiteSpace(defaultMap) || !mapNames.Contains(defaultMap, StringComparer.OrdinalIgnoreCase))
    {
        defaultMap = mapNames.FirstOrDefault();
        if (!string.IsNullOrWhiteSpace(config.DefaultMap) && defaultMap != null && !string.Equals(config.DefaultMap, defaultMap, StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"[warn] DefaultMap '{config.DefaultMap}' not found in TRS. Using '{defaultMap}'.");
        }
    }

    // Put default first in the list for UI convenience
    var ordered = (defaultMap == null)
        ? mapNames
        : new[] { defaultMap }.Concat(mapNames.Where(m => !string.Equals(m, defaultMap, StringComparison.OrdinalIgnoreCase))).ToArray();

    var maps = ordered.Select(n => new { name = n, width = 64, height = 64 }).ToArray();
    return Results.Json(new { version = config.VersionLabel ?? "dev", maps, defaultMap });
});

app.MapGet("/api/minimap/{map}/{x:int}/{y:int}.png", (string map, int x, int y) =>
{
    if (trsIndex == null) return Results.NotFound();
    var key = (map.ToLowerInvariant(), x, y);
    if (!trsIndex.TryGetValue(key, out var blpPath) || !File.Exists(blpPath)) return Results.NotFound();

    if (!tileCache.TryGetValue(key, out var bytes))
    {
        try
        {
            var reader = new BLPReader();
            reader.LoadBLP(blpPath);
            using var ms = new MemoryStream();
            reader.bmp.Save(ms, new PngEncoder());
            bytes = ms.ToArray();
            tileCache[key] = bytes;
        }
        catch
        {
            return Results.StatusCode(500);
        }
    }
    return Results.File(bytes, "image/png");
});

// Debug endpoint to inspect TRS mapping
app.MapGet("/api/debug/trs/{map}/{x:int}/{y:int}", (string map, int x, int y) =>
{
    if (trsIndex == null) return Results.Json(new { found = false, path = (string?)null });
    var key = (map.ToLowerInvariant(), x, y);
    return Results.Json(trsIndex.TryGetValue(key, out var p)
        ? new { found = true, path = p }
        : new { found = false, path = (string?)null });
});
app.UseDefaultFiles();
app.UseStaticFiles();

await app.RunAsync();

// --- helpers ---
static Dictionary<(string map,int x,int y), string>? TryBuildTrsIndex(string minimapRoot, string? trsPath)
{
    if (string.IsNullOrWhiteSpace(minimapRoot) || !Directory.Exists(minimapRoot)) return null;
    string? trs = trsPath;
    if (string.IsNullOrWhiteSpace(trs))
    {
        var p1 = Path.Combine(minimapRoot, "md5translate.trs");
        var p2 = Path.Combine(minimapRoot, "md5translate.txt");
        trs = File.Exists(p1) ? p1 : (File.Exists(p2) ? p2 : null);
    }
    if (string.IsNullOrWhiteSpace(trs) || !File.Exists(trs)) return null;

    var dict = new Dictionary<(string,int,int), string>();
    var baseDir = Path.GetDirectoryName(trs)!;
    string? currentMap = null;
    foreach (var raw in File.ReadAllLines(trs))
    {
        var line = raw.Trim();
        if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
        if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
        {
            currentMap = line.Substring(4).Trim();
            continue;
        }
        if (currentMap == null) continue;
        var parts = line.Split('\t');
        if (parts.Length != 2) continue;
        string a = parts[0].Trim();
        string b = parts[1].Trim();
        string mapSide = (a.Contains("map") && a.Contains(".blp", StringComparison.OrdinalIgnoreCase)) ? a : b;
        string actual = mapSide == a ? b : a;

        var stem = Path.GetFileNameWithoutExtension(mapSide);
        if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
        var xy = stem.Substring(3).Split('_');
        if (xy.Length != 2 || !int.TryParse(xy[0], out var tx) || !int.TryParse(xy[1], out var ty)) continue;
        var fullPath = Path.Combine(baseDir, actual.Replace('/', Path.DirectorySeparatorChar));
        dict[(currentMap.ToLowerInvariant(), tx, ty)] = fullPath;
    }
    return dict;
}

public sealed class MapHostConfig
{
    public string? MinimapRoot { get; set; }
    public string? TrsPath { get; set; }
    public string? DefaultMap { get; set; }
    public string? VersionLabel { get; set; }
}
