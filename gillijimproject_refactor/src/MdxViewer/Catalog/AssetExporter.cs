using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Export;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Catalog;

/// <summary>
/// Exports asset catalog entries to JSON metadata, GLB models, and screenshots.
/// Supports both individual and batch export.
/// </summary>
public class AssetExporter
{
    private readonly GL _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;

    private ScreenshotRenderer? _screenshotRenderer;

    // Fuzzy path resolution: filename (lowercase, no ext) → list of full paths
    private Dictionary<string, List<string>>? _mdxPathIndex;
    private Dictionary<string, List<string>>? _wmoPathIndex;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public AssetExporter(GL gl, IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
    }

    /// <summary>
    /// Export a single entry to its own folder: metadata.json + model.glb + multi-angle screenshots.
    /// Returns an ExportEntryResult with paths and counts.
    /// 
    /// Output structure:
    ///   {outputDir}/{creatures|gameobjects}/{entryId}_{name}/
    ///     metadata.json
    ///     model.glb
    ///     front.png, back.png, left.png, right.png, top.png, three_quarter.png
    /// </summary>
    public ExportEntryResult ExportEntry(AssetCatalogEntry entry, string outputDir)
    {
        var result = new ExportEntryResult();

        string safeName = SanitizeFilename($"{entry.EntryId}_{entry.Name}");
        string typeDir = Path.Combine(outputDir, entry.Type == AssetType.Creature ? "creatures" : "gameobjects");
        string objectDir = Path.Combine(typeDir, safeName);
        Directory.CreateDirectory(objectDir);
        result.ObjectDir = objectDir;

        // Resolve model path once (exact → fuzzy fallback)
        string? resolvedModelPath = null;
        if (!string.IsNullOrEmpty(entry.ModelPath) && _dataSource != null)
        {
            string ext = entry.IsWmo ? ".wmo" : ".mdx";
            resolvedModelPath = entry.ModelPath;
            if (!_dataSource.FileExists(resolvedModelPath))
            {
                resolvedModelPath = FuzzyResolvePath(entry.ModelPath, ext);
                if (resolvedModelPath != null)
                    ViewerLog.Trace($"[AssetExporter] Fuzzy resolved: {entry.ModelPath} → {resolvedModelPath}");
            }
        }

        // JSON metadata
        try
        {
            var metadata = BuildMetadata(entry);
            if (resolvedModelPath != null && resolvedModelPath != entry.ModelPath)
                metadata["resolvedModelPath"] = resolvedModelPath;
            string jsonFile = Path.Combine(objectDir, "metadata.json");
            File.WriteAllText(jsonFile, JsonSerializer.Serialize(metadata, JsonOpts));
            result.JsonPath = jsonFile;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[AssetExporter] JSON export failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
        }

        // GLB model
        if (resolvedModelPath != null && _dataSource != null)
        {
            try
            {
                string glbFile = Path.Combine(objectDir, "model.glb");
                if (entry.IsWmo)
                    result.GlbPath = ExportWmoGlb(entry, glbFile, resolvedModelPath);
                else
                    result.GlbPath = ExportMdxGlb(entry, glbFile, resolvedModelPath);
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[AssetExporter] GLB export failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
            }
        }

        // Multi-angle screenshots
        if (resolvedModelPath != null && !entry.IsWmo && _dataSource != null)
        {
            try
            {
                _screenshotRenderer ??= new ScreenshotRenderer(_gl, _dataSource, _texResolver);
                result.ScreenshotCount = _screenshotRenderer.CaptureMultiAngle(entry, objectDir, resolvedModelPath: resolvedModelPath);
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[AssetExporter] Screenshots failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
            }
        }

        return result;
    }

    /// <summary>
    /// Batch export all entries with progress callback.
    /// </summary>
    public async Task<BatchExportResult> ExportBatchAsync(
        IReadOnlyList<AssetCatalogEntry> entries,
        string outputDir,
        Action<int, int, string>? onProgress = null,
        CancellationToken ct = default)
    {
        var result = new BatchExportResult();
        Directory.CreateDirectory(outputDir);

        for (int i = 0; i < entries.Count; i++)
        {
            if (ct.IsCancellationRequested) break;

            var entry = entries[i];
            onProgress?.Invoke(i + 1, entries.Count, entry.Name);

            var er = ExportEntry(entry, outputDir);
            if (er.JsonPath != null) result.JsonCount++;
            if (er.GlbPath != null) result.GlbCount++;
            result.ScreenshotCount += er.ScreenshotCount;
            if (er.JsonPath == null && er.GlbPath == null) result.FailedCount++;
            result.TotalProcessed++;

            // Yield to avoid blocking the UI thread
            if (i % 10 == 0)
                await Task.Yield();
        }

        return result;
    }

    private string? ExportMdxGlb(AssetCatalogEntry entry, string outputPath, string resolvedPath)
    {
        if (_dataSource == null) return null;

        byte[]? mdxData = _dataSource.ReadFile(resolvedPath);
        if (mdxData == null)
        {
            ViewerLog.Trace($"[AssetExporter] MDX not found: {resolvedPath}");
            return null;
        }

        using var ms = new MemoryStream(mdxData);
        using var br = new BinaryReader(ms);
        var mdx = MdxFile.Load(br);
        string modelDir = Path.GetDirectoryName(resolvedPath)?.Replace('/', '\\') ?? "";
        GlbExporter.ExportMdx(mdx, modelDir, outputPath, _dataSource);
        return outputPath;
    }

    private string? ExportWmoGlb(AssetCatalogEntry entry, string outputPath, string resolvedPath)
    {
        if (_dataSource == null) return null;

        byte[]? wmoData = _dataSource.ReadFile(resolvedPath);
        if (wmoData == null)
        {
            ViewerLog.Trace($"[AssetExporter] WMO not found: {resolvedPath}");
            return null;
        }

        // Write to temp file, parse WMO v14, then export to GLB
        try
        {
            string modelDir = Path.GetDirectoryName(entry.ModelPath)?.Replace('/', '\\') ?? "";
            string tempFile = Path.Combine(Path.GetTempPath(), $"wmo_export_{entry.EntryId}.wmo");
            File.WriteAllBytes(tempFile, wmoData);
            var converter = new WmoV14ToV17Converter();
            var wmo = converter.ParseWmoV14(tempFile);
            GlbExporter.ExportWmo(wmo, modelDir, outputPath, _dataSource);
            try { File.Delete(tempFile); } catch { }
            ViewerLog.Trace($"[AssetExporter] Exported WMO GLB: {outputPath}");
            return outputPath;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[AssetExporter] WMO GLB export failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Build a lazy index of filename (lowercase, no extension) → full paths for fuzzy matching.
    /// </summary>
    private void EnsurePathIndex(string extension)
    {
        if (_dataSource == null) return;

        bool isMdx = extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase);
        ref var index = ref (isMdx ? ref _mdxPathIndex : ref _wmoPathIndex);
        if (index != null) return;

        index = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        var files = _dataSource.GetFileList(extension);
        foreach (var f in files)
        {
            string nameNoExt = Path.GetFileNameWithoutExtension(f).ToLowerInvariant();
            if (!index.TryGetValue(nameNoExt, out var list))
            {
                list = new List<string>(1);
                index[nameNoExt] = list;
            }
            list.Add(f);
        }
        ViewerLog.Trace($"[AssetExporter] Built {extension} path index: {index.Count} unique names from {files.Count} files");
    }

    /// <summary>
    /// Fuzzy-resolve a model path that wasn't found by exact match.
    /// Strategies:
    ///   1. Extract filename, look up in index by name (case-insensitive)
    ///   2. Try common path patterns (Creature\Name\Name.mdx, etc.)
    ///   3. Try with/without extension
    /// </summary>
    private string? FuzzyResolvePath(string originalPath, string extension)
    {
        if (_dataSource == null) return null;
        EnsurePathIndex(extension);

        bool isMdx = extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase);
        var index = isMdx ? _mdxPathIndex : _wmoPathIndex;
        if (index == null) return null;

        // Strategy 1: Extract the filename without extension and look up
        string baseName = Path.GetFileNameWithoutExtension(originalPath).ToLowerInvariant();
        if (index.TryGetValue(baseName, out var candidates) && candidates.Count > 0)
        {
            // Prefer path that contains the original directory hint
            string? dirHint = Path.GetDirectoryName(originalPath)?.Replace('/', '\\');
            if (!string.IsNullOrEmpty(dirHint))
            {
                var match = candidates.FirstOrDefault(c =>
                    c.Contains(dirHint, StringComparison.OrdinalIgnoreCase));
                if (match != null) return match;
            }
            return candidates[0]; // best guess
        }

        // Strategy 2: Try the bare name (no path, no extension) — common for creature models
        // e.g. "Basilisk" → look for any file named "Basilisk.mdx"
        string bareName = originalPath.Replace('\\', '/').Split('/').Last();
        bareName = Path.GetFileNameWithoutExtension(bareName).ToLowerInvariant();
        if (bareName != baseName && index.TryGetValue(bareName, out candidates) && candidates.Count > 0)
            return candidates[0];

        // Strategy 3: Try case-insensitive direct read
        // Some data sources are case-sensitive; try common casing patterns
        string[] casings = {
            originalPath,
            originalPath.ToLowerInvariant(),
            originalPath.Replace('/', '\\'),
            originalPath.Replace('\\', '/'),
        };
        foreach (var c in casings)
        {
            if (_dataSource.FileExists(c))
                return c;
        }

        return null;
    }

    private static Dictionary<string, object?> BuildMetadata(AssetCatalogEntry entry)
    {
        var meta = new Dictionary<string, object?>
        {
            ["entryId"] = entry.EntryId,
            ["type"] = entry.Type.ToString().ToLowerInvariant(),
            ["typeLabel"] = entry.TypeLabel,
            ["name"] = entry.Name,
            ["displayId"] = entry.DisplayId,
            ["modelPath"] = entry.ModelPath,
            ["scale"] = entry.EffectiveScale,
            ["isWmo"] = entry.IsWmo
        };

        // File references (relative to this folder)
        if (!string.IsNullOrEmpty(entry.ModelPath))
        {
            meta["glbFile"] = "model.glb";
            if (!entry.IsWmo)
            {
                meta["screenshots"] = ScreenshotRenderer.CameraAngles
                    .Select(a => new Dictionary<string, object>
                    {
                        ["angle"] = a.name,
                        ["file"] = $"{a.name}.png",
                        ["azimuth"] = a.azimuth,
                        ["elevation"] = a.elevation
                    }).ToList();
            }
        }

        if (entry.Type == AssetType.Creature)
        {
            if (!string.IsNullOrEmpty(entry.Subname))
                meta["subname"] = entry.Subname;
            meta["levelMin"] = entry.LevelMin;
            meta["levelMax"] = entry.LevelMax;
            meta["rank"] = entry.Rank;
            meta["creatureType"] = entry.CreatureType;
            meta["faction"] = entry.Faction;
            meta["npcFlags"] = entry.NpcFlags;
            if (entry.AllDisplayIds.Length > 1)
                meta["allDisplayIds"] = entry.AllDisplayIds;
            if (entry.TextureVariations.Length > 0)
                meta["textureVariations"] = entry.TextureVariations;
        }
        else
        {
            meta["gameObjectType"] = entry.GameObjectType;
            meta["flags"] = entry.Flags;
        }

        if (entry.Spawns.Count > 0)
        {
            meta["spawnCount"] = entry.Spawns.Count;
            meta["spawns"] = entry.Spawns.Select(s => new Dictionary<string, object>
            {
                ["spawnId"] = s.SpawnId,
                ["mapId"] = s.MapId,
                ["x"] = Math.Round(s.X, 2),
                ["y"] = Math.Round(s.Y, 2),
                ["z"] = Math.Round(s.Z, 2),
                ["orientation"] = Math.Round(s.Orientation, 4)
            }).ToList();
        }

        return meta;
    }

    private static string SanitizeFilename(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (c == ' ') sb.Append('_');
            else if (Array.IndexOf(invalid, c) < 0 && c != '\0') sb.Append(c);
        }
        string result = sb.ToString();
        if (result.Length > 80) result = result[..80];
        return result;
    }
}

/// <summary>
/// Result of exporting a single catalog entry.
/// </summary>
public class ExportEntryResult
{
    public string? ObjectDir { get; set; }
    public string? JsonPath { get; set; }
    public string? GlbPath { get; set; }
    public int ScreenshotCount { get; set; }

    public bool HasAnyOutput => JsonPath != null || GlbPath != null || ScreenshotCount > 0;
}

public class BatchExportResult
{
    public int TotalProcessed { get; set; }
    public int JsonCount { get; set; }
    public int GlbCount { get; set; }
    public int ScreenshotCount { get; set; }
    public int FailedCount { get; set; }

    public override string ToString() =>
        $"Processed {TotalProcessed}: {JsonCount} JSON, {GlbCount} GLB, {ScreenshotCount} screenshots, {FailedCount} failed";
}
