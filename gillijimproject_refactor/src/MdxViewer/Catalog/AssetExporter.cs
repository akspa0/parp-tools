using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Export;
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
    /// Export a single entry: JSON metadata + GLB model + screenshot.
    /// Returns (jsonPath, glbPath, screenshotPath) or nulls on failure.
    /// </summary>
    public (string? jsonPath, string? glbPath, string? screenshotPath) ExportEntry(AssetCatalogEntry entry, string outputDir)
    {
        Directory.CreateDirectory(outputDir);

        string safeName = SanitizeFilename($"{entry.EntryId}_{entry.Name}");
        string subDir = Path.Combine(outputDir, entry.Type == AssetType.Creature ? "creatures" : "gameobjects");
        Directory.CreateDirectory(subDir);

        // JSON metadata
        string? jsonPath = null;
        try
        {
            var metadata = BuildMetadata(entry);
            string jsonFile = Path.Combine(subDir, $"{safeName}.json");
            File.WriteAllText(jsonFile, JsonSerializer.Serialize(metadata, JsonOpts));
            jsonPath = jsonFile;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AssetExporter] JSON export failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
        }

        // GLB model
        string? glbPath = null;
        if (!string.IsNullOrEmpty(entry.ModelPath) && _dataSource != null)
        {
            try
            {
                string glbFile = Path.Combine(subDir, $"{safeName}.glb");
                if (entry.IsWmo)
                {
                    // WMO export — use existing WMO→GLB pipeline
                    glbPath = ExportWmoGlb(entry, glbFile);
                }
                else
                {
                    // MDX export
                    glbPath = ExportMdxGlb(entry, glbFile);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AssetExporter] GLB export failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
            }
        }

        // Screenshot
        string? screenshotPath = null;
        if (!string.IsNullOrEmpty(entry.ModelPath) && !entry.IsWmo && _dataSource != null)
        {
            try
            {
                _screenshotRenderer ??= new ScreenshotRenderer(_gl, _dataSource, _texResolver);
                string pngFile = Path.Combine(subDir, $"{safeName}.png");
                if (_screenshotRenderer.CaptureScreenshot(entry, pngFile))
                    screenshotPath = pngFile;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AssetExporter] Screenshot failed for {entry.Name} ({entry.EntryId}): {ex.Message}");
            }
        }

        return (jsonPath, glbPath, screenshotPath);
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

            var (jsonPath, glbPath, ssPath) = ExportEntry(entry, outputDir);
            if (jsonPath != null) result.JsonCount++;
            if (glbPath != null) result.GlbCount++;
            if (ssPath != null) result.ScreenshotCount++;
            if (jsonPath == null && glbPath == null) result.FailedCount++;
            result.TotalProcessed++;

            // Yield to avoid blocking the UI thread
            if (i % 10 == 0)
                await Task.Yield();
        }

        return result;
    }

    private string? ExportMdxGlb(AssetCatalogEntry entry, string outputPath)
    {
        if (_dataSource == null || string.IsNullOrEmpty(entry.ModelPath)) return null;

        // Try to load the MDX file from the data source
        byte[]? mdxData = _dataSource.ReadFile(entry.ModelPath);
        if (mdxData == null)
        {
            // Try with .mdx extension variations
            string altPath = entry.ModelPath.Replace(".mdx", ".MDX", StringComparison.OrdinalIgnoreCase);
            mdxData = _dataSource.ReadFile(altPath);
        }
        if (mdxData == null)
        {
            Console.WriteLine($"[AssetExporter] MDX not found: {entry.ModelPath}");
            return null;
        }

        using var ms = new MemoryStream(mdxData);
        using var br = new BinaryReader(ms);
        var mdx = MdxFile.Load(br);
        string modelDir = Path.GetDirectoryName(entry.ModelPath)?.Replace('/', '\\') ?? "";
        GlbExporter.ExportMdx(mdx, modelDir, outputPath, _dataSource);
        Console.WriteLine($"[AssetExporter] Exported GLB: {outputPath}");
        return outputPath;
    }

    private string? ExportWmoGlb(AssetCatalogEntry entry, string outputPath)
    {
        if (_dataSource == null || string.IsNullOrEmpty(entry.ModelPath)) return null;

        byte[]? wmoData = _dataSource.ReadFile(entry.ModelPath);
        if (wmoData == null)
        {
            Console.WriteLine($"[AssetExporter] WMO not found: {entry.ModelPath}");
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
            Console.WriteLine($"[AssetExporter] Exported WMO GLB: {outputPath}");
            return outputPath;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AssetExporter] WMO GLB export failed: {ex.Message}");
            return null;
        }
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
