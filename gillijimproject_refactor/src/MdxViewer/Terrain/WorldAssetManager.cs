using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Terrain;

/// <summary>
/// Centralized asset manager for world scene rendering.
/// Ensures each model and texture is loaded exactly once into GPU memory,
/// then instanced via transforms for all placements.
/// 
/// Ownership: WorldAssetManager owns all GPU resources (renderers, textures).
/// WorldScene owns the instance lists (transforms) and delegates rendering here.
/// </summary>
public class WorldAssetManager : IDisposable
{
    private readonly GL _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;

    // ── Shared caches ──────────────────────────────────────────────────

    // Model path (normalized) → loaded renderer (null = load attempted but failed)
    private readonly Dictionary<string, MdxRenderer?> _mdxModels = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, WmoRenderer?> _wmoModels = new(StringComparer.OrdinalIgnoreCase);

    // Raw file data cache — avoids re-reading the same file from MPQ multiple times
    private readonly Dictionary<string, byte[]?> _fileDataCache = new(StringComparer.OrdinalIgnoreCase);

    // Stats
    public int MdxModelsLoaded => _mdxModels.Count(kv => kv.Value != null);
    public int MdxModelsFailed => _mdxModels.Count(kv => kv.Value == null);
    public int WmoModelsLoaded => _wmoModels.Count(kv => kv.Value != null);
    public int WmoModelsFailed => _wmoModels.Count(kv => kv.Value == null);

    public WorldAssetManager(GL gl, IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
    }

    /// <summary>
    /// Pre-register all model names referenced by the map so we know the full asset set.
    /// Does NOT load anything yet — just prepares the manifest.
    /// </summary>
    public AssetManifest BuildManifest(IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames,
        IReadOnlyList<MddfPlacement> mddfPlacements, IReadOnlyList<ModfPlacement> modfPlacements)
    {
        var manifest = new AssetManifest();

        // Collect unique MDX models actually referenced by placements
        foreach (var p in mddfPlacements)
        {
            if (p.NameIndex >= 0 && p.NameIndex < mdxNames.Count)
                manifest.ReferencedMdx.Add(NormalizeKey(mdxNames[p.NameIndex]));
        }

        // Collect unique WMO models actually referenced by placements
        foreach (var p in modfPlacements)
        {
            if (p.NameIndex >= 0 && p.NameIndex < wmoNames.Count)
                manifest.ReferencedWmo.Add(NormalizeKey(wmoNames[p.NameIndex]));
        }

        ViewerLog.Important(ViewerLog.Category.General, $"Manifest: {manifest.ReferencedMdx.Count} unique MDX, {manifest.ReferencedWmo.Count} unique WMO");
        ViewerLog.Info(ViewerLog.Category.General, $"Name tables: {mdxNames.Count} MDX names, {wmoNames.Count} WMO names");
        ViewerLog.Info(ViewerLog.Category.General, $"Placements: {mddfPlacements.Count} MDDF, {modfPlacements.Count} MODF");
        if (modfPlacements.Count > 0)
        {
            var p = modfPlacements[0];
            string name = p.NameIndex >= 0 && p.NameIndex < wmoNames.Count ? wmoNames[p.NameIndex] : $"BAD_INDEX({p.NameIndex})";
            ViewerLog.Debug(ViewerLog.Category.Wmo, $"First MODF: nameIdx={p.NameIndex} name=\"{name}\" key=\"{NormalizeKey(name)}\"");
        }
        if (mddfPlacements.Count > 0)
        {
            var p = mddfPlacements[0];
            string name = p.NameIndex >= 0 && p.NameIndex < mdxNames.Count ? mdxNames[p.NameIndex] : $"BAD_INDEX({p.NameIndex})";
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"First MDDF: nameIdx={p.NameIndex} name=\"{name}\" key=\"{NormalizeKey(name)}\"");
        }
        return manifest;
    }

    /// <summary>
    /// Load all models in the manifest. Each model is loaded exactly once.
    /// </summary>
    public void LoadManifest(AssetManifest manifest)
    {
        int mdxOk = 0, mdxFail = 0;
        foreach (var key in manifest.ReferencedMdx)
        {
            if (_mdxModels.ContainsKey(key)) continue;
            var renderer = LoadMdxModel(key);
            _mdxModels[key] = renderer;
            if (renderer != null) mdxOk++; else mdxFail++;
        }

        int wmoOk = 0, wmoFail = 0;
        foreach (var key in manifest.ReferencedWmo)
        {
            if (_wmoModels.ContainsKey(key)) continue;
            var renderer = LoadWmoModel(key);
            _wmoModels[key] = renderer;
            if (renderer != null) wmoOk++; else wmoFail++;
        }

        ViewerLog.Important(ViewerLog.Category.General, $"Loaded: MDX {mdxOk} ok / {mdxFail} failed, WMO {wmoOk} ok / {wmoFail} failed");
    }

    /// <summary>
    /// Ensure an MDX model is loaded (lazy load on first reference).
    /// </summary>
    public void EnsureMdxLoaded(string normalizedKey)
    {
        if (_mdxModels.ContainsKey(normalizedKey)) return;
        var renderer = LoadMdxModel(normalizedKey);
        _mdxModels[normalizedKey] = renderer;
    }

    /// <summary>
    /// Ensure a WMO model is loaded (lazy load on first reference).
    /// </summary>
    public void EnsureWmoLoaded(string normalizedKey)
    {
        if (_wmoModels.ContainsKey(normalizedKey)) return;
        var renderer = LoadWmoModel(normalizedKey);
        _wmoModels[normalizedKey] = renderer;
    }

    /// <summary>
    /// Get a loaded MDX renderer by normalized key. Returns null if not loaded or failed.
    /// </summary>
    public MdxRenderer? GetMdx(string normalizedKey)
    {
        _mdxModels.TryGetValue(normalizedKey, out var r);
        return r;
    }

    /// <summary>
    /// Get the model-space bounding box for a loaded MDX model.
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetMdxBounds(string normalizedKey, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            boundsMin = r.BoundsMin;
            boundsMax = r.BoundsMax;
            return true;
        }
        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get the bounding box center for a loaded MDX model.
    /// MDX geometry is offset from origin — the BB center is the effective pivot.
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetMdxPivotOffset(string normalizedKey, out Vector3 pivotOffset)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            pivotOffset = (r.BoundsMin + r.BoundsMax) * 0.5f;
            return true;
        }
        pivotOffset = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get the MOHD bounding box for a loaded WMO model (local space).
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetWmoBounds(string normalizedKey, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (_wmoModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            boundsMin = r.BoundsMin;
            boundsMax = r.BoundsMax;
            return true;
        }
        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get a loaded WMO renderer by normalized key. Returns null if not loaded or failed.
    /// </summary>
    public WmoRenderer? GetWmo(string normalizedKey)
    {
        _wmoModels.TryGetValue(normalizedKey, out var r);
        return r;
    }

    /// <summary>
    /// Read file data from the data source, with caching to avoid duplicate MPQ reads.
    /// </summary>
    public byte[]? ReadFileData(string virtualPath)
    {
        string key = NormalizeKey(virtualPath);
        if (_fileDataCache.TryGetValue(key, out var cached))
            return cached;

        byte[]? data = _dataSource?.ReadFile(virtualPath);
        if (data == null)
            data = _dataSource?.ReadFile(key);

        // Alpha 0.5.3: WMO/WDT/WDL files are stored as .ext.mpq — try appending .mpq
        if (data == null)
        {
            data = _dataSource?.ReadFile(virtualPath + ".mpq");
            if (data == null)
                data = _dataSource?.ReadFile(key + ".mpq");
        }

        // Fallback: try stripping leading path components (e.g., "World\" prefix)
        // Some MDNM entries may have paths that don't match the MPQ internal structure
        if (data == null)
        {
            // Try just the filename
            string fileName = Path.GetFileName(key);
            if (!string.IsNullOrEmpty(fileName) && fileName != key)
            {
                data = _dataSource?.ReadFile(fileName);
            }

            // Try with common prefixes
            if (data == null)
            {
                string[] prefixes = { "Creature\\", "World\\", "Environment\\" };
                foreach (var prefix in prefixes)
                {
                    if (!key.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    {
                        data = _dataSource?.ReadFile(prefix + key);
                        if (data != null) break;
                        // Also try prefix + just filename
                        data = _dataSource?.ReadFile(prefix + fileName);
                        if (data != null) break;
                    }
                }
            }
        }

        _fileDataCache[key] = data;
        return data;
    }

    public static string NormalizeKey(string path) => path.Replace('/', '\\').ToLowerInvariant();

    // ── Private loading ────────────────────────────────────────────────

    private MdxRenderer? LoadMdxModel(string normalizedKey)
    {
        try
        {
            byte[]? data = ReadFileData(normalizedKey);
            if (data == null || data.Length == 0)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"MDX data null for: \"{normalizedKey}\"");
                return null;
            }

            using var ms = new MemoryStream(data);
            var mdx = MdxFile.Load(ms);
            string modelDir = Path.GetDirectoryName(normalizedKey) ?? "";
            return new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, normalizedKey);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Mdx, $"MDX failed: {Path.GetFileName(normalizedKey)} - {ex.Message}");
            return null;
        }
    }

    private WmoRenderer? LoadWmoModel(string normalizedKey)
    {
        try
        {
            byte[]? data = ReadFileData(normalizedKey);
            if (data == null || data.Length == 0)
            {
                if (_wmoModels.Count < 3)
                    ViewerLog.Debug(ViewerLog.Category.Wmo, $"WMO data null for: \"{normalizedKey}\"");
                return null;
            }
            if (_wmoModels.Count < 3)
                ViewerLog.Debug(ViewerLog.Category.Wmo, $"WMO data found for: \"{normalizedKey}\" ({data.Length} bytes)");

            // WMO v14 needs to be written to temp file for the converter
            string tmpPath = Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp");
            try
            {
                File.WriteAllBytes(tmpPath, data);
                var converter = new WmoV14ToV17Converter();
                var wmo = converter.ParseWmoV14(tmpPath);
                string modelDir = Path.GetDirectoryName(normalizedKey) ?? "";
                return new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver);
            }
            finally
            {
                try { File.Delete(tmpPath); } catch { }
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Wmo, $"WMO failed: {Path.GetFileName(normalizedKey)} - {ex.Message}");
            return null;
        }
    }

    public void Dispose()
    {
        foreach (var r in _mdxModels.Values)
            r?.Dispose();
        _mdxModels.Clear();

        foreach (var r in _wmoModels.Values)
            r?.Dispose();
        _wmoModels.Clear();

        _fileDataCache.Clear();
    }
}

/// <summary>
/// Describes the set of unique assets referenced by a map.
/// Built before loading so we know the full scope.
/// </summary>
public class AssetManifest
{
    public HashSet<string> ReferencedMdx { get; } = new(StringComparer.OrdinalIgnoreCase);
    public HashSet<string> ReferencedWmo { get; } = new(StringComparer.OrdinalIgnoreCase);
}
