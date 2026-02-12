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

    // LRU tracking — keys ordered by last access time (most recent at end)
    private readonly LinkedList<string> _mdxLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _mdxLruMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _wmoLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _wmoLruMap = new(StringComparer.OrdinalIgnoreCase);
    private const int MaxMdxCached = 500;  // Max MDX renderers in GPU memory
    private const int MaxWmoCached = 100;  // Max WMO renderers in GPU memory

    // Raw file data cache — avoids re-reading the same file from MPQ multiple times
    private readonly Dictionary<string, byte[]?> _fileDataCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _fileLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _fileLruMap = new(StringComparer.OrdinalIgnoreCase);
    private const int MaxFileCached = 1000; // Max raw file entries cached

    // Stats
    public int MdxModelsLoaded => _mdxModels.Count(kv => kv.Value != null);
    public int MdxModelsFailed => _mdxModels.Count(kv => kv.Value == null);
    public int WmoModelsLoaded => _wmoModels.Count(kv => kv.Value != null);
    public int WmoModelsFailed => _wmoModels.Count(kv => kv.Value == null);
    public int FileCacheCount => _fileDataCache.Count;

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
        if (_mdxModels.ContainsKey(normalizedKey))
        {
            TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
            return;
        }
        var renderer = LoadMdxModel(normalizedKey);
        _mdxModels[normalizedKey] = renderer;
        TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
        EvictMdxIfNeeded();
    }

    /// <summary>
    /// Ensure a WMO model is loaded (lazy load on first reference).
    /// </summary>
    public void EnsureWmoLoaded(string normalizedKey)
    {
        if (_wmoModels.ContainsKey(normalizedKey))
        {
            TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
            return;
        }
        var renderer = LoadWmoModel(normalizedKey);
        _wmoModels[normalizedKey] = renderer;
        TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
        EvictWmoIfNeeded();
    }

    /// <summary>
    /// Get a loaded MDX renderer by normalized key. Returns null if not loaded or failed.
    /// </summary>
    public MdxRenderer? GetMdx(string normalizedKey)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r))
        {
            TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
            return r;
        }
        return null;
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
        if (_wmoModels.TryGetValue(normalizedKey, out var r))
        {
            TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
            return r;
        }
        return null;
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

        // MDL/MDX interchangeable — game used both extensions for the same format
        if (data == null)
        {
            string altPath = SwapMdlMdxExtension(key);
            if (altPath != null)
                data = _dataSource?.ReadFile(altPath);
        }

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
        TouchLru(_fileLru, _fileLruMap, key);
        EvictFileCacheIfNeeded();
        return data;
    }

    public static string NormalizeKey(string path) => path.Replace('/', '\\').ToLowerInvariant();

    private static string? SwapMdlMdxExtension(string path)
    {
        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            return path[..^4] + ".mdx";
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return path[..^4] + ".mdl";
        // 3.3.5: MMDX stores .m2 paths but MPQ archives contain .mdx files
        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
            return path[..^3] + ".mdx";
        return null;
    }

    // ── Private loading ────────────────────────────────────────────────

    private int _mdxLoadFailCount = 0;
    private MdxRenderer? LoadMdxModel(string normalizedKey)
    {
        try
        {
            byte[]? data = ReadFileData(normalizedKey);
            if (data == null || data.Length == 0)
            {
                if (_mdxLoadFailCount++ < 5)
                    ViewerLog.Important(ViewerLog.Category.Mdx, $"MDX data null for: \"{normalizedKey}\"");
                return null;
            }

            // Detect M2 format (magic 0x3032444D = "MD20") and convert to MDX
            if (data.Length >= 4 && BitConverter.ToUInt32(data, 0) == 0x3032444D)
            {
                data = ConvertM2ToMdx(data, normalizedKey);
                if (data == null) return null;
            }

            using var ms = new MemoryStream(data);
            var mdx = MdxFile.Load(ms);
            string modelDir = Path.GetDirectoryName(normalizedKey) ?? "";
            return new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, normalizedKey);
        }
        catch (Exception ex)
        {
            if (_mdxLoadFailCount++ < 5)
                ViewerLog.Important(ViewerLog.Category.Mdx, $"MDX failed: {Path.GetFileName(normalizedKey)} - {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Convert M2 model bytes to MDX format. Attempts to load companion .skin file.
    /// </summary>
    private byte[]? ConvertM2ToMdx(byte[] m2Bytes, string normalizedKey)
    {
        try
        {
            // Try to find companion .skin file (ModelName00.skin)
            byte[]? skinBytes = null;
            string baseName = Path.GetFileNameWithoutExtension(normalizedKey);
            string dir = Path.GetDirectoryName(normalizedKey) ?? "";
            string[] skinCandidates = {
                Path.ChangeExtension(normalizedKey, "00.skin"),
                string.IsNullOrEmpty(dir) ? $"{baseName}00.skin" : $"{dir}\\{baseName}00.skin",
            };
            foreach (var skinPath in skinCandidates)
            {
                skinBytes = ReadFileData(skinPath);
                if (skinBytes != null)
                {
                    ViewerLog.Trace($"[M2] Loaded skin for {Path.GetFileName(normalizedKey)} ({skinBytes.Length} bytes)");
                    break;
                }
            }

            var converter = new M2ToMdxConverter();
            byte[] mdxBytes = converter.ConvertToBytes(m2Bytes, skinBytes);
            ViewerLog.Trace($"[M2] Converted {Path.GetFileName(normalizedKey)}: {m2Bytes.Length} → {mdxBytes.Length} bytes");
            return mdxBytes;
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Mdx, $"M2→MDX convert failed: {Path.GetFileName(normalizedKey)} - {ex.Message}");
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

            // Detect WMO version from bytes
            int version = DetectWmoVersion(data);

            WmoV14ToV17Converter.WmoV14Data wmo;

            if (version >= 17)
            {
                // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
                var dir = Path.GetDirectoryName(normalizedKey)?.Replace('/', '\\') ?? "";
                var baseName = Path.GetFileNameWithoutExtension(normalizedKey);

                var groupBytesList = new List<byte[]>();
                for (int gi = 0; gi < 512; gi++)
                {
                    var groupName = $"{baseName}_{gi:D3}.wmo";
                    var groupPath = string.IsNullOrEmpty(dir) ? groupName : $"{dir}\\{groupName}";
                    var groupBytes = ReadFileData(groupPath);
                    if (groupBytes == null || groupBytes.Length == 0) break;
                    groupBytesList.Add(groupBytes);
                }

                var v17Parser = new WmoV17ToV14Converter();
                wmo = v17Parser.ParseV17ToModel(data, groupBytesList);
                ViewerLog.Trace($"[WMO] Parsed v{version} direct: {Path.GetFileName(normalizedKey)} ({wmo.Groups.Count} groups)");
            }
            else
            {
                // v14/v16: parse with existing pipeline
                string tmpPath = Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp");
                try
                {
                    File.WriteAllBytes(tmpPath, data);
                    var converter = new WmoV14ToV17Converter();
                    wmo = converter.ParseWmoV14(tmpPath);

                    // v14/v16 split format: load group files from data source
                    if (wmo.Groups.Count == 0 && wmo.GroupCount > 0 && _dataSource != null)
                    {
                        var wmoDir = Path.GetDirectoryName(normalizedKey)?.Replace('/', '\\') ?? "";
                        var wmoBase = Path.GetFileNameWithoutExtension(normalizedKey);
                        for (int gi = 0; gi < wmo.GroupCount; gi++)
                        {
                            var groupName = $"{wmoBase}_{gi:D3}.wmo";
                            var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                            var groupBytes = ReadFileData(groupPath);
                            if (groupBytes != null && groupBytes.Length > 0)
                                converter.ParseGroupFile(groupBytes, wmo, gi);
                        }
                        for (int gi = 0; gi < wmo.Groups.Count; gi++)
                        {
                            if (wmo.Groups[gi].Name == null)
                                wmo.Groups[gi].Name = $"group_{gi}";
                        }
                    }
                }
                finally
                {
                    try { File.Delete(tmpPath); } catch { }
                }
            }

            string modelDir = Path.GetDirectoryName(normalizedKey) ?? "";
            return new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Wmo, $"WMO failed: {Path.GetFileName(normalizedKey)} - {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Detect WMO version from raw bytes. Returns 14 for Alpha (MOMO container), version number for v17+, or 0.
    /// </summary>
    private static int DetectWmoVersion(byte[] data)
    {
        if (data.Length < 12) return 0;
        string magic = System.Text.Encoding.ASCII.GetString(data, 0, 4);
        string reversed = new string(magic.Reverse().ToArray());

        // v14 Alpha: starts with MOMO container
        if (magic == "MOMO" || reversed == "MOMO") return 14;

        // v17+: starts with MVER chunk
        if (magic == "MVER" || reversed == "MVER")
        {
            uint size = BitConverter.ToUInt32(data, 4);
            if (size >= 4 && data.Length >= 12)
                return (int)BitConverter.ToUInt32(data, 8);
        }
        return 0;
    }

    // ── LRU helpers ─────────────────────────────────────────────────────

    private static void TouchLru(LinkedList<string> lru, Dictionary<string, LinkedListNode<string>> map, string key)
    {
        if (map.TryGetValue(key, out var node))
        {
            lru.Remove(node);
            lru.AddLast(node);
        }
        else
        {
            var newNode = lru.AddLast(key);
            map[key] = newNode;
        }
    }

    private void EvictMdxIfNeeded()
    {
        while (_mdxModels.Count > MaxMdxCached && _mdxLru.Count > 0)
        {
            string oldest = _mdxLru.First!.Value;
            _mdxLru.RemoveFirst();
            _mdxLruMap.Remove(oldest);
            if (_mdxModels.TryGetValue(oldest, out var r))
            {
                r?.Dispose();
                _mdxModels.Remove(oldest);
            }
        }
    }

    private void EvictWmoIfNeeded()
    {
        while (_wmoModels.Count > MaxWmoCached && _wmoLru.Count > 0)
        {
            string oldest = _wmoLru.First!.Value;
            _wmoLru.RemoveFirst();
            _wmoLruMap.Remove(oldest);
            if (_wmoModels.TryGetValue(oldest, out var r))
            {
                r?.Dispose();
                _wmoModels.Remove(oldest);
            }
        }
    }

    private void EvictFileCacheIfNeeded()
    {
        while (_fileDataCache.Count > MaxFileCached && _fileLru.Count > 0)
        {
            string oldest = _fileLru.First!.Value;
            _fileLru.RemoveFirst();
            _fileLruMap.Remove(oldest);
            _fileDataCache.Remove(oldest);
        }
    }

    public void Dispose()
    {
        foreach (var r in _mdxModels.Values)
            r?.Dispose();
        _mdxModels.Clear();
        _mdxLru.Clear();
        _mdxLruMap.Clear();

        foreach (var r in _wmoModels.Values)
            r?.Dispose();
        _wmoModels.Clear();
        _wmoLru.Clear();
        _wmoLruMap.Clear();

        _fileDataCache.Clear();
        _fileLru.Clear();
        _fileLruMap.Clear();
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
