using System.Diagnostics;
using System.Text;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Converters;
using WoWViewer.DataSources;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Manages loading, caching, fallback resolution, and GPU texture lifecycle for WMO materials.
/// </summary>
internal sealed class WmoMaterialManager : IDisposable
{
    private const int MaxMaterialFallbackLogs = 20;
    public const int DeferredMaterialTextureLoadsPerFrame = 1;
    public const double DeferredMaterialTextureLoadBudgetMs = 2.0;

    private readonly GL _gl;
    private readonly WmoV14ToV17Converter.WmoV14Data _wmo;
    private readonly string _modelDir;
    private readonly IDataSource? _dataSource;
    private readonly bool _deferInitialMaterialTextureLoads;

    private readonly Dictionary<int, uint> _materialTextures = new();
    private readonly Queue<int> _pendingMaterialTextureLoads = new();
    private readonly HashSet<string> _materialFallbackLogKeys = new(StringComparer.Ordinal);
    private int _materialFallbackLogCount;

    public IReadOnlyDictionary<int, uint> MaterialTextures => _materialTextures;
    public int PendingTextureLoadsCount => _pendingMaterialTextureLoads.Count;

    public WmoMaterialManager(
        GL gl,
        WmoV14ToV17Converter.WmoV14Data wmo,
        string modelDir,
        IDataSource? dataSource,
        bool deferInitialMaterialTextureLoads)
    {
        _gl = gl;
        _wmo = wmo;
        _modelDir = modelDir;
        _dataSource = dataSource;
        _deferInitialMaterialTextureLoads = deferInitialMaterialTextureLoads;

        InitializeTextures();
    }

    public void InitializeTextures()
    {
        if (_deferInitialMaterialTextureLoads)
            QueueDeferredMaterialTextureLoads();
        else
            LoadMaterialTextures();
    }

    public bool TryGetTexture(int materialId, out uint glTex) =>
        _materialTextures.TryGetValue(materialId, out glTex);

    public void LoadMaterialTextures()
    {
        if (_dataSource == null) return;

        int loaded = 0, failed = 0;
        for (int i = 0; i < _wmo.Materials.Count; i++)
            TryLoadMaterialTexture(i, ref loaded, ref failed);

        ViewerLog.Trace($"[WmoRenderer] Textures: {loaded} loaded, {failed} failed out of {_wmo.Materials.Count} materials");
    }

    public void QueueDeferredMaterialTextureLoads()
    {
        _pendingMaterialTextureLoads.Clear();
        for (int i = 0; i < _wmo.Materials.Count; i++)
            _pendingMaterialTextureLoads.Enqueue(i);

        if (_pendingMaterialTextureLoads.Count > 0)
        {
            ViewerLog.Info(ViewerLog.Category.Wmo,
                $"[WMO-LOAD] Deferred {_pendingMaterialTextureLoads.Count} material textures for {_modelDir}");
        }
    }

    public int ProcessDeferredMaterialTextureLoads(
        int maxLoads = DeferredMaterialTextureLoadsPerFrame,
        double maxBudgetMs = DeferredMaterialTextureLoadBudgetMs)
    {
        if (!_deferInitialMaterialTextureLoads || _pendingMaterialTextureLoads.Count == 0 || _dataSource == null)
            return 0;

        if (maxLoads <= 0 || maxBudgetMs <= 0)
            return 0;

        var stopwatch = Stopwatch.StartNew();
        int loadsCompleted = 0;
        int loaded = 0, failed = 0;

        while (loadsCompleted < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _pendingMaterialTextureLoads.TryDequeue(out int materialIndex))
        {
            TryLoadMaterialTexture(materialIndex, ref loaded, ref failed);
            loadsCompleted++;
        }

        return loadsCompleted;
    }

    private void TryLoadMaterialTexture(int i, ref int loaded, ref int failed)
    {
        var mat = _wmo.Materials[i];
        string? texName = ResolveMaterialTextureName(mat);
        if (string.IsNullOrEmpty(texName))
            return;

        if (!texName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            texName += ".blp";

        byte[]? blpData = _dataSource?.ReadFile(texName);

        if (blpData == null)
            blpData = _dataSource?.ReadFile(texName.Replace('/', '\\'));

        if (blpData == null && _dataSource is MpqDataSource mpqDs)
        {
            var found = mpqDs.FindInFileSet(texName);
            if (found != null)
                blpData = _dataSource.ReadFile(found);
        }

        if (blpData != null && blpData.Length > 0)
        {
            uint glTex = LoadWmoTexture(blpData, texName);
            if (glTex != 0)
            {
                _materialTextures[i] = glTex;
                loaded++;
            }
            else
            {
                ViewerLog.Trace($"[WmoRenderer] Mat {i}: BLP decode failed for '{texName}'");
                failed++;
            }
        }
        else
        {
            ViewerLog.Trace($"[WmoRenderer] Mat {i}: texture not found '{texName}'");
            failed++;
        }
    }

    public int ResolveBatchMaterialId(WmoV14ToV17Converter.WmoGroupData group, WmoV14ToV17Converter.WmoBatch batch)
    {
        int originalMaterialId = batch.MaterialId;
        int materialId = originalMaterialId;
        if ((uint)materialId < (uint)_wmo.Materials.Count)
            return materialId;

        int firstFace = (int)(batch.FirstIndex / 3u);
        if ((uint)firstFace < (uint)group.FaceMaterials.Count)
        {
            int faceMaterial = group.FaceMaterials[firstFace];
            if ((uint)faceMaterial < (uint)_wmo.Materials.Count)
            {
                LogMaterialFallback(group, batch, originalMaterialId, faceMaterial, "MOPY");
                return faceMaterial;
            }
        }

        int defaultMaterial = _wmo.Materials.Count > 0 ? 0 : -1;
        LogMaterialFallback(group, batch, originalMaterialId, defaultMaterial, "DEFAULT");
        return defaultMaterial;
    }

    private void LogMaterialFallback(WmoV14ToV17Converter.WmoGroupData group, WmoV14ToV17Converter.WmoBatch batch,
        int originalMaterialId, int resolvedMaterialId, string source)
    {
        if (_materialFallbackLogCount >= MaxMaterialFallbackLogs)
            return;

        string groupName = string.IsNullOrWhiteSpace(group.Name) ? "<unnamed>" : group.Name;
        string key = $"{groupName}|{batch.FirstIndex}|{batch.IndexCount}|{originalMaterialId}|{resolvedMaterialId}|{source}";
        if (!_materialFallbackLogKeys.Add(key))
            return;

        _materialFallbackLogCount++;
        ViewerLog.Info(ViewerLog.Category.Wmo,
            $"[WMO-MAT] Fallback source={source} group='{groupName}' firstIndex={batch.FirstIndex} indexCount={batch.IndexCount} material {originalMaterialId} -> {resolvedMaterialId}");
    }

    private string? ResolveMaterialTextureName(WmoV14ToV17Converter.WmoMaterial material)
    {
        string? textureName = material.BaseTextureName;
        if (!string.IsNullOrWhiteSpace(textureName))
            return textureName;

        if (_wmo.MotxRaw.Length == 0)
            return null;

        textureName = ResolveStringFromRaw(_wmo.MotxRaw, material.Texture1Offset);
        if (string.IsNullOrWhiteSpace(textureName) && material.Texture1Offset >= 8)
            textureName = ResolveStringFromRaw(_wmo.MotxRaw, material.Texture1Offset - 8);

        return textureName;
    }

    private static string? ResolveStringFromRaw(byte[] raw, uint offset)
    {
        if (raw.Length == 0 || offset >= raw.Length)
            return null;

        int start = (int)offset;
        int end = Array.IndexOf(raw, (byte)0, start);
        if (end < 0)
            end = raw.Length;
        if (end <= start)
            return null;

        return Encoding.ASCII.GetString(raw, start, end - start).Trim();
    }

    private unsafe uint LoadWmoTexture(byte[] blpData, string name)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            using var image = blp.GetImage(0);

            // ImageSharp Image<Rgba32> is already tightly-packed RGBA
            int w = image.Width, h = image.Height;
            var pixels = new byte[w * h * 4];
            image.CopyPixelDataTo(pixels);

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true,
                TextureWrapMode.Repeat, TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            return tex;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[WmoRenderer] Failed to decode BLP {name}: {ex.Message}");
            return 0;
        }
    }

    public void Dispose()
    {
        foreach (var tex in _materialTextures.Values)
            _gl.DeleteTexture(tex);
        _materialTextures.Clear();
        _pendingMaterialTextureLoads.Clear();
    }
}
