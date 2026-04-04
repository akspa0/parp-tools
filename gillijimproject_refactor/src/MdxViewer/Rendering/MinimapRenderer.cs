using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;

namespace MdxViewer.Rendering;

/// <summary>
/// Handles loading and caching of minimap tile textures for display in the UI.
/// Uses Md5TranslateResolver to handle hashed file paths in early WoW versions.
/// </summary>
public class MinimapRenderer : IDisposable
{
    private const int BackgroundWorkerCount = 4;

    private readonly GL _gl;
    private readonly IDataSource _dataSource;
    private readonly Md5TranslateIndex? _md5Index;
    private readonly string _cacheRoot;
    private readonly ConcurrentDictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentQueue<MinimapTileRequest> _pendingRequests = new();
    private readonly ConcurrentQueue<DecodedMinimapTileUpload> _readyUploads = new();
    private readonly ConcurrentDictionary<string, byte> _queuedCacheKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, string?> _resolvedTilePathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _requestSignal = new(0);
    private readonly CancellationTokenSource _disposeCts = new();
    private readonly Task[] _loaderTasks;
    private int _completedRequestCount;
    private int _uploadedTileCount;
    private int _failedTileCount;
    private int _queuedRequestCount;
    private int _readyUploadCount;
    private int _inflightRequestCount;

    public MinimapRenderer(GL gl, IDataSource dataSource, Md5TranslateIndex? md5Index, string cacheRoot)
    {
        _gl = gl;
        _dataSource = dataSource;
        _md5Index = md5Index;
        _cacheRoot = cacheRoot;
        Directory.CreateDirectory(_cacheRoot);

        _loaderTasks = Enumerable.Range(0, BackgroundWorkerCount)
            .Select(_ => Task.Run(() => BackgroundLoadLoop(_disposeCts.Token), _disposeCts.Token))
            .ToArray();
    }

    public int PendingTileCount => Math.Max(0, Volatile.Read(ref _queuedRequestCount) + Volatile.Read(ref _readyUploadCount) + Volatile.Read(ref _inflightRequestCount));
    public int UploadedTileCount => _uploadedTileCount;
    public int FailedTileCount => _failedTileCount;
    public bool IsBusy => PendingTileCount > 0;
    public float LoadingProgress
    {
        get
        {
            int total = _completedRequestCount + PendingTileCount;
            return total > 0 ? _completedRequestCount / (float)total : 1f;
        }
    }

    /// <summary>
    /// Gets the GL texture handle for a specific minimap tile.
    /// Returns 0 if the tile is not found or failed to load.
    /// </summary>
    public uint GetTileTexture(string mapName, int tx, int ty)
    {
        string plainPath = MinimapService.GetMinimapTilePath(mapName, tx, ty);
        
        if (_textureCache.TryGetValue(plainPath, out uint cached))
            return cached;

        QueueTileLoad(mapName, tx, ty, plainPath);
        return 0;
    }

    public int ProcessPendingLoads(int maxLoads = 2, double maxBudgetMs = 5.0)
    {
        if (Volatile.Read(ref _readyUploadCount) == 0 || maxLoads <= 0)
            return 0;

        int processed = 0;
        var stopwatch = Stopwatch.StartNew();
        while (processed < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _readyUploads.TryDequeue(out DecodedMinimapTileUpload upload))
        {
            Interlocked.Decrement(ref _readyUploadCount);

            if (_textureCache.ContainsKey(upload.CacheKey))
                continue;

            uint tex = upload.Tile != null ? UploadTexture(upload.Tile) : 0;
            _textureCache[upload.CacheKey] = tex;
            _completedRequestCount++;

            if (tex != 0)
                _uploadedTileCount++;
            else
                _failedTileCount++;

            processed++;
        }

        return processed;
    }

    private void QueueTileLoad(string mapName, int tx, int ty, string cacheKey)
    {
        if (_textureCache.ContainsKey(cacheKey) || !_queuedCacheKeys.TryAdd(cacheKey, 0))
            return;

        _pendingRequests.Enqueue(new MinimapTileRequest(mapName, tx, ty, cacheKey));
        Interlocked.Increment(ref _queuedRequestCount);
        _requestSignal.Release();
    }

    private async Task BackgroundLoadLoop(CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                await _requestSignal.WaitAsync(cancellationToken).ConfigureAwait(false);

                while (_pendingRequests.TryDequeue(out MinimapTileRequest request))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Interlocked.Decrement(ref _queuedRequestCount);

                    if (_textureCache.ContainsKey(request.CacheKey))
                    {
                        _queuedCacheKeys.TryRemove(request.CacheKey, out _);
                        continue;
                    }

                    Interlocked.Increment(ref _inflightRequestCount);
                    try
                    {
                        DecodedMinimapTile? tile = LoadTileData(request.MapName, request.Tx, request.Ty, request.CacheKey);
                        _readyUploads.Enqueue(new DecodedMinimapTileUpload(request.CacheKey, tile));
                        Interlocked.Increment(ref _readyUploadCount);
                    }
                    finally
                    {
                        _queuedCacheKeys.TryRemove(request.CacheKey, out _);
                        Interlocked.Decrement(ref _inflightRequestCount);
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    private DecodedMinimapTile? LoadTileData(string mapName, int tx, int ty, string cacheKey)
    {
        if (TryLoadCachedBitmap(cacheKey, out DecodedMinimapTile? cachedTile) && cachedTile != null)
            return cachedTile;

        byte[]? data = TryReadTileData(cacheKey);
        if (data == null || data.Length == 0)
        {
            foreach (string candidatePath in EnumerateTileCandidates(mapName, tx, ty, cacheKey))
            {
                data = TryReadTileData(candidatePath);
                if (data != null && data.Length > 0)
                    break;
            }
        }

        if (data == null || data.Length == 0)
            return null;

        try
        {
            using var ms = new MemoryStream(data);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);
            DecodedMinimapTile decoded = ConvertBitmap(bmp);
            bmp.Dispose();
            return decoded;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load tile {cacheKey}: {ex.Message}");
            return null;
        }
    }

    private byte[]? TryReadTileData(string plainPath)
    {
        if (_resolvedTilePathCache.TryGetValue(plainPath, out string? resolvedPath))
        {
            if (resolvedPath == null)
                return null;

            byte[]? cachedData = ReadVirtualFile(resolvedPath);
            if (cachedData != null && cachedData.Length > 0)
                return cachedData;

            _resolvedTilePathCache.TryRemove(plainPath, out _);
        }

        byte[]? data = null;

        if (_md5Index != null)
        {
            var normalized = _md5Index.Normalize(plainPath);
            if (_md5Index.PlainToHash.TryGetValue(normalized, out string? hashedPath))
            {
                data = ReadVirtualFile(hashedPath);
                if (data != null && data.Length > 0)
                {
                    _resolvedTilePathCache[plainPath] = hashedPath;
                    return data;
                }
            }
        }

        data = ReadVirtualFile(plainPath);
        if (data != null && data.Length > 0)
        {
            _resolvedTilePathCache[plainPath] = plainPath;
            return data;
        }

        _resolvedTilePathCache[plainPath] = null;
        return null;
    }

    private byte[]? ReadVirtualFile(string virtualPath)
    {
        byte[]? data = _dataSource.ReadFile(virtualPath);
        if (data != null && data.Length > 0)
            return data;

        string altPath = virtualPath.Replace('/', '\\');
        if (!string.Equals(altPath, virtualPath, StringComparison.Ordinal))
        {
            data = _dataSource.ReadFile(altPath);
            if (data != null && data.Length > 0)
                return data;
        }

        return null;
    }

    private static IEnumerable<string> EnumerateTileCandidates(string mapName, int x, int y, string primaryCandidate)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var yieldReturnList = new List<string>();

        void AddCandidate(string candidate)
        {
            if (!string.IsNullOrWhiteSpace(candidate) && seen.Add(candidate))
                yieldReturnList.Add(candidate);
        }

        string normalizedMapName = mapName.ToLowerInvariant();
        string x2 = x.ToString("D2");
        string y2 = y.ToString("D2");
        string trsFormat = $"map{x}_{y2}.blp";

        AddCandidate(primaryCandidate);

        AddCandidate($"{normalizedMapName}\\{trsFormat}");
        AddCandidate($"{normalizedMapName}/{trsFormat}");
        AddCandidate($"textures/minimap/{normalizedMapName}/{trsFormat}");

        AddCandidate($"textures/minimap/{normalizedMapName}/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"{normalizedMapName}/map{x2}_{y2}.blp");

        string mapNameSpace = InsertSpaceBeforeCapitals(mapName).ToLowerInvariant();
        if (!string.Equals(mapNameSpace, normalizedMapName, StringComparison.OrdinalIgnoreCase))
        {
            AddCandidate($"{mapNameSpace}\\{trsFormat}");
            AddCandidate($"textures/minimap/{mapNameSpace}/{trsFormat}");
            AddCandidate($"textures/minimap/{mapNameSpace}/{mapNameSpace}_{x2}_{y2}.blp");
            AddCandidate($"textures/minimap/{mapNameSpace}/map{x2}_{y2}.blp");
            AddCandidate($"{mapNameSpace}/map{x2}_{y2}.blp");
        }

        AddCandidate($"world/minimaps/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"world/minimaps/{normalizedMapName}/map{x}_{y}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{x}_{y}.blp");

        return yieldReturnList;
    }

    private static string InsertSpaceBeforeCapitals(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return value;

        var builder = new System.Text.StringBuilder(value.Length + 8);
        for (int index = 0; index < value.Length; index++)
        {
            char ch = value[index];
            if (index > 0 && char.IsUpper(ch) && !char.IsWhiteSpace(value[index - 1]))
                builder.Append(' ');

            builder.Append(ch);
        }

        return builder.ToString();
    }

    private sealed record DecodedMinimapTile(int Width, int Height, byte[] Pixels);
    private readonly record struct DecodedMinimapTileUpload(string CacheKey, DecodedMinimapTile? Tile);
    private readonly record struct MinimapTileRequest(string MapName, int Tx, int Ty, string CacheKey);

    private static DecodedMinimapTile ConvertBitmap(System.Drawing.Bitmap bmp)
    {
        int width = bmp.Width;
        int height = bmp.Height;
        var pixels = new byte[width * height * 4];
        var rect = new System.Drawing.Rectangle(0, 0, width, height);
        var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
            System.Drawing.Imaging.PixelFormat.Format32bppArgb);

        try
        {
            var srcBytes = new byte[bmpData.Stride * height];
            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, srcBytes, 0, srcBytes.Length);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcIdx = y * bmpData.Stride + x * 4;
                    int dstIdx = (y * width + x) * 4;
                    pixels[dstIdx + 0] = srcBytes[srcIdx + 2];
                    pixels[dstIdx + 1] = srcBytes[srcIdx + 1];
                    pixels[dstIdx + 2] = srcBytes[srcIdx + 0];
                    pixels[dstIdx + 3] = srcBytes[srcIdx + 3];
                }
            }
        }
        finally
        {
            bmp.UnlockBits(bmpData);
        }

        return new DecodedMinimapTile(width, height, pixels);
    }

    private unsafe uint UploadTexture(DecodedMinimapTile tile)
    {
        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);
        fixed (byte* ptr = tile.Pixels)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                (uint)tile.Width, (uint)tile.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return tex;
    }

    private bool TryLoadCachedBitmap(string plainPath, out DecodedMinimapTile? tile)
    {
        tile = null;
        string cachePath = GetCachePath(plainPath);
        if (!File.Exists(cachePath))
            return false;

        try
        {
            using var bmp = new System.Drawing.Bitmap(cachePath);
            tile = ConvertBitmap(bmp);
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load cached tile {cachePath}: {ex.Message}");
            return false;
        }
    }

    private void TrySaveCachedBitmap(string plainPath, System.Drawing.Bitmap bmp)
    {
        string cachePath = GetCachePath(plainPath);
        string? cacheDirectory = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(cacheDirectory))
            Directory.CreateDirectory(cacheDirectory);

        string tempPath = cachePath + ".tmp";
        try
        {
            bmp.Save(tempPath, System.Drawing.Imaging.ImageFormat.Png);
            File.Move(tempPath, cachePath, overwrite: true);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to save cached tile {cachePath}: {ex.Message}");
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private string GetCachePath(string plainPath)
    {
        string normalized = plainPath.Replace('\\', '/').ToLowerInvariant();
        string hash = Convert.ToHexString(SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(normalized))).ToLowerInvariant();
        return Path.Combine(_cacheRoot, hash + ".png");
    }

    public void Dispose()
    {
        _disposeCts.Cancel();
        for (int i = 0; i < _loaderTasks.Length; i++)
            _requestSignal.Release();

        try
        {
            Task.WaitAll(_loaderTasks, TimeSpan.FromSeconds(1));
        }
        catch (AggregateException)
        {
        }

        foreach (var tex in _textureCache.Values)
        {
            if (tex != 0) _gl.DeleteTexture(tex);
        }
        _textureCache.Clear();

        _requestSignal.Dispose();
        _disposeCts.Dispose();
    }
}
