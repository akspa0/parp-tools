using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Blp;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;

namespace WoWViewer.Rendering;

/// <summary>
/// Handles loading and caching of minimap tile textures for display in the UI.
/// Uses Md5TranslateResolver to handle hashed file paths in early WoW versions.
/// </summary>
public class MinimapRenderer : IDisposable
{
    // Minimap reads share the active IDataSource with terrain/object streaming.
    // Keep client reads serialized here so the minimap cannot multiply archive
    // contention while the render thread is resolving world assets.
    private const int BackgroundWorkerCount = 1;

    // Epic 249 R-39c: while world assets are still loading the worker waits in short steps, but never
    // longer than this per tile, so continuous streaming cannot starve the minimap entirely.
    private const int WorldLoadDeferStepMs = 50;
    private const int WorldLoadMaxDeferMs = 1000;

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
    private readonly Thread[] _loaderThreads;
    private readonly bool _compressedUploadSupported;
    private readonly bool _preferModernTilePath;
    private volatile bool _deferBackgroundReads;
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

        // R-39b: DXT tiles go to the GPU compressed when the context has S3TC (checked here, on the
        // thread that owns the GL context). R-39c: modern CASC clients keep minimaps under world/minimaps.
        _compressedUploadSupported = gl.IsExtensionPresent("GL_EXT_texture_compression_s3tc");
        _preferModernTilePath = dataSource is CascDataSource;

        // R-39c: a dedicated lowest-priority thread, not a thread-pool task, so tile reads and decodes
        // yield the CPU to the render thread.
        _loaderThreads = Enumerable.Range(0, BackgroundWorkerCount)
            .Select(_ =>
            {
                var thread = new Thread(() => BackgroundLoadLoop(_disposeCts.Token))
                {
                    IsBackground = true,
                    Priority = ThreadPriority.Lowest,
                    Name = "MinimapTileLoader",
                };
                thread.Start();
                return thread;
            })
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
    public uint GetTileTexture(string mapName, int tx, int ty, string? overlayMapName = null)
    {
        if (!string.IsNullOrEmpty(overlayMapName))
        {
            uint overlayTex = GetTileTexture(overlayMapName, tx, ty);
            if (overlayTex != 0)
                return overlayTex;
        }

        string plainPath = MinimapService.GetMinimapTilePath(mapName, tx, ty);
        
        if (_textureCache.TryGetValue(plainPath, out uint cached))
            return cached;

        QueueTileLoad(mapName, tx, ty, plainPath);
        return 0;
    }

    /// <param name="worldAssetsLoading">
    /// True while world assets are still queued; the background reader then holds off (bounded) so tile
    /// reads do not compete with world streaming (Epic 249 R-39c).
    /// </param>
    public int ProcessPendingLoads(int maxLoads = 2, double maxBudgetMs = 5.0, bool worldAssetsLoading = false)
    {
        _deferBackgroundReads = worldAssetsLoading;
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

    private void BackgroundLoadLoop(CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                _requestSignal.Wait(cancellationToken);

                while (_pendingRequests.TryDequeue(out MinimapTileRequest request))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Interlocked.Decrement(ref _queuedRequestCount);

                    if (_textureCache.ContainsKey(request.CacheKey))
                    {
                        _queuedCacheKeys.TryRemove(request.CacheKey, out _);
                        continue;
                    }

                    WaitWhileWorldAssetsLoad(cancellationToken);
                    Interlocked.Increment(ref _inflightRequestCount);
                    try
                    {
                        DecodedMinimapTile? tile = null;
                        try
                        {
                            tile = LoadTileData(request.MapName, request.Tx, request.Ty, request.CacheKey);
                        }
                        catch (Exception ex) when (ex is not OperationCanceledException)
                        {
                            // On a dedicated thread an escaping exception would end the process; the
                            // pool task this replaced just stopped loading. Record the tile as failed.
                            ViewerLog.Trace($"[MinimapRenderer] Tile read failed {request.CacheKey}: {ex.Message}");
                        }

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
        catch (ObjectDisposedException)
        {
            // Disposed while this thread was still waiting (Dispose joins for at most one second).
        }
    }

    private void WaitWhileWorldAssetsLoad(CancellationToken cancellationToken)
    {
        for (int waitedMs = 0; _deferBackgroundReads && waitedMs < WorldLoadMaxDeferMs; waitedMs += WorldLoadDeferStepMs)
        {
            if (cancellationToken.WaitHandle.WaitOne(WorldLoadDeferStepMs))
                cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private DecodedMinimapTile? LoadTileData(string mapName, int tx, int ty, string cacheKey)
    {
        if (TryLoadCachedTile(cacheKey, out DecodedMinimapTile? cachedTile) && cachedTile != null)
            return cachedTile;

        byte[]? data = null;
        if (_preferModernTilePath)
            data = TryReadTileData(GetModernTilePath(mapName, tx, ty));

        if (data == null || data.Length == 0)
            data = TryReadTileData(cacheKey);
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

        DecodedMinimapTile? tile = DecodeTile(data, cacheKey);
        if (tile != null)
            TrySaveCachedTile(cacheKey, data);

        return tile;
    }

    /// <summary>
    /// Epic 249 R-39b: DXT BLP2 tiles keep their compressed level 0 for a direct GPU upload (no CPU decode);
    /// other encodings decode into a single pixel array. The previous route decoded into an ImageSharp image
    /// and copied it out again — two large-object allocations per tile.
    /// </summary>
    private DecodedMinimapTile? DecodeTile(byte[] data, string cacheKey)
    {
        try
        {
            BlpSummary? summary = null;
            try
            {
                using var summaryStream = new MemoryStream(data, writable: false);
                summary = BlpSummaryReader.Read(summaryStream, cacheKey);
            }
            catch (Exception)
            {
                // Header the summary reader rejects: fall through to the library decode used before R-39.
            }

            if (summary != null && _compressedUploadSupported && TryGetCompressedLevel0(summary, data, out DecodedMinimapTile? compressed))
                return compressed;

            using var ms = new MemoryStream(data);
            using var blp = new BlpFile(ms);
            if (summary is { Format: BlpFormat.Blp2 })
            {
                // Same bytes BlpFile.GetImage(0) would wrap: it swaps to BGRA only for ARGB8888 BLP2.
                byte[] pixels = blp.GetPixels(0, out int width, out int height, bgra: summary.Compression == BlpCompressionType.Uncompressed);
                return new DecodedMinimapTile(width, height, pixels);
            }

            using Image<Rgba32> image = blp.GetImage(0);
            return ConvertImage(image);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load tile {cacheKey}: {ex.Message}");
            return null;
        }
    }

    private static bool TryGetCompressedLevel0(BlpSummary summary, byte[] data, out DecodedMinimapTile? tile)
    {
        tile = null;
        if (summary.Format != BlpFormat.Blp2 || summary.Compression != BlpCompressionType.Dxtc)
            return false;

        BlpMipMapEntry? level0 = summary.MipMaps.FirstOrDefault(static mip => mip.Level == 0);
        if (level0 is not { IsInBounds: true })
            return false;

        // Same format choice as SereniaBLPLib's decoder. Its DXT1 decode yields alpha 0 for the
        // transparent block mode whatever the header's alpha depth, which is GL's RGBA DXT1 behaviour.
        (InternalFormat format, int blockBytes) = summary.AlphaDepthBits > 1
            ? (summary.PixelFormat == WowViewer.Core.Blp.BlpPixelFormat.Dxt5
                ? (InternalFormat.CompressedRgbaS3TCDxt5Ext, 16)
                : (InternalFormat.CompressedRgbaS3TCDxt3Ext, 16))
            : (InternalFormat.CompressedRgbaS3TCDxt1Ext, 8);

        long expectedBytes = (long)((summary.Width + 3) / 4) * ((summary.Height + 3) / 4) * blockBytes;
        if (level0.SizeBytes < expectedBytes || level0.Offset + expectedBytes > data.Length)
            return false;

        tile = new DecodedMinimapTile(summary.Width, summary.Height, data, (int)level0.Offset, (int)expectedBytes, format);
        return true;
    }

    // The path modern CASC clients use; identical to the world/minimaps candidate in EnumerateTileCandidates.
    private static string GetModernTilePath(string mapName, int x, int y)
        => $"world/minimaps/{mapName.ToLowerInvariant()}/map{x:D2}_{y:D2}.blp";

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

    /// <summary>
    /// A tile ready for upload: RGBA pixels, or (when <see cref="CompressedFormat"/> is set) the DXT
    /// level-0 block data at <see cref="Offset"/>/<see cref="Length"/> inside the BLP bytes.
    /// </summary>
    private sealed record DecodedMinimapTile(int Width, int Height, byte[] Pixels, int Offset = 0, int Length = 0, InternalFormat? CompressedFormat = null);
    private readonly record struct DecodedMinimapTileUpload(string CacheKey, DecodedMinimapTile? Tile);
    private readonly record struct MinimapTileRequest(string MapName, int Tx, int Ty, string CacheKey);

    private static DecodedMinimapTile ConvertImage(Image<Rgba32> image)
    {
        int width = image.Width;
        int height = image.Height;
        var pixels = new byte[width * height * 4];
        image.CopyPixelDataTo(pixels);
        return new DecodedMinimapTile(width, height, pixels);
    }

    private unsafe uint UploadTexture(DecodedMinimapTile tile)
    {
        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);
        fixed (byte* ptr = tile.Pixels)
        {
            if (tile.CompressedFormat is InternalFormat compressedFormat)
            {
                _gl.CompressedTexImage2D(TextureTarget.Texture2D, 0, compressedFormat,
                    (uint)tile.Width, (uint)tile.Height, 0, (uint)tile.Length, ptr + tile.Offset);
            }
            else
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)tile.Width, (uint)tile.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return tex;
    }

    /// <summary>
    /// Epic 249 R-39a: the tile's source BLP bytes cached on disk (<c>&lt;hash&gt;.blp</c>) are decoded the
    /// same way as a fresh read; PNG tiles written by earlier builds are still read.
    /// </summary>
    private bool TryLoadCachedTile(string plainPath, out DecodedMinimapTile? tile)
    {
        tile = null;
        string blpPath = GetCachePath(plainPath, ".blp");
        if (File.Exists(blpPath))
        {
            try
            {
                tile = DecodeTile(File.ReadAllBytes(blpPath), plainPath);
                if (tile != null)
                    return true;
            }
            catch (IOException ex)
            {
                ViewerLog.Trace($"[MinimapRenderer] Failed to read cached tile {blpPath}: {ex.Message}");
            }
        }

        return TryLoadCachedBitmap(plainPath, out tile);
    }

    private bool TryLoadCachedBitmap(string plainPath, out DecodedMinimapTile? tile)
    {
        tile = null;
        string cachePath = GetCachePath(plainPath);
        if (!File.Exists(cachePath))
            return false;

        try
        {
            using Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(cachePath);
            tile = ConvertImage(image);
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to load cached tile {cachePath}: {ex.Message}");
            return false;
        }
    }

    private void TrySaveCachedTile(string plainPath, byte[] blpBytes)
    {
        string cachePath = GetCachePath(plainPath, ".blp");
        if (File.Exists(cachePath))
            return;

        string? cacheDirectory = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(cacheDirectory))
            Directory.CreateDirectory(cacheDirectory);

        string tempPath = cachePath + ".tmp";
        try
        {
            File.WriteAllBytes(tempPath, blpBytes);
            File.Move(tempPath, cachePath, overwrite: true);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MinimapRenderer] Failed to save cached tile {cachePath}: {ex.Message}");
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private string GetCachePath(string plainPath, string extension = ".png")
    {
        string normalized = plainPath.Replace('\\', '/').ToLowerInvariant();
        string hash = Convert.ToHexString(SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(normalized))).ToLowerInvariant();
        return Path.Combine(_cacheRoot, hash + extension);
    }

    public void Dispose()
    {
        _disposeCts.Cancel();
        for (int i = 0; i < _loaderThreads.Length; i++)
            _requestSignal.Release();

        foreach (Thread thread in _loaderThreads)
            thread.Join(TimeSpan.FromSeconds(1));

        foreach (var tex in _textureCache.Values)
        {
            if (tex != 0) _gl.DeleteTexture(tex);
        }
        _textureCache.Clear();

        _requestSignal.Dispose();
        _disposeCts.Dispose();
    }
}
