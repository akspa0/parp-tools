using System.Collections.Concurrent;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal sealed class WorldMinimapRenderer : IDisposable
{
    private const int BackgroundWorkerCount = 2;

    private readonly GL _gl;
    private readonly string _clientRoot;
    private readonly string _buildLabel;
    private readonly string _looseOverlayRoot;
    private Md5TranslateIndex? _md5Index;
    private int _md5TranslateLoadAttempted;
    private readonly ConcurrentDictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, string?> _resolvedTilePathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, byte> _queuedCacheKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentQueue<MinimapTileRequest> _pendingRequests = new();
    private readonly ConcurrentQueue<DecodedMinimapTileUpload> _readyUploads = new();
    private readonly SemaphoreSlim _requestSignal = new(0);
    private readonly CancellationTokenSource _disposeCts = new();
    private readonly Task[] _loaderTasks;
    private int _completedRequestCount;
    private int _queuedRequestCount;
    private int _readyUploadCount;
    private int _inflightRequestCount;
    private int _uploadedTileCount;
    private int _failedTileCount;

    public WorldMinimapRenderer(GL gl, string clientRoot, string? buildLabel, string? looseOverlayRoot)
    {
        _gl = gl;
        _clientRoot = clientRoot?.Trim() ?? string.Empty;
        _buildLabel = buildLabel?.Trim() ?? string.Empty;
        _looseOverlayRoot = looseOverlayRoot?.Trim() ?? string.Empty;
        _loaderTasks = Enumerable.Range(0, BackgroundWorkerCount)
            .Select(_ => Task.Run(() => BackgroundLoadLoop(_disposeCts.Token), _disposeCts.Token))
            .ToArray();
    }

    public int PendingTileCount => Math.Max(0, Volatile.Read(ref _queuedRequestCount) + Volatile.Read(ref _readyUploadCount) + Volatile.Read(ref _inflightRequestCount));

    public int UploadedTileCount => Volatile.Read(ref _uploadedTileCount);

    public int FailedTileCount => Volatile.Read(ref _failedTileCount);

    public bool IsBusy => PendingTileCount > 0;

    public float LoadingProgress
    {
        get
        {
            int total = Volatile.Read(ref _completedRequestCount) + PendingTileCount;
            return total > 0 ? Volatile.Read(ref _completedRequestCount) / (float)total : 1f;
        }
    }

    public uint GetTileTexture(string mapName, int tileX, int tileY)
    {
        string plainPath = MinimapService.GetMinimapTilePath(mapName, tileX, tileY);
        if (_textureCache.TryGetValue(plainPath, out uint textureHandle))
            return textureHandle;

        QueueTileLoad(mapName, tileX, tileY, plainPath);
        return 0;
    }

    public int ProcessPendingLoads(int maxLoads = 4, double maxBudgetMs = 4.0)
    {
        if (Volatile.Read(ref _readyUploadCount) == 0 || maxLoads <= 0)
            return 0;

        int processed = 0;
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        while (processed < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _readyUploads.TryDequeue(out DecodedMinimapTileUpload upload))
        {
            Interlocked.Decrement(ref _readyUploadCount);

            if (_textureCache.ContainsKey(upload.CacheKey))
                continue;

            uint textureHandle = upload.Tile is not null ? UploadTexture(upload.Tile) : 0;
            _textureCache[upload.CacheKey] = textureHandle;
            Interlocked.Increment(ref _completedRequestCount);
            if (textureHandle != 0)
                Interlocked.Increment(ref _uploadedTileCount);
            else
                Interlocked.Increment(ref _failedTileCount);

            processed++;
        }

        return processed;
    }

    private void QueueTileLoad(string mapName, int tileX, int tileY, string cacheKey)
    {
        if (_textureCache.ContainsKey(cacheKey) || !_queuedCacheKeys.TryAdd(cacheKey, 0))
            return;

        _pendingRequests.Enqueue(new MinimapTileRequest(mapName, tileX, tileY, cacheKey));
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
                        DecodedMinimapTile? tile = LoadTileData(request.MapName, request.TileX, request.TileY, request.CacheKey);
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

    private DecodedMinimapTile? LoadTileData(string mapName, int tileX, int tileY, string primaryCandidate)
    {
        foreach (string candidate in EnumerateTileCandidates(mapName, tileX, tileY, primaryCandidate))
        {
            if (!TryReadTileData(candidate, out byte[]? data) || data is not { Length: > 0 })
                continue;

            try
            {
                using MemoryStream stream = new(data, writable: false);
                using BlpFile blp = new(stream);
                byte[] rgbaPixels = blp.GetPixels(0, out int width, out int height, bgra: false);
                if (rgbaPixels.Length == 0 || width <= 0 || height <= 0)
                    return null;

                return new DecodedMinimapTile(width, height, rgbaPixels);
            }
            catch
            {
                return null;
            }
        }

        return null;
    }

    private bool TryReadTileData(string virtualPath, out byte[]? data)
    {
        data = null;

        if (_resolvedTilePathCache.TryGetValue(virtualPath, out string? resolvedPath))
        {
            if (resolvedPath is null)
                return false;

            if (TryReadVirtualFileRaw(resolvedPath, out data) && data is { Length: > 0 })
                return true;

            _resolvedTilePathCache.TryRemove(virtualPath, out _);
        }

        foreach (string hashedCandidate in GetMd5HashCandidates(virtualPath))
        {
            if (!TryReadVirtualFileRaw(hashedCandidate, out data) || data is not { Length: > 0 })
                continue;

            _resolvedTilePathCache[virtualPath] = hashedCandidate;
            return true;
        }

        if (TryReadVirtualFileRaw(virtualPath, out data) && data is { Length: > 0 })
        {
            _resolvedTilePathCache[virtualPath] = virtualPath;
            return true;
        }

        _resolvedTilePathCache[virtualPath] = null;
        return false;
    }

    private IReadOnlyList<string> GetMd5HashCandidates(string virtualPath)
    {
        Md5TranslateIndex? index = EnsureMd5TranslateIndexLoaded();
        return index?.GetHashCandidates(virtualPath) ?? Array.Empty<string>();
    }

    private Md5TranslateIndex? EnsureMd5TranslateIndexLoaded()
    {
        if (_md5Index is not null)
            return _md5Index;

        if (Interlocked.Exchange(ref _md5TranslateLoadAttempted, 1) != 0)
            return _md5Index;

        _md5Index = TryLoadMd5TranslateIndex();
        return _md5Index;
    }

    private bool TryReadVirtualFileRaw(string virtualPath, out byte[]? data)
    {
        data = null;

        if (VirtualAssetOverlayResolver.TryReadLooseVirtualFile(virtualPath, _looseOverlayRoot, out data) && data is { Length: > 0 })
            return true;

        if (string.IsNullOrWhiteSpace(_clientRoot) || !Directory.Exists(_clientRoot))
            return false;

        try
        {
            data = ArchiveVirtualFileReader.ReadVirtualFile(
                virtualPath,
                [_clientRoot],
                WowViewerArchiveBootstrap.CreateBootstrapOptions(_buildLabel, _clientRoot));
            return data.Length > 0;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException or IOException or InvalidDataException or InvalidOperationException or NotSupportedException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private Md5TranslateIndex? TryLoadMd5TranslateIndex()
    {
        if (string.IsNullOrWhiteSpace(_clientRoot) || !Directory.Exists(_clientRoot))
            return null;

        return Md5TranslateResolver.TryLoad(
            [_clientRoot],
            archiveFileExists: candidate => TryReadVirtualFileRaw(candidate, out byte[]? bytes) && bytes is { Length: > 0 },
            archiveReadFile: candidate => TryReadVirtualFileRaw(candidate, out byte[]? bytes) ? bytes : null,
            out Md5TranslateIndex? index)
            ? index
            : null;
    }

    private static IEnumerable<string> EnumerateTileCandidates(string mapName, int tileX, int tileY, string primaryCandidate)
    {
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        List<string> yieldReturnList = [];

        void AddCandidate(string candidate)
        {
            if (!string.IsNullOrWhiteSpace(candidate) && seen.Add(candidate.Replace('\\', '/')))
                yieldReturnList.Add(candidate.Replace('\\', '/'));
        }

        string normalizedMapName = mapName.ToLowerInvariant();
        string x2 = tileX.ToString("D2");
        string y2 = tileY.ToString("D2");
        string trsFormat = $"map{tileX}_{y2}.blp";

        AddCandidate(primaryCandidate);
        AddCandidate($"{normalizedMapName}/{trsFormat}");
        AddCandidate($"textures/minimap/{normalizedMapName}/{trsFormat}");
        AddCandidate($"textures/minimap/{normalizedMapName}/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"{normalizedMapName}/map{x2}_{y2}.blp");

        string spacedMapName = InsertSpaceBeforeCapitals(mapName).ToLowerInvariant();
        if (!string.Equals(spacedMapName, normalizedMapName, StringComparison.OrdinalIgnoreCase))
        {
            AddCandidate($"{spacedMapName}/{trsFormat}");
            AddCandidate($"textures/minimap/{spacedMapName}/{trsFormat}");
            AddCandidate($"textures/minimap/{spacedMapName}/{spacedMapName}_{x2}_{y2}.blp");
            AddCandidate($"textures/minimap/{spacedMapName}/map{x2}_{y2}.blp");
            AddCandidate($"{spacedMapName}/map{x2}_{y2}.blp");
        }

        AddCandidate($"world/minimaps/{normalizedMapName}/map{x2}_{y2}.blp");
        AddCandidate($"world/minimaps/{normalizedMapName}/map{tileX}_{tileY}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{x2}_{y2}.blp");
        AddCandidate($"textures/minimap/{normalizedMapName}_{tileX}_{tileY}.blp");

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

    private unsafe uint UploadTexture(DecodedMinimapTile tile)
    {
        uint textureHandle = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureHandle);
        fixed (byte* pixelPtr = tile.Pixels)
        {
            _gl.TexImage2D(
                TextureTarget.Texture2D,
                0,
                InternalFormat.Rgba,
                (uint)tile.Width,
                (uint)tile.Height,
                0,
                PixelFormat.Rgba,
                PixelType.UnsignedByte,
                pixelPtr);
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return textureHandle;
    }

    public void Dispose()
    {
        _disposeCts.Cancel();
        for (int index = 0; index < _loaderTasks.Length; index++)
            _requestSignal.Release();

        try
        {
            Task.WaitAll(_loaderTasks, TimeSpan.FromSeconds(1));
        }
        catch (AggregateException)
        {
        }

        foreach ((_, uint textureHandle) in _textureCache)
        {
            if (textureHandle != 0)
                _gl.DeleteTexture(textureHandle);
        }

        _textureCache.Clear();
        _requestSignal.Dispose();
        _disposeCts.Dispose();
    }

    private sealed record DecodedMinimapTile(int Width, int Height, byte[] Pixels);
    private readonly record struct MinimapTileRequest(string MapName, int TileX, int TileY, string CacheKey);
    private readonly record struct DecodedMinimapTileUpload(string CacheKey, DecodedMinimapTile? Tile);
}
