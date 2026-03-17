using System.Collections.Concurrent;
using System.Text.Json;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using WoWMapConverter.Core.VLM;

namespace MdxViewer.Terrain;

internal enum WdlPreviewWarmState
{
    NotQueued,
    Loading,
    Ready,
    Failed,
}

internal sealed class WdlPreviewData
{
    public int Version { get; set; } = 1;
    public string MapDirectory { get; set; } = string.Empty;
    public int Width { get; set; } = 64;
    public int Height { get; set; } = 64;
    public byte[] PreviewRgba { get; set; } = Array.Empty<byte>();
    public float[] TileCenterHeights { get; set; } = Array.Empty<float>();
    public byte[] TileDataMask { get; set; } = Array.Empty<byte>();
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
}

internal static class WdlPreviewDataBuilder
{
    public static bool TryBuild(IDataSource dataSource, string mapDirectory, out WdlPreviewData? previewData, out string? error)
    {
        previewData = null;
        error = null;

        try
        {
            string wdlPath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdl";
            byte[]? wdlBytes = dataSource.ReadFile(wdlPath);
            if (wdlBytes == null || wdlBytes.Length == 0)
                wdlBytes = dataSource.ReadFile(wdlPath + ".mpq");

            if (wdlBytes == null || wdlBytes.Length == 0)
            {
                error = $"No WDL data found for {mapDirectory}.";
                return false;
            }

            var parsed = WdlParser.Parse(wdlBytes);
            if (parsed == null)
            {
                error = $"Failed to parse WDL for {mapDirectory}.";
                return false;
            }

            previewData = BuildFromParsed(mapDirectory, parsed);
            if (previewData.TileDataMask.All(value => value == 0))
            {
                previewData = null;
                error = $"WDL parsed but contains 0 tiles with data for {mapDirectory}.";
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            error = $"Error building WDL preview for {mapDirectory}: {ex.Message}";
            return false;
        }
    }

    public static WdlPreviewData BuildFromParsed(string mapDirectory, WdlParser.WdlData parsed)
    {
        const int size = 64;
        var preview = new WdlPreviewData
        {
            MapDirectory = mapDirectory,
            Width = size,
            Height = size,
            PreviewRgba = new byte[size * size * 4],
            TileCenterHeights = new float[size * size],
            TileDataMask = new byte[size * size],
            MinHeight = float.MaxValue,
            MaxHeight = float.MinValue,
        };

        for (int tileY = 0; tileY < size; tileY++)
        {
            for (int tileX = 0; tileX < size; tileX++)
            {
                int previewIndex = GetPreviewIndex(tileX, tileY, size);
                int sourceIndex = GetSourceTileIndex(tileX, tileY, size);
                var tile = parsed.Tiles[sourceIndex];
                if (tile?.HasData != true)
                    continue;

                preview.TileDataMask[previewIndex] = 1;
                preview.TileCenterHeights[previewIndex] = tile.Height17[8, 8];
                preview.MinHeight = MathF.Min(preview.MinHeight, tile.MinZ);
                preview.MaxHeight = MathF.Max(preview.MaxHeight, tile.MaxZ);
            }
        }

        if (preview.MinHeight == float.MaxValue)
        {
            preview.MinHeight = 0f;
            preview.MaxHeight = 0f;
        }

        float heightRange = MathF.Max(1f, preview.MaxHeight - preview.MinHeight);
        for (int tileY = 0; tileY < size; tileY++)
        {
            for (int tileX = 0; tileX < size; tileX++)
            {
                int previewIndex = GetPreviewIndex(tileX, tileY, size);
                int sourceIndex = GetSourceTileIndex(tileX, tileY, size);
                int pixelIndex = previewIndex * 4;

                if (preview.TileDataMask[previewIndex] == 0)
                {
                    preview.PreviewRgba[pixelIndex + 0] = 25;
                    preview.PreviewRgba[pixelIndex + 1] = 25;
                    preview.PreviewRgba[pixelIndex + 2] = 25;
                    preview.PreviewRgba[pixelIndex + 3] = 255;
                    continue;
                }

                var tile = parsed.Tiles[sourceIndex]!;
                float averageHeight = 0f;
                for (int row = 0; row < 17; row++)
                {
                    for (int col = 0; col < 17; col++)
                        averageHeight += tile.Height17[row, col];
                }

                averageHeight /= 17f * 17f;
                float normalized = (averageHeight - preview.MinHeight) / heightRange;
                var color = GetColor(normalized);
                preview.PreviewRgba[pixelIndex + 0] = color.r;
                preview.PreviewRgba[pixelIndex + 1] = color.g;
                preview.PreviewRgba[pixelIndex + 2] = color.b;
                preview.PreviewRgba[pixelIndex + 3] = 255;
            }
        }

        return preview;
    }

    private static int GetPreviewIndex(int tileX, int tileY, int size) => tileY * size + tileX;

    private static int GetSourceTileIndex(int tileX, int tileY, int size) => tileY * size + tileX;

    private static (byte r, byte g, byte b) GetColor(float normalized)
    {
        normalized = Math.Clamp(normalized, 0f, 1f);

        float r;
        float g;
        float b;
        if (normalized < 0.33f)
        {
            float t = normalized / 0.33f;
            r = 0f;
            g = t * 0.5f;
            b = 0.5f + t * 0.5f;
        }
        else if (normalized < 0.66f)
        {
            float t = (normalized - 0.33f) / 0.33f;
            r = t * 0.3f;
            g = 0.5f + t * 0.5f;
            b = 1f - t;
        }
        else
        {
            float t = (normalized - 0.66f) / 0.34f;
            r = 0.3f + t * 0.4f;
            g = 1f - t * 0.5f;
            b = t * 0.2f;
        }

        return ((byte)(r * 255f), (byte)(g * 255f), (byte)(b * 255f));
    }
}

internal sealed class WdlPreviewCacheService : IDisposable
{
    private readonly IDataSource _dataSource;
    private readonly string _cacheRoot;
    private readonly ConcurrentDictionary<string, WdlPreviewData> _memoryCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, string> _errors = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, Task> _warmTasks = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _buildGate = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();

    public string CacheRoot => _cacheRoot;

    public WdlPreviewCacheService(IDataSource dataSource, string cacheRoot)
    {
        _dataSource = dataSource;
        _cacheRoot = cacheRoot;
        Directory.CreateDirectory(_cacheRoot);
    }

    public void WarmMaps(IEnumerable<MapDefinition> maps)
    {
        foreach (var map in maps)
        {
            if (map.HasWdl)
                EnsurePrefetch(map.Directory);
        }
    }

    public bool TryBuildPreviewNow(string mapDirectory, out WdlPreviewData? previewData, out string? error)
    {
        previewData = null;
        error = null;

        if (TryGetPreview(mapDirectory, out previewData) && previewData != null)
            return true;

        try
        {
            _buildGate.Wait(_disposeCts.Token);
            try
            {
                if (TryGetPreview(mapDirectory, out previewData) && previewData != null)
                    return true;

                if (!WdlPreviewDataBuilder.TryBuild(_dataSource, mapDirectory, out previewData, out error) || previewData == null)
                {
                    _errors[mapDirectory] = error ?? $"Failed to build preview for {mapDirectory}.";
                    return false;
                }

                _memoryCache[mapDirectory] = previewData;
                SaveToDisk(mapDirectory, previewData);
                _errors.TryRemove(mapDirectory, out _);
                return true;
            }
            finally
            {
                _buildGate.Release();
            }
        }
        catch (OperationCanceledException)
        {
            error = $"Preview generation cancelled for {mapDirectory}.";
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            _errors[mapDirectory] = ex.Message;
            return false;
        }
    }

    public void EnsurePrefetch(string mapDirectory)
    {
        if (_disposeCts.IsCancellationRequested)
            return;

        if (TryGetPreview(mapDirectory, out _))
            return;

        _warmTasks.GetOrAdd(mapDirectory, key => Task.Run(() => WarmMapCore(key, _disposeCts.Token), _disposeCts.Token));
    }

    public bool TryGetPreview(string mapDirectory, out WdlPreviewData? previewData)
    {
        if (_memoryCache.TryGetValue(mapDirectory, out previewData))
            return true;

        if (TryLoadFromDisk(mapDirectory, out previewData))
            return true;

        previewData = null;
        return false;
    }

    public WdlPreviewWarmState GetState(string mapDirectory)
    {
        if (_memoryCache.ContainsKey(mapDirectory))
            return WdlPreviewWarmState.Ready;

        if (TryLoadFromDisk(mapDirectory, out _))
            return WdlPreviewWarmState.Ready;

        if (_warmTasks.TryGetValue(mapDirectory, out var task) && !task.IsCompleted)
            return WdlPreviewWarmState.Loading;

        if (_errors.ContainsKey(mapDirectory))
            return WdlPreviewWarmState.Failed;

        return WdlPreviewWarmState.NotQueued;
    }

    public string? GetError(string mapDirectory)
    {
        return _errors.TryGetValue(mapDirectory, out var error) ? error : null;
    }

    private void WarmMapCore(string mapDirectory, CancellationToken cancellationToken)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (TryGetPreview(mapDirectory, out _))
                return;

            _buildGate.Wait(cancellationToken);
            try
            {
                if (TryGetPreview(mapDirectory, out _))
                    return;

                if (!WdlPreviewDataBuilder.TryBuild(_dataSource, mapDirectory, out var previewData, out var error) || previewData == null)
                {
                    _errors[mapDirectory] = error ?? $"Failed to warm preview for {mapDirectory}.";
                    return;
                }

                _memoryCache[mapDirectory] = previewData;
                SaveToDisk(mapDirectory, previewData);
                _errors.TryRemove(mapDirectory, out _);
                ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview Cache] Warmed {mapDirectory}");
            }
            finally
            {
                _buildGate.Release();
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            _errors[mapDirectory] = ex.Message;
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL Preview Cache] Failed for {mapDirectory}: {ex.Message}");
        }
        finally
        {
            _warmTasks.TryRemove(mapDirectory, out _);
        }
    }

    private bool TryLoadFromDisk(string mapDirectory, out WdlPreviewData? previewData)
    {
        previewData = null;
        string path = GetCachePath(mapDirectory);
        if (!File.Exists(path))
            return false;

        try
        {
            var loaded = JsonSerializer.Deserialize<WdlPreviewData>(File.ReadAllText(path));
            if (loaded == null || loaded.PreviewRgba.Length != loaded.Width * loaded.Height * 4 ||
                loaded.TileCenterHeights.Length != 64 * 64 || loaded.TileDataMask.Length != 64 * 64)
            {
                return false;
            }

            _memoryCache[mapDirectory] = loaded;
            previewData = loaded;
            return true;
        }
        catch (Exception ex)
        {
            _errors[mapDirectory] = ex.Message;
            return false;
        }
    }

    private void SaveToDisk(string mapDirectory, WdlPreviewData previewData)
    {
        string finalPath = GetCachePath(mapDirectory);
        string tempPath = finalPath + ".tmp";
        File.WriteAllText(tempPath, JsonSerializer.Serialize(previewData));
        File.Move(tempPath, finalPath, overwrite: true);
    }

    private string GetCachePath(string mapDirectory)
    {
        string fileName = mapDirectory.ToLowerInvariant() + ".wdlpreview.json";
        return Path.Combine(_cacheRoot, fileName);
    }

    public void Dispose()
    {
        _disposeCts.Cancel();
        _disposeCts.Dispose();
        _buildGate.Dispose();
    }
}