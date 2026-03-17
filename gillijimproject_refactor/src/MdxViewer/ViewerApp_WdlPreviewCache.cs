using MdxViewer.Logging;
using MdxViewer.Terrain;

namespace MdxViewer;

public partial class ViewerApp
{
    internal static bool IsWdlPreviewBuildSupported(string? build)
    {
        return !string.IsNullOrWhiteSpace(build) &&
               build.StartsWith("0.5.", StringComparison.OrdinalIgnoreCase);
    }

    private bool CanUseWdlPreviewFeature()
    {
        return IsWdlPreviewBuildSupported(_dbcBuild);
    }

    private void ResetWdlPreviewSupport()
    {
        _showWdlPreview = false;
        _selectedMapForPreview = null;
        _selectedSpawnTile = null;
        _wdlPreviewRenderer?.Dispose();
        _wdlPreviewRenderer = null;
        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = null;
        _wdlPreviewWarmupStatus = string.Empty;
    }

    private void InitializeWdlPreviewSupport(string gamePath)
    {
        string cacheRoot = Path.Combine(gamePath, ".cache", "MdxViewer", "wdl-preview");
        Directory.CreateDirectory(cacheRoot);

        _wdlPreviewRenderer?.Dispose();
        _wdlPreviewRenderer = new WdlPreviewRenderer(_gl);

        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = new WdlPreviewCacheService(_dataSource!, cacheRoot);
        _wdlPreviewWarmupStatus = $"WDL preview cache: {cacheRoot}";
    }

    private void WarmDiscoveredWdlPreviews()
    {
        if (!CanUseWdlPreviewFeature())
        {
            _wdlPreviewWarmupStatus = $"WDL preview is currently only enabled for Alpha 0.5.x. Current build: {_dbcBuild ?? "unknown"}.";
            return;
        }

        if (_wdlPreviewCacheService == null)
            return;

        var mapsWithWdl = _discoveredMaps.Where(map => map.HasWdl).ToArray();
        int cached = 0;
        foreach (var map in mapsWithWdl)
        {
            if (_wdlPreviewCacheService.TryGetPreview(map.Directory, out _))
                cached++;
        }

        _wdlPreviewWarmupStatus = mapsWithWdl.Length == 0
            ? "No WDL previews found for the loaded client folder."
            : $"WDL preview cache ready. {cached}/{mapsWithWdl.Length} preview(s) already cached at {_wdlPreviewCacheService.CacheRoot}. New previews build on demand.";
    }

    private void OpenWdlPreview(MapDefinition map)
    {
        if (_dataSource == null)
            return;

        if (!CanUseWdlPreviewFeature())
        {
            _statusMessage = $"WDL preview is not supported for {_dbcBuild ?? "this client"}. Loading {map.Name} directly.";
            if (map.HasWdt)
            {
                string wdtPath = $"World\\Maps\\{map.Directory}\\{map.Directory}.wdt";
                LoadFileFromDataSource(wdtPath);
            }
            return;
        }

        _selectedMapForPreview = map;
        _selectedSpawnTile = null;
        _showWdlPreview = true;
        _wdlPreviewRenderer ??= new WdlPreviewRenderer(_gl);
        _wdlPreviewRenderer.ClearPreview();

        if (_wdlPreviewCacheService != null)
        {
            if (TryLoadSelectedWdlPreviewFromCache(map.Directory))
            {
                ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Loaded cached preview for {map.Directory}");
                return;
            }

            if (_wdlPreviewCacheService.TryBuildPreviewNow(map.Directory, out var previewData, out var error) &&
                previewData != null &&
                _wdlPreviewRenderer.LoadPreview(previewData))
            {
                ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Built preview on demand for {map.Directory}");
                return;
            }

            _wdlPreviewRenderer.LastError = error ?? $"Failed to build WDL preview for {map.Directory}.";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL Preview] {_wdlPreviewRenderer.LastError}");
            return;
        }

        bool loaded = _wdlPreviewRenderer.LoadWdl(_dataSource, map.Directory);
        ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL Preview] Direct load result for {map.Directory}: {loaded}, HasPreview: {_wdlPreviewRenderer.HasPreview}");
    }

    private bool TryLoadSelectedWdlPreviewFromCache(string mapDirectory)
    {
        if (_wdlPreviewCacheService == null || _wdlPreviewRenderer == null)
            return false;

        if (!_wdlPreviewCacheService.TryGetPreview(mapDirectory, out var previewData) || previewData == null)
            return false;

        return _wdlPreviewRenderer.LoadPreview(previewData);
    }

    private WdlPreviewWarmState GetSelectedWdlPreviewState()
    {
        if (_selectedMapForPreview == null)
            return WdlPreviewWarmState.NotQueued;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetState(_selectedMapForPreview.Directory);

        return _wdlPreviewRenderer?.HasPreview == true
            ? WdlPreviewWarmState.Ready
            : WdlPreviewWarmState.NotQueued;
    }

    private string? GetSelectedWdlPreviewError()
    {
        if (_selectedMapForPreview == null)
            return null;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetError(_selectedMapForPreview.Directory) ?? _wdlPreviewRenderer?.LastError;

        return _wdlPreviewRenderer?.LastError;
    }

    private (int ready, int loading, int failed, int total) GetWdlPreviewWarmupStats()
    {
        if (!CanUseWdlPreviewFeature() || _wdlPreviewCacheService == null)
            return (0, 0, 0, 0);

        int ready = 0;
        int loading = 0;
        int failed = 0;
        int total = 0;
        foreach (var map in _discoveredMaps)
        {
            if (!map.HasWdl)
                continue;

            total++;
            switch (_wdlPreviewCacheService.GetState(map.Directory))
            {
                case WdlPreviewWarmState.Ready:
                    ready++;
                    break;
                case WdlPreviewWarmState.Loading:
                    loading++;
                    break;
                case WdlPreviewWarmState.Failed:
                    failed++;
                    break;
            }
        }

        return (ready, loading, failed, total);
    }
}