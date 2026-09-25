using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// WDL map preview: preview cache identity/warmup, selected-map preview state, the preview window, and opening a map at its default spawn.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class WdlPreviewService
{
    private readonly IViewerAppHost _host;

    internal WdlPreviewService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private ref GL _gl => ref _host.Gl;
    private ref Vector3? _pendingWorldSpawnOverride => ref _host.PendingWorldSpawnOverride;
    private ref MapDefinition? _selectedMapForPreview => ref _host.SelectedMapForPreview;
    private ref Vector2? _selectedSpawnTile => ref _host.SelectedSpawnTile;
    private ref bool _showWdlPreview => ref _host.ShowWdlPreview;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref WdlPreviewCacheService? _wdlPreviewCacheService => ref _host.WdlPreviewCacheService;
    private ref WdlPreviewRenderer? _wdlPreviewRenderer => ref _host.WdlPreviewRenderer;
    private void LoadFileFromDataSource(string virtualPath) => _host.LoadFileFromDataSource(virtualPath);
    private void LoadMapAtDefaultSpawn(MapDefinition map) => _host.LoadMapAtDefaultSpawn(map);
    private string? ResolveMapWdtPath(string mapDirectory) => _host.ResolveMapWdtPath(mapDirectory);

    private string _wdlPreviewWarmupStatus = string.Empty;

    internal void InitializeWdlPreviewSupport()
    {
        if (_dataSource == null)
            return;

        string cacheIdentity = BuildWdlPreviewCacheIdentity();
        string cacheSegment = BuildCacheSegment(cacheIdentity);

        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = new WdlPreviewCacheService(_dataSource, Path.Combine(CacheDir, "wdl-preview", cacheSegment));
        _wdlPreviewWarmupStatus = string.Empty;
    }

    internal static string BuildCacheSegment(string cacheIdentity)
    {
        string cacheSegment = string.IsNullOrWhiteSpace(cacheIdentity)
            ? "default"
            : Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(cacheIdentity))).ToLowerInvariant();
        return string.IsNullOrWhiteSpace(cacheSegment) ? "default" : cacheSegment;
    }

    internal string BuildWdlPreviewCacheIdentity()
    {
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var parts = new List<string> { mpqDataSource.GamePath };
            parts.AddRange(mpqDataSource.OverlayRoots.OrderBy(path => path, StringComparer.OrdinalIgnoreCase));
            return string.Join("||", parts);
        }

        return _dataSource?.Name ?? "default";
    }

    internal void ResetWdlPreviewSupport()
    {
        _wdlPreviewCacheService?.Dispose();
        _wdlPreviewCacheService = null;
        _wdlPreviewWarmupStatus = string.Empty;
        _wdlPreviewRenderer?.ClearPreview();
    }

    internal void WarmDiscoveredWdlPreviews()
    {
        if (_wdlPreviewCacheService == null || _discoveredMaps.Count == 0)
            return;

        var mapsWithWdl = _discoveredMaps.Where(map => map.HasWdl).ToList();
        if (mapsWithWdl.Count == 0)
            return;

        _wdlPreviewCacheService.WarmMaps(mapsWithWdl);
        _wdlPreviewWarmupStatus = $"Warming {mapsWithWdl.Count} WDL previews in the background.";
    }

    internal bool CanUseWdlPreviewFeature()
    {
        return _dataSource != null;
    }

    private void LoadSelectedPreviewMapAtSpawn()
    {
        if (_selectedMapForPreview == null || !_selectedMapForPreview.HasWdt)
            return;

        string? resolvedWdtPath = ResolveMapWdtPath(_selectedMapForPreview.Directory);
        if (string.IsNullOrWhiteSpace(resolvedWdtPath))
        {
            _statusMessage = $"Failed to resolve WDT for {_selectedMapForPreview.Directory}.";
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[WorldLoad] Failed to resolve map WDT for {_selectedMapForPreview.Directory} from spawn preview.");
            return;
        }

        _pendingWorldSpawnOverride = _selectedSpawnTile.HasValue && _wdlPreviewRenderer?.HasPreview == true
            ? _wdlPreviewRenderer.TileToWorldPosition(
                (int)_selectedSpawnTile.Value.X,
                (int)_selectedSpawnTile.Value.Y)
            : null;

        LoadFileFromDataSource(resolvedWdtPath);

        _showWdlPreview = false;
    }

    internal void OpenWdlPreview(MapDefinition map)
    {
        if (!map.HasWdt)
            return;

        if (!map.HasWdl || !CanUseWdlPreviewFeature())
        {
            LoadMapAtDefaultSpawn(map);
            return;
        }

        _selectedMapForPreview = map;
        _selectedSpawnTile = null;
        _showWdlPreview = true;

        if (_wdlPreviewRenderer == null)
            _wdlPreviewRenderer = new WdlPreviewRenderer(_gl);

        TryLoadSelectedWdlPreviewFromCache(map.Directory);

        if (!_wdlPreviewRenderer.HasPreview && _wdlPreviewCacheService != null)
        {
            if (_wdlPreviewCacheService.TryBuildPreviewNow(map.Directory, out var previewData, out var error) && previewData != null)
            {
                _wdlPreviewRenderer.LoadPreview(previewData);
                _wdlPreviewWarmupStatus = string.Empty;
            }
            else if (!string.IsNullOrWhiteSpace(error))
            {
                _wdlPreviewWarmupStatus = error;
            }
        }

        if (_wdlPreviewRenderer.HasPreview)
        {
            _showWdlPreview = true;
            return;
        }

        if (GetSelectedWdlPreviewState() == WdlPreviewWarmState.Failed)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain,
                $"[WDL] Preview unavailable for {map.Directory}; using default map spawn.");
            LoadMapAtDefaultSpawn(map);
            return;
        }
    }

    private void TryLoadSelectedWdlPreviewFromCache(string mapDirectory)
    {
        if (_wdlPreviewRenderer == null)
            return;

        if (_wdlPreviewCacheService != null && _wdlPreviewCacheService.TryGetPreview(mapDirectory, out var previewData) && previewData != null)
        {
            _wdlPreviewRenderer.LoadPreview(previewData);
            _wdlPreviewWarmupStatus = string.Empty;
            return;
        }

        _wdlPreviewRenderer.ClearPreview();

        if (_wdlPreviewCacheService != null)
        {
            _wdlPreviewCacheService.EnsurePrefetch(mapDirectory);
            var state = _wdlPreviewCacheService.GetState(mapDirectory);
            _wdlPreviewWarmupStatus = state switch
            {
                WdlPreviewWarmState.Ready => string.Empty,
                WdlPreviewWarmState.Failed => _wdlPreviewCacheService.GetError(mapDirectory) ?? $"Failed to prepare preview for {mapDirectory}.",
                _ => $"Preparing WDL preview for {mapDirectory}...",
            };
            return;
        }

        if (_dataSource != null)
        {
            bool loaded = _wdlPreviewRenderer.LoadWdl(_dataSource, mapDirectory);
            _wdlPreviewWarmupStatus = loaded ? string.Empty : _wdlPreviewRenderer.LastError ?? string.Empty;
        }
    }

    private WdlPreviewWarmState GetSelectedWdlPreviewState()
    {
        if (_wdlPreviewRenderer?.HasPreview == true)
            return WdlPreviewWarmState.Ready;

        if (_selectedMapForPreview == null)
            return WdlPreviewWarmState.NotQueued;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetState(_selectedMapForPreview.Directory);

        return string.IsNullOrWhiteSpace(_wdlPreviewRenderer?.LastError)
            ? WdlPreviewWarmState.Loading
            : WdlPreviewWarmState.Failed;
    }

    private string? GetSelectedWdlPreviewError()
    {
        if (_selectedMapForPreview == null)
            return null;

        if (_wdlPreviewCacheService != null)
            return _wdlPreviewCacheService.GetError(_selectedMapForPreview.Directory);

        return _wdlPreviewRenderer?.LastError;
    }

    internal (int total, int ready, int loading, int failed) GetWdlPreviewWarmupStats()
    {
        if (_wdlPreviewCacheService == null || _discoveredMaps.Count == 0)
            return (0, 0, 0, 0);

        int total = 0;
        int ready = 0;
        int loading = 0;
        int failed = 0;

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

        return (total, ready, loading, failed);
    }
}
