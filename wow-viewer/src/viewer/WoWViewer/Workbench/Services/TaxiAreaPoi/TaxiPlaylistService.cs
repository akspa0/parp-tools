using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWViewer.Logging;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Authoritative manager for taxi flight playlists, allowing users and automated harnesses
/// to chain multiple taxi routes into a seamless multi-segment flight and continuous video recording.
/// </summary>
internal sealed class TaxiPlaylistService
{
    private readonly IViewerAppHost _host;
    private readonly List<TaxiPlaylistItem> _items = new();

    private int _currentIndex = -1;
    private bool _isPlaying;
    private bool _isRecording;
    private float _segmentStartTravel;
    private bool _hasTraveledSignificant;
    private string _statusText = "Idle";

    internal TaxiPlaylistService(IViewerAppHost host)
    {
        _host = host;
    }

    public IReadOnlyList<TaxiPlaylistItem> Items => _items;
    public int CurrentIndex => _currentIndex;
    public bool IsPlaying => _isPlaying;
    public bool IsRecording => _isRecording;
    public bool Loop { get; set; }
    public string StatusText => _statusText;

    public TaxiPlaylistItem? CurrentItem =>
        _currentIndex >= 0 && _currentIndex < _items.Count ? _items[_currentIndex] : null;

    public bool AddRoute(int routeId)
    {
        WorldScene? scene = _host.WorldScene;
        TaxiPathLoader? loader = scene?.TaxiActors.TaxiLoader;
        if (loader == null)
            return false;

        TaxiPathLoader.TaxiRoute? route = loader.Routes.FirstOrDefault(r => r.PathId == routeId);
        if (route == null)
            return false;

        string fromName = scene!.TaxiActors.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
        string toName = scene.TaxiActors.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
        float length = ComputeRouteLength(route.Waypoints);

        _items.Add(new TaxiPlaylistItem(route.PathId, route.FromNodeId, route.ToNodeId, fromName, toName, length));
        _statusText = $"Added {fromName} -> {toName} (#{routeId}) to playlist ({_items.Count} total).";
        return true;
    }

    public void AddItem(TaxiPlaylistItem item)
    {
        _items.Add(item);
    }

    public void RemoveAt(int index)
    {
        if (index >= 0 && index < _items.Count)
        {
            if (_isPlaying && _currentIndex == index)
                StopPlaylist("Active route was removed from playlist.");

            _items.RemoveAt(index);
            if (_currentIndex >= _items.Count)
                _currentIndex = _items.Count - 1;
        }
    }

    public void MoveUp(int index)
    {
        if (index > 0 && index < _items.Count)
        {
            (_items[index - 1], _items[index]) = (_items[index], _items[index - 1]);
            if (_currentIndex == index)
                _currentIndex = index - 1;
            else if (_currentIndex == index - 1)
                _currentIndex = index;
        }
    }

    public void MoveDown(int index)
    {
        if (index >= 0 && index < _items.Count - 1)
        {
            (_items[index + 1], _items[index]) = (_items[index], _items[index + 1]);
            if (_currentIndex == index)
                _currentIndex = index + 1;
            else if (_currentIndex == index + 1)
                _currentIndex = index;
        }
    }

    public void Clear()
    {
        if (_isPlaying)
            StopPlaylist("Playlist cleared.");

        _items.Clear();
        _currentIndex = -1;
        _statusText = "Playlist cleared.";
    }

    /// <summary>
    /// Discovers connected outgoing taxi routes starting from a given flight node,
    /// avoiding immediate ping-pong backtracking, up to <paramref name="maxHops"/>.
    /// </summary>
    public int BuildAutoChain(int startNodeId, int maxHops = 4)
    {
        WorldScene? scene = _host.WorldScene;
        TaxiPathLoader? loader = scene?.TaxiActors.TaxiLoader;
        if (loader == null || maxHops <= 0)
            return 0;

        int added = 0;
        int currentNodeId = startNodeId;
        int previousNodeId = -1;

        for (int hop = 0; hop < maxHops; hop++)
        {
            List<TaxiPathLoader.TaxiRoute> outgoing = loader.Routes
                .Where(r => r.FromNodeId == currentNodeId)
                .ToList();

            if (outgoing.Count == 0)
                break;

            // Prefer an outgoing route that doesn't just reverse the immediately preceding hop
            TaxiPathLoader.TaxiRoute nextRoute = outgoing.FirstOrDefault(r => r.ToNodeId != previousNodeId)
                ?? outgoing[0];

            if (AddRoute(nextRoute.PathId))
            {
                added++;
                previousNodeId = currentNodeId;
                currentNodeId = nextRoute.ToNodeId;
            }
            else
            {
                break;
            }
        }

        _statusText = $"Auto-chained {added} routes from node #{startNodeId}.";
        return added;
    }

    /// <summary>
    /// Starts playlist flight, optionally beginning a continuous video recording.
    /// </summary>
    public bool StartPlaylist(
        bool recordVideo = false,
        int videoFps = 60,
        bool includeUi = true,
        string? customOutput = null,
        bool exitAfterRecord = false,
        bool includeShowreelOverlay = false)
    {
        if (_items.Count == 0)
        {
            _statusText = "Cannot start empty playlist.";
            return false;
        }

        WorldScene? scene = _host.WorldScene;
        if (scene == null)
        {
            _statusText = "No active world scene.";
            return false;
        }

        _currentIndex = 0;
        _isPlaying = true;
        _isRecording = recordVideo;

        TaxiPlaylistItem firstItem = _items[0];
        ActivateRouteSegment(firstItem.PathId);

        if (recordVideo)
        {
            string output = customOutput ?? Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "captures",
                $"taxi_playlist_{DateTime.Now:yyyyMMdd_HHmmss}.mp4");

            var request = new RecordingRequest
            {
                SourceKind = RecordingSourceKind.TaxiRoute,
                OutputPathOverride = output,
                Fps = videoFps,
                IncludeUi = includeUi,
                IncludeShowreelOverlay = includeShowreelOverlay || _host.ShowreelOverlay.Config.EnableOverlay,
                AutoStopOnRouteArrival = false, // Playlist manager handles destination arrival!
                RestoreUiChromeOnStop = includeUi,
                PreviousHideUiChrome = _host.HideUiChrome,
                ExitAfterRecording = exitAfterRecord,
            };

            if (!_host.RecordingCoordinator.TryStartRecording(request, out string? error))
            {
                _isRecording = false;
                _statusText = $"Playlist started, but video recording failed: {error}";
                ViewerLog.Error(ViewerLog.Category.Export, _statusText);
                return true;
            }
        }

        _statusText = $"Playing playlist segment 1/{_items.Count}: {firstItem.DisplayLabel}";
        ViewerLog.Important(ViewerLog.Category.General, $"[TaxiPlaylist] Started: {_items.Count} segments.");
        return true;
    }

    /// <summary>
    /// Advances playback to the next route in the playlist, or finishes.
    /// </summary>
    public void AdvanceNextRoute()
    {
        if (!_isPlaying || _items.Count == 0)
            return;

        int nextIndex = _currentIndex + 1;
        if (nextIndex < _items.Count)
        {
            _currentIndex = nextIndex;
            TaxiPlaylistItem item = _items[_currentIndex];
            ActivateRouteSegment(item.PathId);
            _statusText = $"Advanced to segment {nextIndex + 1}/{_items.Count}: {item.DisplayLabel}";
            ViewerLog.Important(ViewerLog.Category.General, $"[TaxiPlaylist] {StatusText}");
        }
        else if (Loop)
        {
            _currentIndex = 0;
            TaxiPlaylistItem item = _items[0];
            ActivateRouteSegment(item.PathId);
            _statusText = $"Looped to segment 1/{_items.Count}: {item.DisplayLabel}";
            ViewerLog.Important(ViewerLog.Category.General, $"[TaxiPlaylist] {StatusText}");
        }
        else
        {
            StopPlaylist($"Taxi playlist completed ({_items.Count} routes).");
        }
    }

    public void StopPlaylist(string reason = "Stopped")
    {
        if (!_isPlaying)
            return;

        _isPlaying = false;
        _statusText = reason;

        if (_isRecording && _host.RecordingCoordinator.IsRecording)
        {
            _host.RecordingCoordinator.StopRecording(reason);
            _isRecording = false;
        }

        _host.StopTaxiRideCamera("Playlist stopped.");
        ViewerLog.Important(ViewerLog.Category.General, $"[TaxiPlaylist] Stopped: {reason}");
    }

    /// <summary>
    /// Evaluates route travel progression and triggers auto-advance on arrival.
    /// </summary>
    public void Update(double dt)
    {
        if (!_isPlaying || _items.Count == 0 || _currentIndex < 0 || _currentIndex >= _items.Count)
            return;

        WorldScene? scene = _host.WorldScene;
        if (scene == null)
            return;

        TaxiPlaylistItem current = _items[_currentIndex];
        if (scene.TaxiActors.TryGetTaxiRouteProgress(current.PathId, out float travelDist, out float totalLen))
        {
            float deltaTravel = travelDist - _segmentStartTravel;
            if (deltaTravel < 0f && totalLen > 0f)
                deltaTravel += totalLen;

            if (deltaTravel > totalLen * 0.25f)
                _hasTraveledSignificant = true;

            // Arrival check: completed 98% of path length
            if (_hasTraveledSignificant && (deltaTravel >= totalLen * 0.98f || deltaTravel < 10f))
            {
                AdvanceNextRoute();
            }
        }
    }

    public TaxiPlaylistStatus GetStatus()
    {
        if (!_isPlaying || _items.Count == 0 || _currentIndex < 0 || _currentIndex >= _items.Count)
        {
            return new TaxiPlaylistStatus(
                IsPlaying: false,
                IsRecording: false,
                CurrentIndex: _currentIndex,
                TotalCount: _items.Count,
                CurrentItem: null,
                SegmentProgressFraction: 0f,
                OverallProgressFraction: 0f,
                StatusText: _statusText);
        }

        TaxiPlaylistItem item = _items[_currentIndex];
        float segFrac = 0f;
        if (_host.WorldScene != null &&
            _host.WorldScene.TaxiActors.TryGetTaxiRouteProgress(item.PathId, out float travelDist, out float totalLen) &&
            totalLen > 0f)
        {
            segFrac = Math.Clamp(travelDist / totalLen, 0f, 1f);
        }

        float overallFrac = Math.Clamp((_currentIndex + segFrac) / Math.Max(1, _items.Count), 0f, 1f);
        return new TaxiPlaylistStatus(
            IsPlaying: _isPlaying,
            IsRecording: _isRecording,
            CurrentIndex: _currentIndex,
            TotalCount: _items.Count,
            CurrentItem: item,
            SegmentProgressFraction: segFrac,
            OverallProgressFraction: overallFrac,
            StatusText: _statusText);
    }

    private void ActivateRouteSegment(int pathId)
    {
        WorldScene? scene = _host.WorldScene;
        if (scene == null)
            return;

        scene.TaxiActors.SelectedTaxiRouteId = pathId;
        scene.TaxiActors.SelectedTaxiNodeId = -1;
        scene.TaxiActors.ResetTaxiRouteTravel(pathId);
        _host.TaxiPanel.AttachTaxiRideCamera(pathId);

        _segmentStartTravel = 0f;
        _hasTraveledSignificant = false;
    }

    private static float ComputeRouteLength(List<Vector3> waypoints)
    {
        if (waypoints == null || waypoints.Count < 2)
            return 0f;

        float len = 0f;
        for (int i = 0; i < waypoints.Count - 1; i++)
            len += Vector3.Distance(waypoints[i], waypoints[i + 1]);
        return len;
    }
}
