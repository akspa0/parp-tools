using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.PromoVideo;

namespace WoWViewer;

/// <summary>
/// Dynamic showreel HUD and broadcast tour overlay service.
/// Renders cinematic zone transition banners, live coordinates, ADT tile/chunk telemetry,
/// flight progress, landmark discovery callouts, and technical showcase badges directly
/// onto the scene for video capture and live viewing without external video editing.
/// Enhanced with live engine pipeline telemetry, raw data processing metrics, and
/// real-time frame pacing / hitch diagnostic alerts.
/// </summary>
internal sealed class ShowreelOverlayService
{
    private readonly IViewerAppHost _host;

    public ShowreelOverlayConfig Config { get; } = new();

    private string? _activeBannerZone;
    private string? _activeBannerSubzone;
    private int _activeBannerAreaId;
    private float _bannerElapsedSeconds;
    private bool _bannerActive;

    private string? _lastSeenZone;
    private string? _lastSeenSubzone;

    private float _badgeTimerSeconds;
    private int _badgeIndex;

    // Performance & Hitch Diagnostics Tracking
    private float _lastFrameDt;
    private int _sessionHitchCount;
    private float _worstFrameTimeMs;
    private string? _lastHitchStage;
    private float _lastHitchDurationMs;
    private float _hitchAlertTimerSeconds;
    private const float HitchAlertHoldSeconds = 2.5f;

    private static readonly (string Category, string Description)[] ShowcaseBadges =
    {
        ("WoW Engine Architecture", "Authentic 1.12.1 DBC Flight Mechanics & World Navigation"),
        ("Terrain Engine", "64-bit ADT Hole Punching & High-Resolution Alpha Liquids"),
        ("Data Pipeline", "Multi-Era DBC/DBCD CASC & MPQ Archive Loading"),
        ("Graphics Pipeline", "Native M2 GPU Hardware Skinning & Instanced Geometry"),
        ("Cartography Platform", "Alpha 0.5.3 to LK Multi-Era World Exploration"),
    };

    internal ShowreelOverlayService(IViewerAppHost host)
    {
        _host = host;
    }

    /// <summary>
    /// Per-frame update for area transitions, badge timers, hitch diagnostics, and banner animations.
    /// </summary>
    public void Update(double dt)
    {
        if (!ShouldDraw())
            return;

        float delta = (float)Math.Max(0.0, dt);
        _lastFrameDt = delta;

        // 1. Rolling frame hitch detection & stage attribution
        float frameTimeMs = delta * 1000f;
        if (frameTimeMs >= Config.HitchThresholdMs && Config.ShowHitchAlerts)
        {
            _sessionHitchCount++;
            _lastHitchDurationMs = frameTimeMs;
            _worstFrameTimeMs = Math.Max(_worstFrameTimeMs, frameTimeMs);
            _lastHitchStage = DetectDominantStage();
            _hitchAlertTimerSeconds = HitchAlertHoldSeconds;
        }
        else if (_hitchAlertTimerSeconds > 0f)
        {
            _hitchAlertTimerSeconds = Math.Max(0f, _hitchAlertTimerSeconds - delta);
        }

        // 2. Zone transition detection
        if (Config.ShowZoneBanners)
        {
            var areaLookup = _host.CurrentAreaLookup;
            string currentZone = areaLookup?.ZoneText ?? string.Empty;
            string currentSubzone = areaLookup?.SubzoneText ?? _host.CurrentAreaName;

            if (!string.IsNullOrEmpty(currentZone) &&
                (currentZone != _lastSeenZone || currentSubzone != _lastSeenSubzone))
            {
                _lastSeenZone = currentZone;
                _lastSeenSubzone = currentSubzone;
                _activeBannerZone = currentZone;
                _activeBannerSubzone = !string.Equals(currentSubzone, currentZone, StringComparison.OrdinalIgnoreCase)
                    ? currentSubzone
                    : null;
                _activeBannerAreaId = areaLookup?.CanonicalAreaId ?? areaLookup?.RawAreaId ?? 0;
                _bannerElapsedSeconds = 0f;
                _bannerActive = true;
            }

            if (_bannerActive)
            {
                _bannerElapsedSeconds += delta;
                if (_bannerElapsedSeconds >= Config.BannerHoldSeconds + Config.BannerFadeSeconds)
                {
                    _bannerActive = false;
                }
            }
        }

        // 3. Engine showcase badge cycling
        if (Config.ShowEngineBadges)
        {
            _badgeTimerSeconds += delta;
            if (_badgeTimerSeconds >= 8.0f)
            {
                _badgeTimerSeconds = 0f;
                _badgeIndex = (_badgeIndex + 1) % ShowcaseBadges.Length;
            }
        }
    }

    /// <summary>
    /// Evaluates whether the showreel/tour overlay should be actively updated and drawn.
    /// Only renders when the tour feature is turned on for video recording, or when explicitly
    /// enabled via preview in the viewport.
    /// </summary>
    public bool ShouldDraw()
    {
        if (Config.PreviewInViewport)
            return true;

        if (_host.RecordingCoordinator.IsRecording)
        {
            var session = _host.RecordingCoordinator.ActiveSession;
            if (session != null)
            {
                if (session.HasTour || Config.EnableOverlay)
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Identifies the render stage that consumed the largest CPU time in the last frame.
    /// </summary>
    private string DetectDominantStage()
    {
        var scene = _host.WorldScene;
        if (scene == null)
            return "Render";

        var stats = scene.LastRenderFrameStats;
        if (stats.GcPauseMs > 8.0)
            return "GC Pause";

        (string Name, double Ms)[] stages =
        [
            ("DeferredLoads", stats.DeferredAssetLoads.DurationMs),
            ("Terrain", stats.Terrain.DurationMs),
            ("WmoSubmission", stats.WmoSubmission.DurationMs + stats.WmoTransparentSubmission.DurationMs),
            ("M2Submission", stats.MdxOpaqueSubmission.DurationMs + stats.MdxTransparentSubmission.DurationMs),
            ("M2Visibility", stats.MdxVisibility.DurationMs),
            ("M2Animation", stats.MdxAnimation.DurationMs),
            ("Liquid", stats.Liquid.DurationMs),
            ("Lighting", stats.Lighting.DurationMs),
            ("Wdl", stats.Wdl.DurationMs),
            ("Taxi", stats.TaxiActorUpdate.DurationMs),
            ("Overlay", stats.Overlay.DurationMs)
        ];

        string dominant = "Render";
        double maxMs = 0.0;
        foreach (var (name, ms) in stages)
        {
            if (ms > maxMs)
            {
                maxMs = ms;
                dominant = name;
            }
        }

        return maxMs > 1.5 ? dominant : "Sync";
    }

    /// <summary>
    /// Captures a telemetry snapshot of the current frame including live performance,
    /// pipeline counters, and format metrics.
    /// </summary>
    public ShowreelTelemetrySnapshot CaptureTelemetrySnapshot()
    {
        Vector3 pos = _host.Camera.Position;
        float yaw = _host.Camera.Yaw;
        float pitch = _host.Camera.Pitch;
        string compass = ShowreelTelemetryMath.GetCompassDirection(yaw);

        ShowreelTelemetryMath.WorldToAdtTile(pos, out int tileX, out int tileY);
        ShowreelTelemetryMath.WorldToMcnkChunk(pos, tileX, tileY, out int chunkX, out int chunkY);

        var area = _host.CurrentAreaLookup;
        string? zone = area?.ZoneText;
        string? subzone = area?.SubzoneText ?? _host.CurrentAreaName;
        int? areaId = area?.CanonicalAreaId ?? area?.RawAreaId;

        // Query flight playlist or single route
        string? routeLabel = null;
        float? flightProgress = null;
        int? segIndex = null;
        int? segTotal = null;

        if (_host.TaxiPlaylist.IsPlaying)
        {
            var status = _host.TaxiPlaylist.GetStatus();
            routeLabel = status.CurrentItem?.DisplayLabel;
            flightProgress = status.SegmentProgressFraction;
            segIndex = status.CurrentIndex + 1;
            segTotal = status.TotalCount;
        }
        else if (_host.WorldScene != null && _host.WorldScene.TaxiActors.ActiveTaxiRideRouteId >= 0)
        {
            int routeId = _host.WorldScene.TaxiActors.ActiveTaxiRideRouteId;
            var route = _host.WorldScene.TaxiActors.GetTaxiRoute(routeId);
            if (route != null)
            {
                string from = _host.WorldScene.TaxiActors.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
                string to = _host.WorldScene.TaxiActors.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
                routeLabel = $"{from} \u2192 {to}";

                if (_host.WorldScene.TaxiActors.TryGetTaxiRouteProgress(routeId, out float dist, out float total) && total > 0f)
                    flightProgress = Math.Clamp(dist / total, 0f, 1f);
            }
        }

        // Proximity check for closest taxi node
        string? nearestNodeName = null;
        float? nearestNodeDist = null;
        if (Config.ShowLandmarkCallouts && _host.WorldScene?.TaxiActors.TaxiLoader is TaxiPathLoader loader)
        {
            float bestDistSq = Config.ProximityDistanceYards * Config.ProximityDistanceYards;
            TaxiPathLoader.TaxiNode? bestNode = null;

            foreach (var node in loader.Nodes)
            {
                float distSq = Vector3.DistanceSquared(pos, node.Position);
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    bestNode = node;
                }
            }

            if (bestNode != null)
            {
                nearestNodeName = bestNode.Name;
                nearestNodeDist = MathF.Sqrt(bestDistSq);
            }
        }

        // Performance & Frame Pacing
        float fps = ImGui.GetIO().Framerate;
        float frameTimeMs = _lastFrameDt * 1000f;
        bool isHitch = frameTimeMs >= Config.HitchThresholdMs;
        float gcPause = 0f;
        int tDraws = 0, wmoDraws = 0, m2Draws = 0;
        int tChunksRendered = 0, tChunksCulled = 0;
        int wmoBatches = 0, wmoInstances = 0;
        int m2Instanced = 0, m2Unbatched = 0;
        int liquidMeshes = 0;
        int loadedTiles = 0;
        int cacheHits = 0, fileCount = 0, defLoads = 0;

        if (_host.TerrainManager is { } tm)
        {
            loadedTiles = tm.LoadedTileCount;
            if (tm.Renderer is { } tr)
                tDraws = tr.LastFrameDrawCalls;
            if (tm.LiquidRenderer is { } lr)
                liquidMeshes = lr.LastVisibleTerrainMeshCount;
        }
        else if (_host.VlmTerrainManager is { } vlm)
        {
            loadedTiles = vlm.LoadedTileCount;
            if (vlm.Renderer is { } tr)
                tDraws = tr.LastFrameDrawCalls;
            if (vlm.LiquidRenderer is { } lr)
                liquidMeshes = lr.LastVisibleTerrainMeshCount;
        }

        if (_host.WorldScene != null)
        {
            var rStats = _host.WorldScene.LastRenderFrameStats;
            gcPause = (float)rStats.GcPauseMs;
            wmoDraws = rStats.WmoDrawCallCount;
            m2Draws = rStats.OpaqueModelSubmission.DrawCalls + rStats.TransparentModelSubmission.DrawCalls;
            tChunksRendered = rStats.TerrainChunksRendered;
            tChunksCulled = rStats.TerrainChunksCulled;
            wmoBatches = rStats.WmoBatchDrawCallCount;
            wmoInstances = rStats.WmoOpaqueBatchInstanceCount;
            m2Instanced = rStats.OpaqueModelSubmission.Instanced;
            m2Unbatched = rStats.OpaqueModelSubmission.Unbatched + rStats.TransparentModelSubmission.Unbatched;

            var aStats = _host.WorldScene.Assets.GetReadStats();
            cacheHits = (int)aStats.FileCacheHits;
            fileCount = (int)aStats.FileCacheCount;
            defLoads = (int)_host.WorldScene.Assets.LoadBudget.OversizedAdmissionCount;
        }

        int totalDraws = tDraws + wmoDraws + m2Draws;
        int detailFlora = _host.GroundEffects?.ActiveDetailDoodadCount ?? 0;
        float managedMem = (float)(GC.GetTotalMemory(false) / (1024.0 * 1024.0));
        float processMem = (float)(Environment.WorkingSet / (1024.0 * 1024.0));

        string? era = !string.IsNullOrWhiteSpace(_host.LoadedFileName)
            ? Path.GetExtension(_host.LoadedFileName).ToUpperInvariant().TrimStart('.')
            : "WDT";

        return new ShowreelTelemetrySnapshot(
            Position: pos,
            YawDegrees: yaw,
            PitchDegrees: pitch,
            CompassDirection: compass,
            MapId: _host.CurrentMapId,
            MapName: _host.LoadedFileName,
            AdtTileX: tileX,
            AdtTileY: tileY,
            McnkChunkX: chunkX,
            McnkChunkY: chunkY,
            ZoneName: zone,
            SubzoneName: subzone,
            AreaTableId: areaId,
            ActiveFlightRouteLabel: routeLabel,
            FlightProgressFraction: flightProgress,
            FlightSegmentIndex: segIndex,
            FlightSegmentTotal: segTotal,
            ApproachingLandmarkName: nearestNodeName,
            ApproachingLandmarkDistanceYards: nearestNodeDist,
            // Diagnostic & Pipeline
            Fps: fps,
            FrameTimeMs: frameTimeMs,
            IsHitch: isHitch,
            RecentHitchCount: _sessionHitchCount,
            WorstFrameTimeMs: _worstFrameTimeMs,
            LastHitchStage: _lastHitchStage,
            LastHitchDurationMs: _lastHitchDurationMs,
            GcPauseMs: gcPause,
            ManagedMemoryMb: managedMem,
            ProcessWorkingSetMb: processMem,
            TotalDrawCalls: totalDraws,
            TerrainDrawCalls: tDraws,
            TerrainChunksRendered: tChunksRendered,
            TerrainChunksCulled: tChunksCulled,
            WmoBatchCount: wmoBatches,
            WmoInstanceCount: wmoInstances,
            M2InstancedCount: m2Instanced,
            M2UnbatchedCount: m2Unbatched,
            DetailDoodadCount: detailFlora,
            LiquidMeshCount: liquidMeshes,
            LoadedTilesCount: loadedTiles,
            FileCacheHits: cacheHits,
            FileCacheCount: fileCount,
            DeferredLoadsCount: defLoads,
            DataSourceEra: era);
    }

    /// <summary>
    /// Renders the complete dynamic showreel overlay onto the foreground draw list.
    /// </summary>
    public void Draw()
    {
        if (!ShouldDraw())
            return;

        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        if (displaySize.X <= 0 || displaySize.Y <= 0)
            return;

        ImDrawListPtr drawList = ImGui.GetForegroundDrawList();
        ShowreelTelemetrySnapshot snap = CaptureTelemetrySnapshot();

        // 1. Zone Entry Banner (Large & Centered in Frame)
        if (Config.ShowZoneBanners && _bannerActive && !string.IsNullOrWhiteSpace(_activeBannerZone))
        {
            DrawZoneBanner(drawList, displaySize);
        }

        // 2. Live Telemetry HUD (Bottom Left)
        if (Config.ShowLiveTelemetry)
        {
            DrawTelemetryHud(drawList, displaySize, in snap);
        }

        // 3. Landmark Proximity & Technical Showcase Badges (Top Right)
        if (Config.ShowLandmarkCallouts || Config.ShowEngineBadges)
        {
            DrawLandmarkAndBadgeHud(drawList, displaySize, in snap);
        }
    }

    private void DrawZoneBanner(ImDrawListPtr drawList, Vector2 displaySize)
    {
        float alpha = 1.0f;
        if (_bannerElapsedSeconds < 0.5f)
            alpha = Math.Clamp(_bannerElapsedSeconds / 0.5f, 0f, 1f);
        else if (_bannerElapsedSeconds > Config.BannerHoldSeconds)
        {
            float fadeProgress = (_bannerElapsedSeconds - Config.BannerHoldSeconds) / Math.Max(0.1f, Config.BannerFadeSeconds);
            alpha = Math.Clamp(1.0f - fadeProgress, 0f, 1f);
        }

        uint bgAlpha = (uint)(alpha * 225);
        uint borderAlpha = (uint)(alpha * 255);
        uint textAlpha = (uint)(alpha * 255);

        uint bgColor = (bgAlpha << 24) | 0x00100C16;
        uint borderColor = (borderAlpha << 24) | 0x00D4AF37; // Rich WoW gold
        uint textTitleColor = (textAlpha << 24) | 0x00FFFFFF;
        uint textSubColor = (textAlpha << 24) | 0x00E8D9B8;

        ImFontPtr font = ImGui.GetFont();
        float baseFontSize = ImGui.GetFontSize();
        float titleFontSize = MathF.Round(baseFontSize * 2.15f);
        float subFontSize = MathF.Round(baseFontSize * 1.35f);
        float badgeFontSize = MathF.Round(baseFontSize * 0.95f);

        bool hasSubzone = !string.IsNullOrWhiteSpace(_activeBannerSubzone);
        float bannerWidth = Math.Clamp(displaySize.X * 0.52f, 520f, 780f);
        float bannerHeight = hasSubzone ? 132f : 104f;

        // Positioned in the center of the frame (with subtle upward bias for cinematic horizon visibility)
        Vector2 bannerMin = new((displaySize.X - bannerWidth) * 0.5f, (displaySize.Y - bannerHeight) * 0.42f);
        Vector2 bannerMax = bannerMin + new Vector2(bannerWidth, bannerHeight);

        // Backdrop panel with double-layered WoW gold border & rounded corners
        drawList.AddRectFilled(bannerMin, bannerMax, bgColor, 14f);
        drawList.AddRect(bannerMin, bannerMax, borderColor, 14f, ImDrawFlags.None, 2.0f);
        drawList.AddRect(bannerMin + new Vector2(3f, 3f), bannerMax - new Vector2(3f, 3f), (borderAlpha / 3 << 24) | 0x00D4AF37, 12f, ImDrawFlags.None, 1.0f);

        // Decorative horizontal divider line
        float dividerY = bannerMin.Y + (hasSubzone ? 88f : 64f);
        drawList.AddLine(
            new Vector2(bannerMin.X + 48f, dividerY),
            new Vector2(bannerMax.X - 48f, dividerY),
            (borderAlpha / 2 << 24) | 0x00D4AF37,
            1.2f);

        // Centered Main Zone Title
        string title = _activeBannerZone!.ToUpperInvariant();
        float titleWidth = ImGui.CalcTextSize(title).X * (titleFontSize / baseFontSize);
        float titleX = bannerMin.X + (bannerWidth - titleWidth) * 0.5f;
        float titleY = bannerMin.Y + 16f;

        // Deep drop-shadow for contrast against terrain or bright sky
        drawList.AddText(font, titleFontSize, new Vector2(titleX + 2f, titleY + 2f), (textAlpha * 220 / 255 << 24) | 0x00000000, title);
        drawList.AddText(font, titleFontSize, new Vector2(titleX, titleY), textTitleColor, title);

        // Centered Subzone Title (if present)
        if (hasSubzone)
        {
            float subWidth = ImGui.CalcTextSize(_activeBannerSubzone!).X * (subFontSize / baseFontSize);
            float subX = bannerMin.X + (bannerWidth - subWidth) * 0.5f;
            float subY = titleY + titleFontSize + 6f;

            drawList.AddText(font, subFontSize, new Vector2(subX + 1.5f, subY + 1.5f), (textAlpha * 200 / 255 << 24) | 0x00000000, _activeBannerSubzone!);
            drawList.AddText(font, subFontSize, new Vector2(subX, subY), textSubColor, _activeBannerSubzone!);
        }

        // Centered Area Badge / Discovery Tag
        string areaBadge = _activeBannerAreaId > 0 ? $"AREA #{_activeBannerAreaId} • DISCOVERY" : "DISCOVERY";
        float badgeWidth = ImGui.CalcTextSize(areaBadge).X * (badgeFontSize / baseFontSize);
        float badgeX = bannerMin.X + (bannerWidth - badgeWidth) * 0.5f;
        float badgeY = bannerMax.Y - badgeFontSize - 10f;

        drawList.AddText(font, badgeFontSize, new Vector2(badgeX, badgeY), (textAlpha * 190 / 255 << 24) | 0x00D4AF37, areaBadge);
    }

    private void DrawTelemetryHud(ImDrawListPtr drawList, Vector2 displaySize, in ShowreelTelemetrySnapshot snap)
    {
        const float margin = 28f;
        float cardWidth = 460f;

        bool hasRoute = snap.ActiveFlightRouteLabel != null;
        bool hasPerf = Config.ShowPerformanceTelemetry;
        bool hasPipeline = Config.ShowPipelineTelemetry;
        bool hasHitchAlert = Config.ShowHitchAlerts && _hitchAlertTimerSeconds > 0f && snap.LastHitchDurationMs > 0f;

        float lineSpacing = 22f;
        int lineCount = 3; // POS, DIR, ZONE
        if (hasRoute) lineCount++;
        if (hasPerf) lineCount++;
        if (hasHitchAlert) lineCount++;
        if (hasPipeline) lineCount += 2; // Geometry Pipeline, I/O & Cache

        float cardHeight = 18f + (lineCount * lineSpacing);

        // If a feature tour beat presentation is currently rendering at the bottom left,
        // stack the telemetry card neatly above it so they do not collide.
        float bottomMargin = margin;
        if (_host.RecordingCoordinator.ActiveTourPresentation != null)
        {
            bottomMargin = 36f + 104f + 12f;
        }

        Vector2 cardMin = new(margin, displaySize.Y - bottomMargin - cardHeight);
        Vector2 cardMax = cardMin + new Vector2(cardWidth, cardHeight);

        // Sleek dark frosted glass
        drawList.AddRectFilled(cardMin, cardMax, 0xDC0D111A, 8f);
        drawList.AddRect(cardMin, cardMax, 0x8848729A, 8f, ImDrawFlags.None, 1.2f);

        float curY = cardMin.Y + 10f;
        float startX = cardMin.X + 14f;

        // 1. Hitch Alert Callout Banner (if frame spiked above threshold)
        if (hasHitchAlert)
        {
            float alertFraction = _hitchAlertTimerSeconds / HitchAlertHoldSeconds;
            uint alertAlpha = (uint)(Math.Clamp(alertFraction * 2.0f, 0.45f, 1.0f) * 255);
            uint bannerBg = (alertAlpha * 185 / 255 << 24) | 0x00141470; // glowing crimson
            uint bannerBorder = (alertAlpha << 24) | 0x004050FF; // vibrant red/amber
            uint bannerText = (alertAlpha << 24) | 0x00FFFFFF;

            Vector2 bMin = new(startX - 4f, curY - 2f);
            Vector2 bMax = new(cardMax.X - 10f, curY + 18f);
            drawList.AddRectFilled(bMin, bMax, bannerBg, 4f);
            drawList.AddRect(bMin, bMax, bannerBorder, 4f, ImDrawFlags.None, 1.0f);

            string hitchText = $"[!] HITCH SPIKE: +{snap.LastHitchDurationMs:F1} ms  (STAGE: {snap.LastHitchStage ?? "Render"})";
            drawList.AddText(new Vector2(startX + 6f, curY), bannerText, hitchText);
            curY += lineSpacing;
        }

        // 2. Performance & Memory line
        if (hasPerf)
        {
            uint fpsColor = snap.Fps >= 55f ? 0xFF75E095 : (snap.Fps >= 30f ? 0xFFFFD060 : 0xFFFF6565);
            string fpsStr = $"FPS  {snap.Fps,5:F1} ({snap.FrameTimeMs,4:F1}ms)";
            drawList.AddText(new Vector2(startX, curY), fpsColor, fpsStr);

            string memStr = $" |  RAM: {ShowreelTelemetryMath.FormatMemoryMb(snap.ProcessWorkingSetMb)}  |  DRAWS: {snap.TotalDrawCalls}";
            float fpsWidth = ImGui.CalcTextSize(fpsStr).X;
            drawList.AddText(new Vector2(startX + fpsWidth, curY), 0xFFE0E0E0, memStr);
            curY += lineSpacing;
        }

        // 3. Pipeline Telemetry lines
        if (hasPipeline)
        {
            int totalM2 = snap.M2InstancedCount + snap.M2UnbatchedCount;
            int instPct = totalM2 > 0 ? (int)MathF.Round((float)snap.M2InstancedCount / totalM2 * 100f) : 100;
            string pipeStr = $"PIPE M2: {ShowreelTelemetryMath.FormatCompactNumber(totalM2)} ({instPct}% inst) | FLORA: {ShowreelTelemetryMath.FormatCompactNumber(snap.DetailDoodadCount)} | WMO: {snap.WmoBatchCount} bth | LIQ: {snap.LiquidMeshCount}";
            drawList.AddText(new Vector2(startX, curY), 0xFF65D5FF, pipeStr);
            curY += lineSpacing;

            string ioStr = $"I/O  Tiles: {snap.LoadedTilesCount} | Cache: {ShowreelTelemetryMath.FormatCompactNumber(snap.FileCacheHits)} hits ({snap.FileCacheCount} files) | Def: {snap.DeferredLoadsCount}";
            drawList.AddText(new Vector2(startX, curY), 0xFFD8B0FF, ioStr);
            curY += lineSpacing;
        }

        // 4. Position & Orientation line
        string posText = $"POS  X: {snap.Position.X,8:F1}   Y: {snap.Position.Y,8:F1}   Z: {snap.Position.Z,6:F1}";
        drawList.AddText(new Vector2(startX, curY), 0xFFFFFFFF, posText);
        curY += lineSpacing;

        // 5. Compass & Tile line
        string compassTile = $"DIR  {snap.YawDegrees,5:F1}° {snap.CompassDirection,-2}  |  ADT [{snap.AdtTileX,2},{snap.AdtTileY,2}]  MCNK [{snap.McnkChunkX,2},{snap.McnkChunkY,2}]";
        drawList.AddText(new Vector2(startX, curY), 0xFFB0D0F0, compassTile);
        curY += lineSpacing;

        // 6. Location line
        string locText = !string.IsNullOrWhiteSpace(snap.ZoneName)
            ? $"ZONE {snap.ZoneName}  ({snap.SubzoneName ?? "General"})"
            : $"MAP  {snap.MapName ?? "World"} (ID: {snap.MapId ?? 0})";
        drawList.AddText(new Vector2(startX, curY), 0xFF8AE0A0, locText);
        curY += lineSpacing;

        // 7. Active Flight line (if riding taxi)
        if (hasRoute)
        {
            float frac = snap.FlightProgressFraction ?? 0f;
            string segInfo = snap.FlightSegmentIndex != null ? $" [{snap.FlightSegmentIndex}/{snap.FlightSegmentTotal}]" : "";
            string flightText = $"FLY  {snap.ActiveFlightRouteLabel}{segInfo} • {frac:P0}";
            drawList.AddText(new Vector2(startX, curY), 0xFFFFCC66, flightText);
        }

        // 8. Expanded Diagnostics Side Panel (if toggled)
        if (Config.ExpandedDiagnostics && _host.WorldScene != null)
        {
            DrawExpandedDiagnostics(drawList, new Vector2(cardMax.X + 10f, cardMin.Y), cardHeight);
        }
    }

    private void DrawExpandedDiagnostics(ImDrawListPtr drawList, Vector2 pos, float height)
    {
        var stats = _host.WorldScene?.LastRenderFrameStats;
        if (stats == null) return;

        float width = 250f;
        Vector2 panelMin = pos;
        Vector2 panelMax = panelMin + new Vector2(width, height);

        drawList.AddRectFilled(panelMin, panelMax, 0xDC0D111A, 8f);
        drawList.AddRect(panelMin, panelMax, 0x889A7248, 8f, ImDrawFlags.None, 1.2f);

        float curY = panelMin.Y + 10f;
        float startX = panelMin.X + 12f;

        drawList.AddText(new Vector2(startX, curY), 0xFFFFD700, "STAGE TIMINGS (MS)");
        curY += 20f;

        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"Terrain:      {stats.Value.Terrain.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"WMO Sub:      {stats.Value.WmoSubmission.DurationMs + stats.Value.WmoTransparentSubmission.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"M2 Sub:       {stats.Value.MdxOpaqueSubmission.DurationMs + stats.Value.MdxTransparentSubmission.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"M2 Vis/Anim:  {stats.Value.MdxVisibility.DurationMs + stats.Value.MdxAnimation.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"Liquid:       {stats.Value.Liquid.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"Deferred I/O: {stats.Value.DeferredAssetLoads.DurationMs,5:F2} ms");
        curY += 18f;
        drawList.AddText(new Vector2(startX, curY), 0xFFE0E0E0, $"GC Pause:     {stats.Value.GcPauseMs,5:F2} ms");
    }

    private void DrawLandmarkAndBadgeHud(ImDrawListPtr drawList, Vector2 displaySize, in ShowreelTelemetrySnapshot snap)
    {
        const float margin = 28f;
        float cardWidth = 380f;
        float cardHeight = 84f;
        Vector2 cardMin = new(displaySize.X - margin - cardWidth, margin);
        Vector2 cardMax = cardMin + new Vector2(cardWidth, cardHeight);

        drawList.AddRectFilled(cardMin, cardMax, 0xDC0D111A, 8f);
        drawList.AddRect(cardMin, cardMax, 0x889A7248, 8f, ImDrawFlags.None, 1.2f);

        // 1. Landmark discovery callout if approaching
        if (Config.ShowLandmarkCallouts && snap.ApproachingLandmarkName != null && snap.ApproachingLandmarkDistanceYards != null)
        {
            drawList.AddText(cardMin + new Vector2(14f, 10f), 0xFFFFD700, "APPROACHING FLIGHT NODE");
            string landmarkText = $"{snap.ApproachingLandmarkName} ({snap.ApproachingLandmarkDistanceYards:F0} yd)";
            drawList.AddText(cardMin + new Vector2(14f, 30f), 0xFFFFFFFF, landmarkText);
        }
        else
        {
            // 2. Rotating technical showcase badge
            var badge = ShowcaseBadges[_badgeIndex];
            drawList.AddText(cardMin + new Vector2(14f, 10f), 0xFF85C8FF, badge.Category.ToUpperInvariant());
            drawList.AddText(cardMin + new Vector2(14f, 30f), 0xFFE0E0E0, badge.Description);
        }

        // Tool watermark & branding badge
        drawList.AddText(cardMin + new Vector2(14f, 56f), 0xFF888888, "wow-viewer • Format Inspection & Exploration Engine");
    }
}
