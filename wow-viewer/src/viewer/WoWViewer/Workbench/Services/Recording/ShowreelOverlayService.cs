using System;
using System.Collections.Generic;
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
    /// Per-frame update for area transitions, badge timers, and banner animations.
    /// </summary>
    public void Update(double dt)
    {
        if (!ShouldDraw())
            return;

        float delta = (float)Math.Max(0.0, dt);

        // 1. Zone transition detection
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

        // 2. Engine showcase badge cycling
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
    /// Captures a telemetry snapshot of the current frame.
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
            ApproachingLandmarkDistanceYards: nearestNodeDist);
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
        float cardWidth = 370f;
        float cardHeight = snap.ActiveFlightRouteLabel != null ? 104f : 84f;

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

        // Position & Orientation line
        string posText = $"POS   X: {snap.Position.X,8:F1}   Y: {snap.Position.Y,8:F1}   Z: {snap.Position.Z,6:F1}";
        drawList.AddText(cardMin + new Vector2(14f, 10f), 0xFFFFFFFF, posText);

        // Compass & Tile line
        string compassTile = $"DIR   {snap.YawDegrees,5:F1}° {snap.CompassDirection,-2}  |  ADT [{snap.AdtTileX,2},{snap.AdtTileY,2}]  MCNK [{snap.McnkChunkX,2},{snap.McnkChunkY,2}]";
        drawList.AddText(cardMin + new Vector2(14f, 32f), 0xFFB0D0F0, compassTile);

        // Location line
        string locText = !string.IsNullOrWhiteSpace(snap.ZoneName)
            ? $"ZONE  {snap.ZoneName}  ({snap.SubzoneName ?? "General"})"
            : $"MAP   {snap.MapName ?? "World"} (ID: {snap.MapId ?? 0})";
        drawList.AddText(cardMin + new Vector2(14f, 54f), 0xFF8AE0A0, locText);

        // Active Flight line (if riding taxi)
        if (snap.ActiveFlightRouteLabel != null)
        {
            float frac = snap.FlightProgressFraction ?? 0f;
            string segInfo = snap.FlightSegmentIndex != null ? $" [{snap.FlightSegmentIndex}/{snap.FlightSegmentTotal}]" : "";
            string flightText = $"FLY   {snap.ActiveFlightRouteLabel}{segInfo} • {frac:P0}";
            drawList.AddText(cardMin + new Vector2(14f, 76f), 0xFFFFCC66, flightText);
        }
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
