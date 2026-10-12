using System;
using System.Numerics;

namespace WowViewer.Core.Runtime.PromoVideo;

/// <summary>
/// Configuration toggles for the marketing showreel telemetry HUD and overlays.
/// </summary>
public sealed class ShowreelOverlayConfig
{
    public bool EnableOverlay { get; set; } = false;
    public bool PreviewInViewport { get; set; } = false;
    public bool ShowLiveTelemetry { get; set; } = true;
    public bool ShowPerformanceTelemetry { get; set; } = true;
    public bool ShowPipelineTelemetry { get; set; } = true;
    public bool ShowHitchAlerts { get; set; } = true;
    public float HitchThresholdMs { get; set; } = 33.3f;
    public bool ExpandedDiagnostics { get; set; } = false;
    public bool ShowZoneBanners { get; set; } = true;
    public bool ShowLandmarkCallouts { get; set; } = true;
    public bool ShowEngineBadges { get; set; } = true;
    public float BannerHoldSeconds { get; set; } = 3.5f;
    public float BannerFadeSeconds { get; set; } = 0.6f;
    public float ProximityDistanceYards { get; set; } = 350f;
}

/// <summary>
/// Severity classification for frame time spikes.
/// </summary>
public enum HitchSeverity
{
    None = 0,
    Minor = 1,
    Severe = 2,
    Critical = 3,
}

/// <summary>
/// Immutable snapshot of live world, pipeline, and diagnostic telemetry captured for a showreel frame or beat.
/// </summary>
public readonly record struct ShowreelTelemetrySnapshot(
    Vector3 Position,
    float YawDegrees,
    float PitchDegrees,
    string CompassDirection,
    int? MapId,
    string? MapName,
    int? AdtTileX,
    int? AdtTileY,
    int? McnkChunkX,
    int? McnkChunkY,
    string? ZoneName,
    string? SubzoneName,
    int? AreaTableId,
    string? ActiveFlightRouteLabel,
    float? FlightProgressFraction,
    int? FlightSegmentIndex,
    int? FlightSegmentTotal,
    string? ApproachingLandmarkName,
    float? ApproachingLandmarkDistanceYards,
    // Performance & Diagnostic Telemetry
    float Fps = 0f,
    float FrameTimeMs = 0f,
    bool IsHitch = false,
    int RecentHitchCount = 0,
    float WorstFrameTimeMs = 0f,
    string? LastHitchStage = null,
    float LastHitchDurationMs = 0f,
    float GcPauseMs = 0f,
    float ManagedMemoryMb = 0f,
    float ProcessWorkingSetMb = 0f,
    // Geometry & Rendering Pipeline Telemetry
    int TotalDrawCalls = 0,
    int TerrainDrawCalls = 0,
    int TerrainChunksRendered = 0,
    int TerrainChunksCulled = 0,
    int WmoBatchCount = 0,
    int WmoInstanceCount = 0,
    int M2InstancedCount = 0,
    int M2UnbatchedCount = 0,
    int DetailDoodadCount = 0,
    int LiquidMeshCount = 0,
    // I/O & Asset Pipeline Telemetry
    int LoadedTilesCount = 0,
    int FileCacheHits = 0,
    int FileCacheCount = 0,
    int DeferredLoadsCount = 0,
    string? DataSourceEra = null);

/// <summary>
/// Mathematical and formatting helpers for showreel telemetry.
/// </summary>
public static class ShowreelTelemetryMath
{
    public const float AdtTileSize = 533.33333f;
    public const float McnkChunkSize = 33.33333f;
    public const float AdtCenterTile = 32.0f;

    /// <summary>
    /// Converts a WoW world position (X, Y) to ADT tile indices (0..63).
    /// </summary>
    public static void WorldToAdtTile(Vector3 position, out int tileX, out int tileY)
    {
        tileX = (int)MathF.Floor(AdtCenterTile - (position.X / AdtTileSize));
        tileY = (int)MathF.Floor(AdtCenterTile - (position.Y / AdtTileSize));
    }

    /// <summary>
    /// Converts a WoW world position (X, Y) to MCNK chunk indices (0..15) within the given ADT tile.
    /// </summary>
    public static void WorldToMcnkChunk(Vector3 position, int tileX, int tileY, out int chunkX, out int chunkY)
    {
        float tileOriginX = (AdtCenterTile - tileX) * AdtTileSize;
        float tileOriginY = (AdtCenterTile - tileY) * AdtTileSize;

        chunkX = Math.Clamp((int)MathF.Floor((tileOriginX - position.X) / McnkChunkSize), 0, 15);
        chunkY = Math.Clamp((int)MathF.Floor((tileOriginY - position.Y) / McnkChunkSize), 0, 15);
    }

    /// <summary>
    /// Normalizes degrees into [0, 360).
    /// </summary>
    public static float NormalizeDegrees(float degrees)
    {
        float norm = degrees % 360f;
        if (norm < 0f)
            norm += 360f;
        return norm;
    }

    /// <summary>
    /// Computes 8-cardinal compass string from camera yaw in degrees.
    /// </summary>
    public static string GetCompassDirection(float yawDegrees)
    {
        float deg = NormalizeDegrees(yawDegrees);

        // 8 sectors of 45 degrees, centered on 0, 45, 90, etc.
        // Sector offset by 22.5 deg:
        int sector = (int)MathF.Floor((deg + 22.5f) / 45f) % 8;
        return sector switch
        {
            0 => "N",
            1 => "NE",
            2 => "E",
            3 => "SE",
            4 => "S",
            5 => "SW",
            6 => "W",
            7 => "NW",
            _ => "N"
        };
    }

    /// <summary>
    /// Formats integer count with compact K/M abbreviations (e.g. 14,250 -> 14.3K).
    /// </summary>
    public static string FormatCompactNumber(int number)
    {
        if (number < 1_000)
            return number.ToString();
        if (number < 1_000_000)
            return (number / 1_000.0f).ToString("0.#") + "K";
        return (number / 1_000_000.0f).ToString("0.##") + "M";
    }

    /// <summary>
    /// Formats memory in megabytes into readable string with MB or GB suffix.
    /// </summary>
    public static string FormatMemoryMb(float mb)
    {
        if (mb >= 1024f)
            return (mb / 1024f).ToString("0.00") + " GB";
        return mb.ToString("0.0") + " MB";
    }

    /// <summary>
    /// Classifies frame time spike severity against configured threshold.
    /// </summary>
    public static HitchSeverity ClassifyHitchSeverity(float frameTimeMs, float thresholdMs)
    {
        if (frameTimeMs < thresholdMs)
            return HitchSeverity.None;
        if (frameTimeMs < thresholdMs * 1.5f)
            return HitchSeverity.Minor;
        if (frameTimeMs < thresholdMs * 3.0f)
            return HitchSeverity.Severe;
        return HitchSeverity.Critical;
    }
}
