using System.Numerics;
using WowViewer.Core.Runtime.PromoVideo;
using Xunit;

namespace WowViewer.Core.Tests.PromoVideo;

public class ShowreelModelTests
{
    [Theory]
    [InlineData(0f, "N")]
    [InlineData(20f, "N")]
    [InlineData(45f, "NE")]
    [InlineData(90f, "E")]
    [InlineData(135f, "SE")]
    [InlineData(180f, "S")]
    [InlineData(225f, "SW")]
    [InlineData(270f, "W")]
    [InlineData(315f, "NW")]
    [InlineData(350f, "N")]
    [InlineData(-45f, "NW")]
    [InlineData(720f, "N")]
    public void GetCompassDirection_ComputesCorrectOctant(float yawDegrees, string expectedCompass)
    {
        string actual = ShowreelTelemetryMath.GetCompassDirection(yawDegrees);
        Assert.Equal(expectedCompass, actual);
    }

    [Fact]
    public void WorldToAdtTile_ComputesExpectedTileIndices()
    {
        // Origin (0,0) in WoW coordinates corresponds to center boundary between tiles (31,31) and (32,32).
        // A point slightly negative in X and Y falls in tile 32,32:
        Vector3 centerPoint = new Vector3(-10f, -10f, 0f);
        ShowreelTelemetryMath.WorldToAdtTile(centerPoint, out int tileX, out int tileY);
        Assert.Equal(32, tileX);
        Assert.Equal(32, tileY);

        // A point positive in X falls in tile 31:
        Vector3 northPoint = new Vector3(1000f, -10f, 0f);
        ShowreelTelemetryMath.WorldToAdtTile(northPoint, out int tileXNorth, out int tileYNorth);
        Assert.Equal(30, tileXNorth);
        Assert.Equal(32, tileYNorth);
    }

    [Fact]
    public void WorldToMcnkChunk_ClampsWithinTileBounds()
    {
        Vector3 point = new Vector3(0f, 0f, 0f);
        ShowreelTelemetryMath.WorldToAdtTile(point, out int tileX, out int tileY);
        ShowreelTelemetryMath.WorldToMcnkChunk(point, tileX, tileY, out int chunkX, out int chunkY);

        Assert.InRange(chunkX, 0, 15);
        Assert.InRange(chunkY, 0, 15);
    }

    [Fact]
    public void ShowreelTelemetrySnapshot_ConstructsWithValidFields()
    {
        var snapshot = new ShowreelTelemetrySnapshot(
            Position: new Vector3(100f, 200f, 50f),
            YawDegrees: 90f,
            PitchDegrees: -15f,
            CompassDirection: "E",
            MapId: 0,
            MapName: "Azeroth",
            AdtTileX: 31,
            AdtTileY: 31,
            McnkChunkX: 5,
            McnkChunkY: 8,
            ZoneName: "Elwynn Forest",
            SubzoneName: "Goldshire",
            AreaTableId: 12,
            ActiveFlightRouteLabel: "Stormwind -> Sentinel Hill",
            FlightProgressFraction: 0.45f,
            FlightSegmentIndex: 1,
            FlightSegmentTotal: 3,
            ApproachingLandmarkName: "Sentinel Hill Flight Master",
            ApproachingLandmarkDistanceYards: 120.5f);

        Assert.Equal("Elwynn Forest", snapshot.ZoneName);
        Assert.Equal("Goldshire", snapshot.SubzoneName);
        Assert.Equal("E", snapshot.CompassDirection);
        Assert.Equal(0.45f, snapshot.FlightProgressFraction);
        Assert.Equal(1, snapshot.FlightSegmentIndex);
        Assert.Equal(3, snapshot.FlightSegmentTotal);
        Assert.Equal("Sentinel Hill Flight Master", snapshot.ApproachingLandmarkName);
    }

    [Fact]
    public void ShowreelOverlayConfig_DefaultsToDisabledForNormalViewing()
    {
        var config = new ShowreelOverlayConfig();
        Assert.False(config.EnableOverlay, "Showreel HUD must not be enabled by default during normal viewing.");
        Assert.False(config.PreviewInViewport, "Preview in viewport must not be enabled by default.");
        Assert.True(config.ShowLiveTelemetry);
        Assert.True(config.ShowPerformanceTelemetry);
        Assert.True(config.ShowPipelineTelemetry);
        Assert.True(config.ShowHitchAlerts);
        Assert.Equal(33.3f, config.HitchThresholdMs);
        Assert.False(config.ExpandedDiagnostics);
        Assert.True(config.ShowZoneBanners);
        Assert.True(config.ShowLandmarkCallouts);
        Assert.True(config.ShowEngineBadges);
    }

    [Fact]
    public void ShowreelTelemetrySnapshot_IncludesDiagnosticAndPipelineMetrics()
    {
        var snapshot = new ShowreelTelemetrySnapshot(
            Position: Vector3.Zero,
            YawDegrees: 0f,
            PitchDegrees: 0f,
            CompassDirection: "N",
            MapId: 1,
            MapName: "Kalimdor",
            AdtTileX: 32,
            AdtTileY: 32,
            McnkChunkX: 0,
            McnkChunkY: 0,
            ZoneName: "Durotar",
            SubzoneName: "Valley of Trials",
            AreaTableId: 14,
            ActiveFlightRouteLabel: null,
            FlightProgressFraction: null,
            FlightSegmentIndex: null,
            FlightSegmentTotal: null,
            ApproachingLandmarkName: null,
            ApproachingLandmarkDistanceYards: null,
            Fps: 59.8f,
            FrameTimeMs: 16.7f,
            IsHitch: true,
            RecentHitchCount: 3,
            WorstFrameTimeMs: 48.2f,
            LastHitchStage: "DeferredAssetLoads",
            LastHitchDurationMs: 48.2f,
            GcPauseMs: 0.5f,
            ManagedMemoryMb: 245.5f,
            ProcessWorkingSetMb: 612.0f,
            TotalDrawCalls: 142,
            TerrainDrawCalls: 28,
            TerrainChunksRendered: 16,
            TerrainChunksCulled: 240,
            WmoBatchCount: 34,
            WmoInstanceCount: 6,
            M2InstancedCount: 1250,
            M2UnbatchedCount: 42,
            DetailDoodadCount: 8400,
            LiquidMeshCount: 8,
            LoadedTilesCount: 9,
            FileCacheHits: 1204,
            FileCacheCount: 512,
            DeferredLoadsCount: 2,
            DataSourceEra: "1.12.1 MPQ");

        Assert.Equal(59.8f, snapshot.Fps);
        Assert.True(snapshot.IsHitch);
        Assert.Equal(3, snapshot.RecentHitchCount);
        Assert.Equal("DeferredAssetLoads", snapshot.LastHitchStage);
        Assert.Equal(142, snapshot.TotalDrawCalls);
        Assert.Equal(8400, snapshot.DetailDoodadCount);
        Assert.Equal(1250, snapshot.M2InstancedCount);
        Assert.Equal("1.12.1 MPQ", snapshot.DataSourceEra);
    }

    [Theory]
    [InlineData(42, "42")]
    [InlineData(999, "999")]
    [InlineData(1000, "1K")]
    [InlineData(14500, "14.5K")]
    [InlineData(1200000, "1.2M")]
    public void FormatCompactNumber_FormatsExpectedAbbreviations(int number, string expected)
    {
        string actual = ShowreelTelemetryMath.FormatCompactNumber(number);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(256.4f, "256.4 MB")]
    [InlineData(1024f, "1.00 GB")]
    [InlineData(2048.5f, "2.00 GB")]
    public void FormatMemoryMb_FormatsExpectedUnits(float mb, string expected)
    {
        string actual = ShowreelTelemetryMath.FormatMemoryMb(mb);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(16.0f, 33.3f, HitchSeverity.None)]
    [InlineData(40.0f, 33.3f, HitchSeverity.Minor)]
    [InlineData(65.0f, 33.3f, HitchSeverity.Severe)]
    [InlineData(120.0f, 33.3f, HitchSeverity.Critical)]
    public void ClassifyHitchSeverity_ClassifiesCorrectly(float frameTimeMs, float thresholdMs, HitchSeverity expected)
    {
        HitchSeverity actual = ShowreelTelemetryMath.ClassifyHitchSeverity(frameTimeMs, thresholdMs);
        Assert.Equal(expected, actual);
    }
}

