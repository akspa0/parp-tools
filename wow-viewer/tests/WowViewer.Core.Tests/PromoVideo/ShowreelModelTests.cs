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
        Assert.True(config.ShowZoneBanners);
        Assert.True(config.ShowLandmarkCallouts);
        Assert.True(config.ShowEngineBadges);
    }
}
