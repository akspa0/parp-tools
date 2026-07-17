using System.Numerics;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.Tests;

public sealed class TerrainSolarDirectionTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(0.25f)]
    [InlineData(0.5f)]
    [InlineData(0.75f)]
    public void Evaluate_NeverCollapsesToAStraightOverheadSunAtNoonOrMidnight(float gameTime)
    {
        Vector3 direction = TerrainSolarDirection.Evaluate(gameTime);

        // A prior implementation swept the horizontal bearing with time of day and it collapsed to
        // zero exactly at solar noon/midnight (gameTime 0.5/0.0), producing a straight-overhead sun
        // with no shadow direction at all. The traced native ray holds a constant azimuth, so the
        // horizontal component must stay meaningfully non-zero at every sampled time.
        float horizontalMagnitude = new Vector2(direction.X, direction.Y).Length();
        Assert.True(horizontalMagnitude > 0.1f);
    }

    [Fact]
    public void Evaluate_KeepsAFixedNorthWestBearingAcrossTheDay()
    {
        Vector3 dawn = TerrainSolarDirection.Evaluate(0.25f);
        Vector3 noon = TerrainSolarDirection.Evaluate(0.5f);
        Vector3 dusk = TerrainSolarDirection.Evaluate(0.75f);

        // Raw MCNR/MCVT world axes are +X = North, +Y = West, +Z = Up. The traced native ray holds a
        // constant azimuth (theta = 225 degrees). Normalizing the full 3D vector redistributes
        // magnitude between the horizontal and vertical parts as elevation changes, so X and Y
        // shrink together with time of day -- but their *ratio* (the azimuth/bearing) must stay
        // fixed at a 45-degree north-west split.
        Assert.True(dawn.X > 0f);
        Assert.True(dawn.Y > 0f);
        Assert.Equal(dawn.X, dawn.Y, 5);
        Assert.Equal(dawn.X / dawn.Y, noon.X / noon.Y, 4);
        Assert.Equal(dawn.X / dawn.Y, dusk.X / dusk.Y, 4);
    }

    [Fact]
    public void Evaluate_PeaksElevationNearSolarNoon()
    {
        Vector3 noon = TerrainSolarDirection.Evaluate(0.5f);
        Vector3 midMorning = TerrainSolarDirection.Evaluate(0.4f);
        Vector3 midnight = TerrainSolarDirection.Evaluate(0f);

        Assert.True(noon.Z > midMorning.Z);
        Assert.True(midMorning.Z > midnight.Z);
    }
}
