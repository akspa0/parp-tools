using System.Numerics;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class ValidationCaptureCameraSolverTests
{
    [Fact]
    public void ComputeTileCenter_MatchesTerrainMeshTileAxisContract()
    {
        Vector2 center = ValidationCaptureCameraSolver.ComputeTileCenter(
            tileX: 30,
            tileY: 48,
            mapOrigin: 17066.666f,
            tileWorldSize: 533.33333f);

        // Terrain mesh world X is derived from ADT tile Y; world Y from ADT tile X.
        Assert.Equal(-8800f, center.X, 3);
        Assert.Equal(800f, center.Y, 3);
    }

    [Fact]
    public void SolveTopDown_WideAspect_KeepsDesiredVerticalSpan()
    {
        ValidationCaptureCameraFrame frame = ValidationCaptureCameraSolver.SolveTopDown(new ValidationCaptureCameraInput(
            TileX: 30,
            TileY: 48,
            AspectRatio: 2.0f,
            GroundHeight: 128f,
            MapOrigin: 17066.666f,
            TileWorldSize: 533.33333f,
            DesiredSpan: 533.33333f,
            EyeHeightOffset: 2048f,
            NearPlane: 0.1f,
            FarPlane: 20000f,
            Up: Vector3.UnitX));

        Assert.Equal(1066.6666f, frame.WorldSpanX, 3);
        Assert.Equal(533.3333f, frame.WorldSpanY, 3);
        Assert.Equal(2176f, frame.Eye.Z, 3);
        Assert.Equal(128f, frame.Target.Z, 3);
        Assert.Equal(Vector3.UnitX, frame.Up);
    }

    [Fact]
    public void SolveTopDown_TallAspect_KeepsDesiredHorizontalSpan()
    {
        ValidationCaptureCameraFrame frame = ValidationCaptureCameraSolver.SolveTopDown(new ValidationCaptureCameraInput(
            TileX: 30,
            TileY: 48,
            AspectRatio: 0.5f,
            GroundHeight: 0f,
            MapOrigin: 17066.666f,
            TileWorldSize: 533.33333f,
            DesiredSpan: 533.33333f,
            EyeHeightOffset: 2048f,
            NearPlane: 0.1f,
            FarPlane: 20000f,
            Up: Vector3.Zero));

        Assert.Equal(533.3333f, frame.WorldSpanX, 3);
        Assert.Equal(1066.6666f, frame.WorldSpanY, 3);
        Assert.Equal(Vector3.UnitX, frame.Up);
    }
}
