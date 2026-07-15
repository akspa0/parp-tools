using System.Numerics;
using WowViewer.Core.Renderer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class TerrainCaptureViewTests
{
    [Fact]
    public void CreateTopDown_FramesExactlyOneTileWithDatasetAxisOrientation()
    {
        TerrainCaptureCamera camera = TerrainCaptureView.CreateTopDown(30, 48, -20f, 180f);
        float expectedX = TerrainConstants.MapOrigin - (48.5f * TerrainConstants.TileSize);
        float expectedY = TerrainConstants.MapOrigin - (30.5f * TerrainConstants.TileSize);

        Assert.Equal(expectedX, camera.Position.X, 3);
        Assert.Equal(expectedY, camera.Position.Y, 3);
        Assert.Equal(1204f, camera.Position.Z, 3);

        Vector3 viewOrigin = Vector3.Transform(camera.Position, camera.View);
        Vector3 tileXStep = Vector3.Transform(camera.Position - Vector3.UnitY, camera.View) - viewOrigin;
        Vector3 tileYStep = Vector3.Transform(camera.Position - Vector3.UnitX, camera.View) - viewOrigin;
        Assert.True(tileXStep.X > 0f);
        Assert.True(tileYStep.Y < 0f);
        Assert.Equal(2f / TerrainConstants.TileSize, camera.Projection.M11, 6);
        Assert.Equal(2f / TerrainConstants.TileSize, camera.Projection.M22, 6);
    }

    [Fact]
    public void CreateTopDown_RejectsInvalidHeightBounds()
    {
        Assert.Throws<ArgumentException>(() => TerrainCaptureView.CreateTopDown(0, 0, 5f, 4f));
        Assert.Throws<ArgumentException>(() => TerrainCaptureView.CreateTopDown(0, 0, float.NaN, 4f));
    }
}
