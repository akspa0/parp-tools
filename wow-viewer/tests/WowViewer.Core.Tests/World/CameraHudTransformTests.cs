using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests.World;

public sealed class CameraHudTransformTests
{
    [Fact]
    public void LocalAxesMapToCameraBasis()
    {
        CameraHudTransform transform = CameraHudTransform.Create(
            new Vector3(10f, 20f, 30f),
            Vector3.UnitY,
            Vector3.UnitZ);

        Assert.Equal(new Vector3(12f, 20f, 30f), transform.ToWorldPoint(new Vector3(2f, 0f, 0f)));
        Assert.Equal(new Vector3(10f, 20f, 33f), transform.ToWorldPoint(new Vector3(0f, 3f, 0f)));
        Assert.Equal(new Vector3(10f, 24f, 30f), transform.ToWorldPoint(new Vector3(0f, 0f, 4f)));
    }

    [Fact]
    public void ModelMatrixAnchorsLocalMeshAtCameraSpacePosition()
    {
        CameraHudTransform transform = CameraHudTransform.Create(
            new Vector3(10f, 20f, 30f),
            Vector3.UnitY,
            Vector3.UnitZ);

        Matrix4x4 model = transform.CreateModelMatrix(
            new Vector3(2f, 3f, 4f),
            Quaternion.Identity,
            uniformScale: 1f);

        Assert.Equal(new Vector3(12f, 24f, 33f), Vector3.Transform(Vector3.Zero, model));
    }

    [Fact]
    public void ProjectionClampsDepthAndViewportCoordinates()
    {
        CameraSpaceProjection projection = new(
            verticalFovDegrees: 90f,
            aspectRatio: 2f,
            nearDepth: 0.5f,
            farDepth: 2f);

        Assert.Equal(0.5f, projection.ClampDepth(float.NaN));
        Assert.Equal(2f, projection.ClampDepth(20f));

        Vector2 extents = projection.GetHalfExtents(depth: 1f);
        Assert.Equal(2f, extents.X, precision: 5);
        Assert.Equal(1f, extents.Y, precision: 5);
        Assert.Equal(new Vector3(2f, -1f, 1f), projection.GetLocalPoint(new Vector2(3f, -2f), depth: 1f));
    }
}
