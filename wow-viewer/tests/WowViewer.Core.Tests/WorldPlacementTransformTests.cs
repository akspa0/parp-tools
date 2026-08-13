using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class WorldPlacementTransformTests
{
    [Fact]
    public void BuildUsesSharedRendererRotationConventionForMdxAndWmo()
    {
        Vector3 position = new(10f, 20f, 30f);
        Vector3 rotation = new(17f, -29f, 43f);
        const float scale = 1.75f;

        float rotationX = -rotation.Y * (MathF.PI / 180f);
        float rotationY = -rotation.X * (MathF.PI / 180f);
        float rotationZ = rotation.Z * (MathF.PI / 180f);
        Matrix4x4 expected = Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rotationX)
            * Matrix4x4.CreateRotationY(rotationY)
            * Matrix4x4.CreateRotationZ(rotationZ)
            * Matrix4x4.CreateTranslation(position);

        Matrix4x4 actual = WorldPlacementTransform.Build(position, rotation, scale);

        AssertMatrixEqual(expected, actual);
    }

    [Fact]
    public void BuildPreservesPlacementPosition()
    {
        Vector3 position = new(-120f, 80f, 14f);

        Matrix4x4 transform = WorldPlacementTransform.Build(position, Vector3.Zero);

        Assert.Equal(position, transform.Translation);
    }

    private static void AssertMatrixEqual(Matrix4x4 expected, Matrix4x4 actual)
    {
        float[] expectedValues =
        [
            expected.M11, expected.M12, expected.M13, expected.M14,
            expected.M21, expected.M22, expected.M23, expected.M24,
            expected.M31, expected.M32, expected.M33, expected.M34,
            expected.M41, expected.M42, expected.M43, expected.M44
        ];
        float[] actualValues =
        [
            actual.M11, actual.M12, actual.M13, actual.M14,
            actual.M21, actual.M22, actual.M23, actual.M24,
            actual.M31, actual.M32, actual.M33, actual.M34,
            actual.M41, actual.M42, actual.M43, actual.M44
        ];

        for (int index = 0; index < expectedValues.Length; index++)
            Assert.Equal(expectedValues[index], actualValues[index], precision: 5);
    }
}
