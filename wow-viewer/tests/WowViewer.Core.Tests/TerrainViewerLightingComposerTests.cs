using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TerrainViewerLightingComposerTests
{
    private static readonly TerrainViewerLightingState Global = new(
        new Vector3(0.8f, 0.78f, 0.7f),
        new Vector3(0.55f, 0.55f, 0.6f),
        new Vector3(0.6f, 0.7f, 0.85f),
        200f,
        1500f);

    private static readonly TerrainViewerLightingState Local = new(
        new Vector3(0.2f, 0.3f, 0.4f),
        new Vector3(0.1f, 0.2f, 0.3f),
        new Vector3(0.3f, 0.2f, 0.1f),
        100f,
        500f);

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void ComposeGlobalWithLocal_NoUsableLocalWeight_PreservesGlobalExactly(float weight)
    {
        TerrainViewerLightingState result =
            TerrainViewerLightingComposer.ComposeGlobalWithLocal(Global, Local, weight);

        Assert.Equal(Global, result);
    }

    [Fact]
    public void ComposeGlobalWithLocal_PartialLocalWeight_BlendsEveryEffectiveSignal()
    {
        TerrainViewerLightingState result =
            TerrainViewerLightingComposer.ComposeGlobalWithLocal(Global, Local, 0.5f);

        AssertVector(new Vector3(0.5f, 0.54f, 0.55f), result.DirectionalColor);
        AssertVector(new Vector3(0.325f, 0.375f, 0.45f), result.AmbientColor);
        AssertVector(new Vector3(0.45f, 0.45f, 0.475f), result.FogColor);
        Assert.Equal(150f, result.FogStart);
        Assert.Equal(1000f, result.FogEnd);
    }

    [Fact]
    public void ComposeGlobalWithLocal_WeightAboveOne_UsesLocalProfile()
    {
        TerrainViewerLightingState result =
            TerrainViewerLightingComposer.ComposeGlobalWithLocal(Global, Local, 2f);

        Assert.Equal(Local, result);
    }

    private static void AssertVector(Vector3 expected, Vector3 actual)
    {
        Assert.Equal(expected.X, actual.X, 5);
        Assert.Equal(expected.Y, actual.Y, 5);
        Assert.Equal(expected.Z, actual.Z, 5);
    }
}
