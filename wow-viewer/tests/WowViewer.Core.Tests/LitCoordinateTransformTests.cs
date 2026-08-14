using System.Numerics;
using WowViewer.Core.Lit;

namespace WowViewer.Core.Tests;

public sealed class LitCoordinateTransformTests
{
    [Fact]
    public void ReportedDuskwoodHeaderConvertsFromRawXzyIntoMapRendererSpace()
    {
        Vector3 gamePosition = LitCoordinateTransform.ToGameWorldPosition(
            new Vector3(612096f, 0f, 998400f));
        Vector3 rendererPosition = LitCoordinateTransform.ToRendererPosition(gamePosition, 17066.66666f);

        Assert.Equal(612096f / 36f, gamePosition.X, 4);
        Assert.Equal(998400f / 36f, gamePosition.Y, 4);
        Assert.Equal(0f, gamePosition.Z, 4);
        Assert.Equal(17066.66666f - (998400f / 36f), rendererPosition.X, 4);
        Assert.Equal(17066.66666f - (612096f / 36f), rendererPosition.Y, 4);
        Assert.Equal(0f, rendererPosition.Z, 4);
    }
}
