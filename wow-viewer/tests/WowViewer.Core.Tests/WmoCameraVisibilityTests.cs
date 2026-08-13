using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class WmoCameraVisibilityTests
{
    [Fact]
    public void GroupContainmentKeepsInteriorVisibleWhenRootBoundsMiss()
    {
        var groups = new List<(Vector3 Min, Vector3 Max)>
        {
            (new Vector3(-10f, -10f, -10f), new Vector3(10f, 10f, 10f))
        };

        bool inside = WmoCameraVisibility.IsInsideRootOrGroup(
            Vector3.Zero,
            new Vector3(100f, 100f, 100f),
            new Vector3(200f, 200f, 200f),
            groups,
            padding: 0f);

        Assert.True(inside);
    }

    [Fact]
    public void CameraOutsideRootAndGroupsDoesNotForceInteriorVisibility()
    {
        var groups = new List<(Vector3 Min, Vector3 Max)>
        {
            (new Vector3(-10f, -10f, -10f), new Vector3(10f, 10f, 10f))
        };

        bool inside = WmoCameraVisibility.IsInsideRootOrGroup(
            new Vector3(50f, 50f, 50f),
            new Vector3(-20f, -20f, -20f),
            new Vector3(20f, 20f, 20f),
            groups,
            padding: 0f);

        Assert.False(inside);
    }
}
