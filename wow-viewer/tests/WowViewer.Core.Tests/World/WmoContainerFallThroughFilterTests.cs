using System.Numerics;
using WowViewer.Core.Runtime.World;
using Xunit;

namespace WowViewer.Core.Tests.World;

public sealed class WmoContainerFallThroughFilterTests
{
    [Fact]
    public void IsPointInsideAabb_ReturnsTrue_WhenPointIsInsideBounds()
    {
        var min = new Vector3(0, 0, 0);
        var max = new Vector3(10, 10, 10);
        var point = new Vector3(5, 5, 5);

        Assert.True(WmoContainerFallThroughFilter.IsPointInsideAabb(point, min, max));
    }

    [Fact]
    public void IsPointInsideAabb_ReturnsTrue_WhenPointIsWithinMargin()
    {
        var min = new Vector3(0, 0, 0);
        var max = new Vector3(10, 10, 10);
        var point = new Vector3(10.3f, 5, 5);

        Assert.True(WmoContainerFallThroughFilter.IsPointInsideAabb(point, min, max, margin: 0.5f));
    }

    [Fact]
    public void IsPointInsideAabb_ReturnsFalse_WhenPointIsOutsideMargin()
    {
        var min = new Vector3(0, 0, 0);
        var max = new Vector3(10, 10, 10);
        var point = new Vector3(12, 5, 5);

        Assert.False(WmoContainerFallThroughFilter.IsPointInsideAabb(point, min, max, margin: 0.5f));
    }

    [Fact]
    public void ApplyFallThrough_RemovesEnclosingWmo_WhenInteriorObjectIsPresent()
    {
        // Enclosing WMO (e.g. Town Hall)
        var wmo = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 1,
            IsWmo: true,
            BoundsMin: new Vector3(0, 0, 0),
            BoundsMax: new Vector3(50, 50, 20),
            SelectionPoint: new Vector3(25, 25, 10));

        // Interior Doodad or MDX (e.g. Throne / Table inside Town Hall)
        var interiorProp = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 2,
            IsWmo: false,
            BoundsMin: new Vector3(20, 20, 0),
            BoundsMax: new Vector3(24, 24, 4),
            SelectionPoint: new Vector3(22, 22, 2));

        var candidates = new List<WmoContainerFallThroughFilter.CandidateObject> { wmo, interiorProp };

        var result = WmoContainerFallThroughFilter.ApplyFallThrough(candidates);

        // Enclosing WMO should be removed so click falls through to interior prop
        Assert.Single(result);
        Assert.Equal(2, result[0].Id);
        Assert.False(result[0].IsWmo);
    }

    [Fact]
    public void ApplyFallThrough_PreservesWmo_WhenNoInteriorObjectHit()
    {
        // WMO clicked from exterior (no interior hits along ray)
        var wmo = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 1,
            IsWmo: true,
            BoundsMin: new Vector3(0, 0, 0),
            BoundsMax: new Vector3(50, 50, 20),
            SelectionPoint: new Vector3(25, 25, 10));

        var candidates = new List<WmoContainerFallThroughFilter.CandidateObject> { wmo };

        var result = WmoContainerFallThroughFilter.ApplyFallThrough(candidates);

        // Exterior WMO is retained
        Assert.Single(result);
        Assert.Equal(1, result[0].Id);
        Assert.True(result[0].IsWmo);
    }

    [Fact]
    public void ApplyFallThrough_HandlesNestedWmos_PrioritizesInnerWmo()
    {
        // Enclosing Large WMO (e.g. Courtyard / Fortress)
        var outerWmo = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 1,
            IsWmo: true,
            BoundsMin: new Vector3(0, 0, 0),
            BoundsMax: new Vector3(100, 100, 40),
            SelectionPoint: new Vector3(50, 50, 20));

        // Nested Small WMO (e.g. Tower / Forge inside Courtyard)
        var innerWmo = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 2,
            IsWmo: true,
            BoundsMin: new Vector3(20, 20, 0),
            BoundsMax: new Vector3(40, 40, 30),
            SelectionPoint: new Vector3(30, 30, 15));

        var candidates = new List<WmoContainerFallThroughFilter.CandidateObject> { outerWmo, innerWmo };

        var result = WmoContainerFallThroughFilter.ApplyFallThrough(candidates);

        // Outer container falls through to inner WMO
        Assert.Single(result);
        Assert.Equal(2, result[0].Id);
    }

    [Fact]
    public void ApplyFallThrough_PreservesWmoDoodadCandidateAndUnrelatedRayOrder()
    {
        var parentWmo = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 0,
            IsWmo: true,
            BoundsMin: new Vector3(0, 0, 0),
            BoundsMax: new Vector3(50, 50, 20),
            SelectionPoint: new Vector3(25, 25, 10));
        var activeSetDoodad = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 1,
            IsWmo: false,
            BoundsMin: new Vector3(19, 19, 0),
            BoundsMax: new Vector3(21, 21, 4),
            SelectionPoint: new Vector3(20, 20, 2));
        var unrelatedObject = new WmoContainerFallThroughFilter.CandidateObject(
            Id: 2,
            IsWmo: false,
            BoundsMin: new Vector3(70, 70, 0),
            BoundsMax: new Vector3(74, 74, 4),
            SelectionPoint: new Vector3(72, 72, 2));

        var result = WmoContainerFallThroughFilter.ApplyFallThrough(
            [parentWmo, activeSetDoodad, unrelatedObject]);

        Assert.Equal([1, 2], result.Select(static candidate => candidate.Id));
    }
}
