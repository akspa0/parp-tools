using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadSetResolverTests
{
    [Fact]
    public void ResolveActiveSetIndex_EmptyOrNegativeCount_ReturnsNegativeOne()
    {
        Assert.Equal(-1, WmoDoodadSetResolver.ResolveActiveSetIndex(0, 0));
        Assert.Equal(-1, WmoDoodadSetResolver.ResolveActiveSetIndex(2, -1));
    }

    [Fact]
    public void ResolveActiveSetIndex_ValidIndex_ReturnsRequested()
    {
        Assert.Equal(0, WmoDoodadSetResolver.ResolveActiveSetIndex(0, 3));
        Assert.Equal(1, WmoDoodadSetResolver.ResolveActiveSetIndex(1, 3));
        Assert.Equal(2, WmoDoodadSetResolver.ResolveActiveSetIndex(2, 3));
    }

    [Fact]
    public void ResolveActiveSetIndex_OutOfBounds_ClampsToZero()
    {
        Assert.Equal(0, WmoDoodadSetResolver.ResolveActiveSetIndex(-1, 3));
        Assert.Equal(0, WmoDoodadSetResolver.ResolveActiveSetIndex(3, 3));
        Assert.Equal(0, WmoDoodadSetResolver.ResolveActiveSetIndex(100, 3));
    }

    [Theory]
    [InlineData(0, 0, 5, true)]
    [InlineData(4, 0, 5, true)]
    [InlineData(5, 0, 5, false)]
    [InlineData(-1, 0, 5, false)]
    [InlineData(10, 10, 5, true)]
    [InlineData(14, 10, 5, true)]
    [InlineData(15, 10, 5, false)]
    [InlineData(9, 10, 5, false)]
    public void IsPlacementInSet_EvaluatesPlacementRange(int placementIndex, int startIndex, int count, bool expected)
    {
        Assert.Equal(expected, WmoDoodadSetResolver.IsPlacementInSet(placementIndex, startIndex, count));
    }

    [Fact]
    public void GetPlacementRange_BoundsToTotalPlacements()
    {
        // Normal within bounds
        var (s1, e1) = WmoDoodadSetResolver.GetPlacementRange(2, 5, 20);
        Assert.Equal(2, s1);
        Assert.Equal(7, e1);

        // Clamped by total placements
        var (s2, e2) = WmoDoodadSetResolver.GetPlacementRange(15, 10, 20);
        Assert.Equal(15, s2);
        Assert.Equal(20, e2);

        // Start out of range
        var (s3, e3) = WmoDoodadSetResolver.GetPlacementRange(25, 5, 20);
        Assert.Equal(0, s3);
        Assert.Equal(0, e3);
    }

    [Fact]
    public void ResolveActiveSet_Details_ComputesAccurateResolution()
    {
        var sets = new List<WmoDoodadSetDetail>
        {
            new(0, "Set $Default", 0, 10, 10, 0),
            new(1, "Set HighElves", 10, 8, 18, 0),
            new(2, "Set Scourge", 18, 12, 30, 0),
        };

        // Resolve active set 1
        var res1 = WmoDoodadSetResolver.ResolveActiveSet(sets, 1, 30);
        Assert.True(res1.IsValid);
        Assert.False(res1.IsClamped);
        Assert.Equal(1, res1.ActiveIndex);
        Assert.Equal("Set HighElves", res1.Name);
        Assert.Equal(10, res1.StartIndex);
        Assert.Equal(8, res1.Count);
        Assert.Equal(18, res1.RangeEndExclusive);
        Assert.Equal(3, res1.TotalSets);

        // Resolve out of bounds index (clamped to 0)
        var resClamped = WmoDoodadSetResolver.ResolveActiveSet(sets, 99, 30);
        Assert.True(resClamped.IsValid);
        Assert.True(resClamped.IsClamped);
        Assert.Equal(0, resClamped.ActiveIndex);
        Assert.Equal("Set $Default", resClamped.Name);
        Assert.Equal(0, resClamped.StartIndex);
        Assert.Equal(10, resClamped.Count);

        // Null or empty
        var resEmpty = WmoDoodadSetResolver.ResolveActiveSet((IReadOnlyList<WmoDoodadSetDetail>?)null, 0, 30);
        Assert.False(resEmpty.IsValid);
        Assert.Equal(-1, resEmpty.ActiveIndex);
    }

    [Fact]
    public void FormatSetSummary_GeneratesReadableOutput()
    {
        var valid = new WmoActiveDoodadSetResolution(0, "Set $Default", 0, 15, 15, 3, false, true);
        Assert.Equal("[0] \"Set $Default\": 15 doodads (0..14)", WmoDoodadSetResolver.FormatSetSummary(valid));

        var clamped = new WmoActiveDoodadSetResolution(0, "Set $Default", 0, 15, 15, 3, true, true);
        Assert.Equal("[0] \"Set $Default\": 15 doodads (0..14) (clamped)", WmoDoodadSetResolver.FormatSetSummary(clamped));

        var empty = new WmoActiveDoodadSetResolution(1, "EmptySet", 10, 0, 10, 3, false, true);
        Assert.Equal("[1] \"EmptySet\": 0 doodads", WmoDoodadSetResolver.FormatSetSummary(empty));

        var invalid = new WmoActiveDoodadSetResolution(-1, "", 0, 0, 0, 0, false, false);
        Assert.Equal("No doodad sets", WmoDoodadSetResolver.FormatSetSummary(invalid));
    }
}
