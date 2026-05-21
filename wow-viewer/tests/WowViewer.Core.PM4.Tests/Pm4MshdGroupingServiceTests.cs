using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4MshdGroupingServiceTests
{
    [Fact]
    public void Describe_NullHeader_ReturnsEmptyGroupingInfo()
    {
        Pm4MshdGroupingInfo info = Pm4MshdGroupingService.Describe(null);

        Assert.Equal(0u, info.Field00);
        Assert.Equal(0u, info.RegionId);
        Assert.Equal(0u, info.Field08);
        Assert.False(info.IsEmptyStubRegion);
    }

    [Fact]
    public void Describe_Field04EqualsOne_MarksStubRegion()
    {
        Pm4MshdHeader header = new(534, 1, 534, 0, 0, 0, 0, 0);

        Pm4MshdGroupingInfo info = Pm4MshdGroupingService.Describe(header);

        Assert.Equal(header.Field00, info.Field00);
        Assert.Equal(header.Field04, info.RegionId);
        Assert.Equal(header.Field08, info.Field08);
        Assert.True(info.IsEmptyStubRegion);
    }

    [Fact]
    public void Describe_DevelopmentTile_ExposesNonStubRegionForActiveTile()
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath);
        Pm4MshdHeader header = Assert.IsType<Pm4MshdHeader>(document.KnownChunks.Mshd);

        Pm4MshdGroupingInfo info = Pm4MshdGroupingService.Describe(header);

        Assert.True(document.KnownChunks.Msur.Count > 0);
        Assert.Equal(header.Field00, info.Field00);
        Assert.Equal(header.Field04, info.RegionId);
        Assert.Equal(header.Field08, info.Field08);
        Assert.NotEqual(1u, info.RegionId);
        Assert.False(info.IsEmptyStubRegion);
    }
}
