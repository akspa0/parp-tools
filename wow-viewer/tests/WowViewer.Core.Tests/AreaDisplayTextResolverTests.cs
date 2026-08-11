using WowViewer.Core.World;

namespace WowViewer.Core.Tests;

public sealed class AreaDisplayTextResolverTests
{
    [Fact]
    public void ChildAreaProducesZoneTextAndSubzoneText()
    {
        var zone = new AreaContextEntry(10, "Elwynn Forest", 0, 0, 0, 0, 0x000A0000);
        var subzone = new AreaContextEntry(11, "Goldshire", 10, 0, 0, 0, 0x000A0001);

        AreaDisplayText result = AreaDisplayTextResolver.Resolve(
            subzone,
            zone,
            AreaContextSource.PackedAreaNumber,
            AreaResolutionReason.Resolved);

        Assert.Equal("Elwynn Forest", result.ZoneText);
        Assert.Equal("Goldshire", result.SubzoneText);
        Assert.Equal("Goldshire", result.PrimaryText);
        Assert.Equal(AreaResolutionReason.Resolved, result.Reason);
    }

    [Fact]
    public void ZoneOnlyAreaUsesZoneAsSubzoneFallback()
    {
        var zone = new AreaContextEntry(10, "Elwynn Forest", 0, 0, 0, 0, 0x000A0000);

        AreaDisplayText result = AreaDisplayTextResolver.Resolve(
            zone,
            parent: null,
            AreaContextSource.DirectAreaId,
            AreaResolutionReason.Resolved);

        Assert.Equal("Elwynn Forest", result.ZoneText);
        Assert.Equal("Elwynn Forest", result.SubzoneText);
    }

    [Fact]
    public void MissingLocalizedNameRemainsExplicit()
    {
        var zone = new AreaContextEntry(10, string.Empty, 0, 0, 0, 0, 0x000A0000);

        AreaDisplayText result = AreaDisplayTextResolver.Resolve(
            zone,
            parent: null,
            AreaContextSource.DirectAreaId,
            AreaResolutionReason.Resolved);

        Assert.Equal(AreaResolutionReason.MissingLocalizedName, result.Reason);
        Assert.Null(result.PrimaryText);
    }
}
