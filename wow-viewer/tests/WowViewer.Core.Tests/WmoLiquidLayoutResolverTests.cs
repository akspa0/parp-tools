using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoLiquidLayoutResolverTests
{
    [Fact]
    public void ResolveCoordinateFamily_UsesAssetVersionBeforeBuildHint()
    {
        WmoLiquidCoordinateFamily family = WmoLiquidLayoutResolver.ResolveCoordinateFamily(14, "3.3.5.12340");

        Assert.Equal(WmoLiquidCoordinateFamily.LegacyV14, family);
    }

    [Fact]
    public void ResolveCoordinateFamily_MapsStandardAssetVersions()
    {
        WmoLiquidCoordinateFamily family = WmoLiquidLayoutResolver.ResolveCoordinateFamily(17, "0.5.3");

        Assert.Equal(WmoLiquidCoordinateFamily.StandardV17Plus, family);
    }

    [Fact]
    public void ResolveCoordinateFamily_FallsBackToBuildFamilyWhenVersionUnknown()
    {
        WmoLiquidCoordinateFamily family = WmoLiquidLayoutResolver.ResolveCoordinateFamily(null, "0.6.0.3592");

        Assert.Equal(WmoLiquidCoordinateFamily.LegacyV16, family);
    }

    [Fact]
    public void GetBaselineRotationQuarterTurns_RemainsNeutralForLegacyAndStandardFamilies()
    {
        Assert.Equal(0, WmoLiquidLayoutResolver.GetBaselineRotationQuarterTurns(14, "0.5.3"));
        Assert.Equal(0, WmoLiquidLayoutResolver.GetBaselineRotationQuarterTurns(17, "3.3.5.12340"));
    }
}