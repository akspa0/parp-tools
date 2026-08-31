using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MapConversionFormatTests
{
    [Theory]
    [InlineData("alpha-wdt-0.5.3", MapConversionTargetFormat.AlphaWdt053)]
    [InlineData("lk-adt-v18", MapConversionTargetFormat.LkAdtV18)]
    [InlineData("mop-split-adt", MapConversionTargetFormat.MopSplitAdt)]
    public void ParsesExplicitTargetNames(string value, MapConversionTargetFormat expected)
    {
        Assert.True(MapConversionFormats.TryParseTarget(value, out MapConversionTargetFormat actual));
        Assert.Equal(expected, actual);
        Assert.Equal(value, MapConversionFormats.GetCommandValue(actual));
    }

    [Fact]
    public void RejectsMoPSplitTargetWithoutRoutingThroughLkWriter()
    {
        MapConversionValidationResult result = MapConversionFormats.Validate(
            MapConversionSourceFormat.SplitAdtFamily,
            MapConversionTargetFormat.MopSplitAdt);

        Assert.False(result.IsSupported);
        Assert.Contains("not implemented", result.Error, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("LkAdtWriter", result.Error, StringComparison.Ordinal);
    }

    [Fact]
    public void ReportsExplicitLossForSplitToLkDownConversion()
    {
        MapConversionValidationResult result = MapConversionFormats.Validate(
            MapConversionSourceFormat.SplitAdtFamily,
            MapConversionTargetFormat.LkAdtV18);

        Assert.True(result.IsSupported);
        Assert.True(result.IsLossy);
        Assert.Contains(result.Warnings, warning => warning.Contains("monolithic", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(result.Warnings, warning => warning.Contains("companion", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void AlphaIdentityRouteIsNotLossy()
    {
        MapConversionValidationResult result = MapConversionFormats.Validate(
            MapConversionSourceFormat.AlphaWdt053,
            MapConversionTargetFormat.AlphaWdt053);

        Assert.True(result.IsSupported);
        Assert.False(result.IsLossy);
        Assert.Empty(result.Warnings);
    }
}
