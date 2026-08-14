using WowViewer.Core.World;

namespace WowViewer.Core.Tests;

public sealed class AreaNumberPartsTests
{
    [Fact]
    public void FromRawSplitsAlphaAreaNumberIntoZoneAndSubzoneWords()
    {
        AreaNumberParts parts = AreaNumberParts.FromRaw(unchecked((int)0x12345678));

        Assert.Equal((ushort)0x1234, parts.Zone);
        Assert.Equal((ushort)0x5678, parts.Subzone);
        Assert.Equal(0x12340000u, parts.ZoneBase);
        Assert.Equal(0x12345678u, parts.Raw);
    }

    [Fact]
    public void FromRawPreservesPackedValuesWithHighSignedBit()
    {
        AreaNumberParts parts = AreaNumberParts.FromRaw(unchecked((int)0x80010002));

        Assert.Equal((ushort)0x8001, parts.Zone);
        Assert.Equal((ushort)0x0002, parts.Subzone);
        Assert.Equal(unchecked((int)0x80010002), parts.SignedRaw);
    }
}
