using MdxLTool.Formats.Mdx;
using MdxViewer.Rendering;
using Warcraft.NET.Files.M2.Entries;
using Xunit;

namespace MdxViewer.Tests.Rendering;

public sealed class WarcraftNetM2AdapterTests
{
    [Fact]
    public void MapLayerFlags_PreservesBasicSurfaceFlags_WithoutInjectingDepthDisables()
    {
        var flags = WarcraftNetM2Adapter.MapLayerFlags(0x7, 0);

        Assert.True(flags.HasFlag(MdlGeoFlags.Unshaded));
        Assert.True(flags.HasFlag(MdlGeoFlags.Unfogged));
        Assert.True(flags.HasFlag(MdlGeoFlags.TwoSided));
        Assert.False(flags.HasFlag(MdlGeoFlags.NoDepthTest));
        Assert.False(flags.HasFlag(MdlGeoFlags.NoDepthSet));
    }

    [Theory]
    [InlineData(0x8)]
    [InlineData(0x10)]
    [InlineData(0x18)]
    public void MapLayerFlags_DoesNotTreatUnknownHighBitsAsDepthOverrides(ushort renderFlags)
    {
        var flags = WarcraftNetM2Adapter.MapLayerFlags(renderFlags, 0);

        Assert.False(flags.HasFlag(MdlGeoFlags.NoDepthTest));
        Assert.False(flags.HasFlag(MdlGeoFlags.NoDepthSet));
    }

    [Fact]
    public void MapLayerFlags_MapsTextureWrapFlags()
    {
        var flags = WarcraftNetM2Adapter.MapLayerFlags(0, TextureFlags.Flag_0x1_WrapX | TextureFlags.Flag_0x2_WrapY);

        Assert.True(flags.HasFlag(MdlGeoFlags.WrapWidth));
        Assert.True(flags.HasFlag(MdlGeoFlags.WrapHeight));
    }
}