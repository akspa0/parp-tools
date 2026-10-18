using WowViewer.Core.Runtime.World;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class FormatFlagsDescriberTests
{
    [Fact]
    public void DescribeWdtMphdFlags_DecodesKnownBitsAndUnknown()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeWdtMphdFlags(0));
        
        uint flags = 0x01 | 0x04 | 0x20;
        string desc = FormatFlagsDescriber.DescribeWdtMphdFlags(flags);
        Assert.Contains("GlobalMapObj(0x1)", desc);
        Assert.Contains("EnvMapShader/MultiPass(0x4)", desc);
        Assert.Contains("FlipGround(0x20)", desc);

        string withUnk = FormatFlagsDescriber.DescribeWdtMphdFlags(0x80);
        Assert.Contains("Unk=0x80", withUnk);
    }

    [Fact]
    public void DescribeWdtMainFlags_DecodesKnownBits()
    {
        Assert.Equal("Empty(0x0)", FormatFlagsDescriber.DescribeWdtMainFlags(0));
        
        string desc = FormatFlagsDescriber.DescribeWdtMainFlags(0x01 | 0x02);
        Assert.Contains("HasAdt(0x1)", desc);
        Assert.Contains("InfiniteOcean(0x2)", desc);
    }

    [Fact]
    public void DescribeWmoRootFlags_DecodesKnownBits()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeWmoRootFlags(0));

        uint flags = 0x02 | 0x08;
        string desc = FormatFlagsDescriber.DescribeWmoRootFlags(flags);
        Assert.Contains("UnifiedLighting(0x2)", desc);
        Assert.Contains("PreserveMocvAlpha(0x8)", desc);
    }

    [Fact]
    public void DescribeWmoGroupFlags_DecodesAntiportalAndOccluder()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeWmoGroupFlags(0));

        uint antiportalFlags = 0x4000080;
        string desc = FormatFlagsDescriber.DescribeWmoGroupFlags(antiportalFlags, "antiportal_001");
        Assert.Contains("Antiportal/Occluder(0x4M)", desc);
        Assert.Contains("NoCollision/Occluder(0x80)", desc);

        uint regularExterior = 0x0000008 | 0x0000040;
        string extDesc = FormatFlagsDescriber.DescribeWmoGroupFlags(regularExterior);
        Assert.Contains("Exterior(0x8)", extDesc);
        Assert.Contains("ExteriorLit(0x40)", extDesc);

        string namedAntiportal = FormatFlagsDescriber.DescribeWmoGroupFlags(0, "stormwind_antiportal_01");
        Assert.Contains("AntiportalByName", namedAntiportal);
    }

    [Fact]
    public void DescribeM2ModelFlags_DecodesSurfaceBasisAndShadows()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeM2ModelFlags(0));

        string desc = FormatFlagsDescriber.DescribeM2ModelFlags(0x03 | 0x80);
        Assert.Contains("SurfaceBasis(0x3)", desc);
        Assert.Contains("CastShadows(0x80)", desc);

        string tilt = FormatFlagsDescriber.DescribeM2ModelFlags(0x01);
        Assert.Contains("TiltNormal/AlignUp(0x1)", tilt);
    }

    [Fact]
    public void DescribeM2MaterialFlags_DecodesShadingAndShadowBits()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeM2MaterialFlags(0));

        ushort flags = 0x001 | 0x004 | 0x040;
        string desc = FormatFlagsDescriber.DescribeM2MaterialFlags(flags);
        Assert.Contains("Unlit(0x1)", desc);
        Assert.Contains("TwoSided(0x4)", desc);
        Assert.Contains("DisableShadowCasting(0x40)", desc);
    }

    [Fact]
    public void DescribeMcnkFlags_DecodesMccvAndAlpha()
    {
        Assert.Equal("None", FormatFlagsDescriber.DescribeMcnkFlags(0));

        int raw = 0x01 | 0x40 | 0x8000;
        string desc = FormatFlagsDescriber.DescribeMcnkFlags(raw);
        Assert.Contains("HasShadows(0x1)", desc);
        Assert.Contains("HasMccv(0x40)", desc);
        Assert.Contains("FixedAlpha(0x8000)", desc);
    }
}
