using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxMaterialRenderPolicyTests
{
    [Fact]
    public void ClampFinite_RejectsNonFiniteAndBoundsLegacyLighting()
    {
        Assert.Equal(0.0f, MdxMaterialRenderPolicy.ClampFinite(float.NaN, 0.0f, 4.0f));
        Assert.Equal(0.0f, MdxMaterialRenderPolicy.ClampFinite(float.PositiveInfinity, 0.0f, 4.0f));
        Assert.Equal(0.0f, MdxMaterialRenderPolicy.ClampFinite(-1.0f, 0.0f, 4.0f));
        Assert.Equal(4.0f, MdxMaterialRenderPolicy.ClampFinite(8.0f, 0.0f, 4.0f));
        Assert.Equal(2.0f, MdxMaterialRenderPolicy.ClampFinite(2.0f, 0.0f, 4.0f));
    }

    [Fact]
    public void SelectUvSet_OnlyUsesSecondSetWhenItExists()
    {
        Assert.Equal(0, MdxMaterialRenderPolicy.SelectUvSet(0, 2));
        Assert.Equal(1, MdxMaterialRenderPolicy.SelectUvSet(1, 2));
        Assert.Equal(0, MdxMaterialRenderPolicy.SelectUvSet(1, 1));
        Assert.Equal(0, MdxMaterialRenderPolicy.SelectUvSet(2, 4));
    }

    [Fact]
    public void UsesSphereEnvironmentMap_UsesLegacyFlagBit()
    {
        Assert.False(MdxMaterialRenderPolicy.UsesSphereEnvironmentMap(0));
        Assert.True(MdxMaterialRenderPolicy.UsesSphereEnvironmentMap(0x2));
        Assert.True(MdxMaterialRenderPolicy.UsesSphereEnvironmentMap(0x82));
    }
}
