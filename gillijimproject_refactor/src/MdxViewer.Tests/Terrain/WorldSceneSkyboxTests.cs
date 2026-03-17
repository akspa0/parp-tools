using MdxViewer.Terrain;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class WorldSceneSkyboxTests
{
    [Theory]
    [InlineData("Environments/Stars/IceCrownCitadelSky.m2")]
    [InlineData("Environments\\Stars\\BladesedgeSkyBox.m2")]
    [InlineData("World/Outland/PassiveDoodads/SkyBox/Clouds03.m2")]
    public void IsSkyboxModelPath_ReturnsTrue_ForKnownSkyboxFamilies(string modelPath)
    {
        Assert.True(WorldScene.IsSkyboxModelPath(modelPath));
    }

    [Theory]
    [InlineData("World/Generic/Undead/Passive Doodads/UndercitySkylight/UndercitySkylight01.m2")]
    [InlineData("World/Expansion01/Doodads/TheExodar/Passivedoodads/SKY_PORTAL/Exodar_Sky_portal.m2")]
    [InlineData("World/Generic/Human/PassiveDoodads/Barrel/Barrel01.m2")]
    public void IsSkyboxModelPath_ReturnsFalse_ForRegularWorldProps(string modelPath)
    {
        Assert.False(WorldScene.IsSkyboxModelPath(modelPath));
    }
}