using WoWViewer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class CinematicCameraOriginResolverTests
{
    [Fact]
    public void ModernCinematicCameraRowMatchesByFileDataIdWithoutModelPath()
    {
        Assert.True(CinematicCameraOriginResolver.IsAssetMatch(
            "Cameras\\FlybyUndead.M2",
            modelPath: null,
            assetFileDataId: 116912,
            rowFileDataId: 116912));
    }

    [Fact]
    public void CinematicCameraFileDataIdMismatchDoesNotMatch()
    {
        Assert.False(CinematicCameraOriginResolver.IsAssetMatch(
            "Cameras\\FlybyUndead.M2",
            modelPath: null,
            assetFileDataId: 116912,
            rowFileDataId: 116913));
    }

    [Fact]
    public void LegacyCinematicCameraRowStillMatchesByModelPath()
    {
        Assert.True(CinematicCameraOriginResolver.IsAssetMatch(
            "Cameras\\FlybyUndead.M2",
            modelPath: "Cameras\\FlybyUndead.mdx",
            assetFileDataId: null,
            rowFileDataId: null));
    }
}
