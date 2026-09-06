using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class CameraPathBindingPolicyTests
{
    [Fact]
    public void MapPathAndWdtNamesNormalizeToTheSameIdentity()
    {
        Assert.True(CameraPathBindingPolicy.AreEquivalentMapNames(
            "World\\Maps\\Azeroth.wdt",
            "azeroth"));
        Assert.False(CameraPathBindingPolicy.AreEquivalentMapNames("Azeroth", "Kalimdor"));
    }

    [Fact]
    public void BuildFormattingNormalizesWithoutDroppingIdentity()
    {
        Assert.True(CameraPathBindingPolicy.AreEquivalentBuildVersions("build=0_5_3_3368", "v0.5.3.3368"));
        Assert.False(CameraPathBindingPolicy.AreEquivalentBuildVersions("0.5.3.3368-alpha", "0.5.3.3368"));
        Assert.False(CameraPathBindingPolicy.AreEquivalentBuildVersions("0.5.3.3368", "0.5.3.3369"));
    }

    [Fact]
    public void MissingBuildIsNotEquivalentDuringPlayback()
    {
        Assert.False(CameraPathBindingPolicy.AreEquivalent(
            "Azeroth",
            "unknown",
            "Azeroth",
            "3.3.5.12340"));
        Assert.False(CameraPathBindingPolicy.AreEquivalent(
            "Azeroth",
            "unknown",
            "Kalimdor",
            "3.3.5.12340"));
        Assert.False(CameraPathBindingPolicy.AreEquivalentBuildVersions("client-alpha", "3.3.5.12340"));
        Assert.False(CameraPathBindingPolicy.AreEquivalentBuildVersions("3.3.5.12340", "unknown_build"));
    }
}
