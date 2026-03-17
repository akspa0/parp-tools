using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class WdlPreviewSupportTests
{
    [Theory]
    [InlineData("0.5.3.3368")]
    [InlineData("0.5.5.3494")]
    [InlineData("0.5.9.1234")]
    public void IsWdlPreviewBuildSupported_AllowsAlphaPointFiveBuilds(string build)
    {
        Assert.True(ViewerApp.IsWdlPreviewBuildSupported(build));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("0.6.0.3592")]
    [InlineData("3.3.5.12340")]
    public void IsWdlPreviewBuildSupported_RejectsNonAlphaPointFiveBuilds(string? build)
    {
        Assert.False(ViewerApp.IsWdlPreviewBuildSupported(build));
    }
}