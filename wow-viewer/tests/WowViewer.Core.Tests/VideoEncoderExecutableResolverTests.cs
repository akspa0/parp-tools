using WoWViewer.Capture;

namespace WowViewer.Core.Tests;

public sealed class VideoEncoderExecutableResolverTests
{
    [Theory]
    [InlineData(null, "")]
    [InlineData("", "")]
    [InlineData("  ffmpeg  ", "ffmpeg")]
    [InlineData("  \"C:\\tools\\ffmpeg\\ffmpeg.exe\"  ", "C:\\tools\\ffmpeg\\ffmpeg.exe")]
    public void NormalizeConfiguredExecutable_RemovesOnlyOuterWhitespaceAndQuotes(string? configured, string expected)
    {
        Assert.Equal(expected, VideoEncoderExecutableResolver.NormalizeConfiguredExecutable(configured));
    }

    [Fact]
    public void Resolve_DefaultConfigurationFallsBackToPathWhenNoAppLocalEncoderExists()
    {
        string appDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));

        VideoEncoderResolution resolution = VideoEncoderExecutableResolver.Resolve("ffmpeg", appDirectory);

        Assert.Equal(VideoEncoderSource.Path, resolution.Source);
        Assert.Equal("ffmpeg", resolution.Executable);
        Assert.Equal(Path.Combine(appDirectory, "ffmpeg.exe"), resolution.AppLocalCandidate);
    }

    [Fact]
    public void Resolve_ExplicitPathStaysConfiguredEvenWhenItIsNotYetReachable()
    {
        VideoEncoderResolution resolution = VideoEncoderExecutableResolver.Resolve("  \"D:\\portable\\ffmpeg.exe\" ", "C:\\viewer");

        Assert.Equal(VideoEncoderSource.Configured, resolution.Source);
        Assert.Equal("D:\\portable\\ffmpeg.exe", resolution.Executable);
    }
}
