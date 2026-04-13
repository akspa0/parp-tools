using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmDatasetExporterTests
{
    [Fact]
    public void TryResolveArchiveMapDirectoryAlias_ExactNormalizedMatch_ReturnsDirectory()
    {
        string[] knownFiles =
        [
            "World/Maps/LostIsles/LostIsles.wdt"
        ];

        string? resolved = VlmDatasetExporter.TryResolveArchiveMapDirectoryAlias("Lost Isles", knownFiles);

        Assert.Equal("LostIsles", resolved);
    }

    [Fact]
    public void TryResolveArchiveMapDirectoryAlias_NearMatch_ReturnsBestDirectory()
    {
        string[] knownFiles =
        [
            "World/Maps/Deephome/Deephome.wdt",
            "World/Maps/EmeraldDream/EmeraldDream.wdt"
        ];

        string? resolved = VlmDatasetExporter.TryResolveArchiveMapDirectoryAlias("Deepholm", knownFiles);

        Assert.Equal("Deephome", resolved);
    }
}