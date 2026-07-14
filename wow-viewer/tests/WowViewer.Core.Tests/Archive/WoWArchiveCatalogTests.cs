using WowViewer.Core.IO.Archive;
using Xunit;

namespace WowViewer.Core.Tests.Archive;

public sealed class WoWArchiveCatalogTests
{
    [Fact]
    public void Scan_NonexistentDirectory_ThrowsDirectoryNotFound()
    {
        Assert.Throws<DirectoryNotFoundException>(() =>
            WoWArchiveCatalog.Scan(@"X:\Nonexistent\Path"));
    }

    [Fact]
    public void Scan_NoClientFilesInDirectory_ReturnsEmpty()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(tempDir);
            Assert.Empty(entries);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void WoWArchiveEra_Classify_ReturnsExpectedLabels()
    {
        Assert.Equal("Alpha", WoWArchiveEra.Classify("0.X"));
        Assert.Equal("Vanilla", WoWArchiveEra.Classify("1.X"));
        Assert.Equal("TBC", WoWArchiveEra.Classify("2.X"));
        Assert.Equal("Wrath", WoWArchiveEra.Classify("3.X"));
        Assert.Equal("Cata", WoWArchiveEra.Classify("4.X"));
        Assert.Equal("5.X", WoWArchiveEra.Classify("5.X"));
    }
}
