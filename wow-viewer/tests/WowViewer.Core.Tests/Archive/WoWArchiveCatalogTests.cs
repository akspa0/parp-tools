using WowViewer.Core.IO.Archive;
using Xunit;

namespace WowViewer.Core.Tests.Archive;

public sealed class WoWArchiveCatalogTests
{
    private const string RealArchiveRoot = @"G:\WoW\WoWArchive-0.X-3.X";

    private static bool ArchiveAvailable => Directory.Exists(RealArchiveRoot);

    [Fact]
    public void Scan_RealArchive_ReturnsOver200Entries()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        Assert.True(entries.Count > 200, $"Expected >200 entries, got {entries.Count}");
    }

    [Fact]
    public void Scan_RealArchive_EntriesAreSortedByEraThenBuild()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        for (int i = 1; i < entries.Count; i++)
        {
            int eraCompare = string.Compare(entries[i - 1].Era, entries[i].Era, StringComparison.Ordinal);
            Assert.True(eraCompare <= 0,
                $"Entries not sorted by era at index {i}: '{entries[i - 1].Era}' -> '{entries[i].Era}'");
            if (eraCompare == 0)
            {
                int buildCompare = string.Compare(entries[i - 1].BuildVersion, entries[i].BuildVersion, StringComparison.Ordinal);
                Assert.True(buildCompare <= 0,
                    $"Entries not sorted by build at index {i}: '{entries[i - 1].BuildVersion}' -> '{entries[i].BuildVersion}'");
            }
        }
    }

    [Fact]
    public void Scan_RealArchive_HasKnownEras()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        var eras = entries.Select(e => e.Era).Distinct().OrderBy(e => e).ToList();
        Assert.Contains("Alpha", eras);
        Assert.Contains("Vanilla", eras);
        Assert.Contains("Wrath", eras);
    }

    [Fact]
    public void Scan_RealArchive_HasKnownPlatforms()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        var platforms = entries.Select(e => e.Platform).Distinct().OrderBy(p => p).ToList();
        Assert.Contains("Windows", platforms);
    }

    [Fact]
    public void Scan_RealArchive_NoDuplicateBuildVersions()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        var versions = entries.Select(e =>
        {
            int altIndex = e.BuildVersion.IndexOf("-alt", StringComparison.OrdinalIgnoreCase);
            return altIndex > 0 ? e.BuildVersion[..altIndex] : e.BuildVersion;
        }).ToList();
        var duplicates = versions.GroupBy(v => v).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
        Assert.Empty(duplicates);
    }

    [Fact]
    public void Scan_RealArchive_StatusIsSet()
    {
        SkipIfArchiveMissing();
        IReadOnlyList<WoWArchiveBuildEntry> entries = WoWArchiveCatalog.Scan(RealArchiveRoot);
        foreach (var entry in entries)
        {
            Assert.True(entry.Status is WoWArchiveBuildStatus.MountLive
                or WoWArchiveBuildStatus.Staged
                or WoWArchiveBuildStatus.Available,
                $"Unrecognized status {entry.Status} for {entry.BuildVersion}");
        }
    }

    [Fact]
    public void Scan_RealArchive_CompletesWithin250ms()
    {
        SkipIfArchiveMissing();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        WoWArchiveCatalog.Scan(RealArchiveRoot);
        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds < 250,
            $"Scan took {sw.ElapsedMilliseconds}ms, expected < 250ms");
    }

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

    private static void SkipIfArchiveMissing()
    {
        if (!ArchiveAvailable)
            Assert.True(true, "Skipped: WoWArchive not available at expected path.");
    }
}
