using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class ArchiveVirtualFileReaderTests
{
    [Fact]
    public void ReadVirtualFile_UsesCatalogFactoryAndListfile()
    {
        FakeArchiveCatalog catalog = new();
        FakeArchiveCatalogFactory factory = new(catalog);
        string tempListfile = Path.GetTempFileName();
        string expectedRoot = Path.GetFullPath("I:/fake/game");

        try
        {
            File.WriteAllText(tempListfile, "world/wmo/khazmodan/cities/ironforge/ironforge.wmo");

            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                "world/wmo/khazmodan/cities/ironforge/ironforge.wmo",
                ["I:/fake/game"],
                tempListfile,
                factory);

            Assert.Equal(expectedRoot, Assert.Single(catalog.LoadedRoots));
            Assert.Equal(tempListfile, catalog.LoadedListfilePath);
            Assert.Equal("world/wmo/khazmodan/cities/ironforge/ironforge.wmo", catalog.ReadRequests.Single());
            Assert.Equal([1, 2, 3], bytes);
        }
        finally
        {
            File.Delete(tempListfile);
        }
    }

    [Fact]
    public void ReadVirtualFile_ThrowsWhenCatalogCannotResolvePath()
    {
        FakeArchiveCatalog catalog = new();
        FakeArchiveCatalogFactory factory = new(catalog);

        FileNotFoundException exception = Assert.Throws<FileNotFoundException>(() =>
            ArchiveVirtualFileReader.ReadVirtualFile("world/wmo/missing.wmo", ["I:/fake/game"], (string?)null, factory));

        Assert.Equal("world/wmo/missing.wmo", exception.FileName);
    }

    [Fact]
    public void ReadVirtualFile_LoadsCachedEntriesWhenBootstrapOptionsProvided()
    {
        FakeArchiveCatalog catalog = new();
        FakeArchiveCatalogFactory factory = new(catalog);
        string tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        string expectedRoot = Path.GetFullPath("I:/fake/game");
        Directory.CreateDirectory(tempDirectory);

        try
        {
            ArchiveListfileCache.Write(
                tempDirectory,
                "3.3.5.12340",
                ["I:/fake/game"],
                ["world/wmo/khazmodan/cities/ironforge/ironforge.wmo"],
                ["creature/azjolnerub/giant/azjolroofgiant.m2"]);

            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                "world/wmo/khazmodan/cities/ironforge/ironforge.wmo",
                ["I:/fake/game"],
                new ArchiveCatalogBootstrapOptions(
                    ListfileCacheKey: "3.3.5.12340",
                    ListfileCacheDirectoryPath: tempDirectory),
                factory);

            Assert.Equal(expectedRoot, Assert.Single(catalog.LoadedRoots));
            Assert.Contains(catalog.LoadedListfileEntries, static entry => string.Equals(entry.Replace('\\', '/'), "world/wmo/khazmodan/cities/ironforge/ironforge.wmo", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(catalog.LoadedListfileEntries, static entry => string.Equals(entry.Replace('\\', '/'), "creature/azjolnerub/giant/azjolroofgiant.m2", StringComparison.OrdinalIgnoreCase));
            Assert.Equal([1, 2, 3], bytes);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadVirtualFile_FallsBackToAlphaPerAssetMpqPath_WhenCatalogMisses()
    {
        if (!File.Exists(WmoTestPaths.StagedStormwindAlphaMpqPath))
            return;

        FakeArchiveCatalog catalog = new();
        FakeArchiveCatalogFactory factory = new(catalog);
        string tempDirectory = Path.Combine(Path.GetTempPath(), $"wowviewer-archive-reader-{Guid.NewGuid():N}");
        string mpqFilePath = Path.Combine(tempDirectory, "Data", "World", "wmo", "Azeroth", "Buildings", "Stormwind", "Stormwind.wmo.MPQ");
        Directory.CreateDirectory(Path.GetDirectoryName(mpqFilePath)!);

        try
        {
            File.Copy(WmoTestPaths.StagedStormwindAlphaMpqPath, mpqFilePath, overwrite: true);

            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                "World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ",
                [tempDirectory],
                (string?)null,
                factory);

            Assert.NotEmpty(bytes);
            Assert.Equal("REVM", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ReadVirtualFile_ReusesBootstrappedCatalogAcrossRepeatedReads()
    {
        ArchiveCatalogSessionCache.InvalidateAll();
        CountingArchiveCatalogFactory factory = new();
        string virtualPath = "world/wmo/khazmodan/cities/ironforge/ironforge.wmo";

        try
        {
            byte[] first = ArchiveVirtualFileReader.ReadVirtualFile(
                virtualPath,
                ["I:/fake/game"],
                (string?)null,
                factory);

            byte[] second = ArchiveVirtualFileReader.ReadVirtualFile(
                virtualPath,
                ["I:/fake/game"],
                (string?)null,
                factory);

            Assert.Equal([1, 2, 3], first);
            Assert.Equal([1, 2, 3], second);
            Assert.Equal(1, factory.CreateCount);
            Assert.Single(factory.CreatedCatalogs);
            Assert.Equal(1, factory.CreatedCatalogs[0].LoadArchivesCallCount);
            Assert.Equal(2, factory.CreatedCatalogs[0].ReadRequests.Count);
        }
        finally
        {
            ArchiveCatalogSessionCache.InvalidateAll();
        }
    }

    private sealed class FakeArchiveCatalogFactory : IArchiveCatalogFactory
    {
        private readonly IArchiveCatalog _catalog;

        public FakeArchiveCatalogFactory(IArchiveCatalog catalog)
        {
            _catalog = catalog;
        }

        public IArchiveCatalog Create() => _catalog;
    }

    private sealed class CountingArchiveCatalogFactory : IArchiveCatalogFactory
    {
        public int CreateCount { get; private set; }

        public List<CountingArchiveCatalog> CreatedCatalogs { get; } = [];

        public IArchiveCatalog Create()
        {
            CreateCount++;
            CountingArchiveCatalog catalog = new();
            CreatedCatalogs.Add(catalog);
            return catalog;
        }
    }

    private sealed class FakeArchiveCatalog : IArchiveCatalog
    {
        public List<string> LoadedRoots { get; } = [];

        public string? LoadedListfilePath { get; private set; }

		public List<string> LoadedListfileEntries { get; } = [];

        public List<string> ReadRequests { get; } = [];

        public void LoadArchives(IEnumerable<string> searchPaths)
        {
            LoadedRoots.AddRange(searchPaths);
        }

        public void LoadListfile(string path)
        {
            LoadedListfilePath = path;
        }

        public void LoadListfileEntries(IEnumerable<string> entries)
        {
			LoadedListfileEntries.AddRange(entries);
        }

        public IReadOnlyList<string> ExtractInternalListfiles() => Array.Empty<string>();

        public IReadOnlyList<string> GetAllKnownFiles() => Array.Empty<string>();

        public bool FileExists(string virtualPath) => false;

        public byte[]? ReadFile(string virtualPath)
        {
            ReadRequests.Add(virtualPath);
            return string.Equals(virtualPath, "world/wmo/khazmodan/cities/ironforge/ironforge.wmo", StringComparison.OrdinalIgnoreCase)
                ? [1, 2, 3]
                : null;
        }

        public void Dispose()
        {
        }
    }

    private sealed class CountingArchiveCatalog : IArchiveCatalog
    {
        public int LoadArchivesCallCount { get; private set; }

        public List<string> ReadRequests { get; } = [];

        public void LoadArchives(IEnumerable<string> searchPaths)
        {
            LoadArchivesCallCount++;
        }

        public void LoadListfile(string path)
        {
        }

        public void LoadListfileEntries(IEnumerable<string> entries)
        {
        }

        public IReadOnlyList<string> ExtractInternalListfiles() => Array.Empty<string>();

        public IReadOnlyList<string> GetAllKnownFiles() => Array.Empty<string>();

        public bool FileExists(string virtualPath) => false;

        public byte[]? ReadFile(string virtualPath)
        {
            ReadRequests.Add(virtualPath);
            return string.Equals(virtualPath, "world/wmo/khazmodan/cities/ironforge/ironforge.wmo", StringComparison.OrdinalIgnoreCase)
                ? [1, 2, 3]
                : null;
        }

        public void Dispose()
        {
        }
    }
}