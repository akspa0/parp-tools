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

        try
        {
            File.WriteAllText(tempListfile, "world/wmo/khazmodan/cities/ironforge/ironforge.wmo");

            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                "world/wmo/khazmodan/cities/ironforge/ironforge.wmo",
                ["I:/fake/game"],
                tempListfile,
                factory);

            Assert.Equal("I:/fake/game", Assert.Single(catalog.LoadedRoots));
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

            Assert.Equal("I:/fake/game", Assert.Single(catalog.LoadedRoots));
            Assert.Contains(catalog.LoadedListfileEntries, static entry => string.Equals(entry.Replace('\\', '/'), "world/wmo/khazmodan/cities/ironforge/ironforge.wmo", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(catalog.LoadedListfileEntries, static entry => string.Equals(entry.Replace('\\', '/'), "creature/azjolnerub/giant/azjolroofgiant.m2", StringComparison.OrdinalIgnoreCase));
            Assert.Equal([1, 2, 3], bytes);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
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
}