using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class ArchiveCatalogBootstrapperTests
{
    [Fact]
    public void ParseExternalListfileLines_HandlesIdPrefixedRows()
    {
        IReadOnlyList<string> entries = ArchiveCatalogBootstrapper.ParseExternalListfileLines(
        [
            "123;World\\Maps\\Azeroth\\Azeroth.wdt",
            "  Creature\\Wolf\\wolf.mdx  ",
            "",
        ]);

        Assert.Equal(
        [
            "World\\Maps\\Azeroth\\Azeroth.wdt",
            "Creature\\Wolf\\wolf.mdx",
        ], entries);
    }

    [Fact]
    public void Bootstrap_LoadsArchiveRootsAndCombinesKnownFiles()
    {
        FakeArchiveCatalog archiveCatalog = new(
            internalFiles: ["World\\Maps\\Azeroth\\Azeroth.wdt"],
            knownFiles: ["World\\Maps\\Azeroth\\Azeroth_0_0.adt"]);

        string tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(tempFile,
            [
                "42;World\\Maps\\Azeroth\\Azeroth_0_0_tex0.adt",
            ]);

            ArchiveCatalogBootstrapResult result = ArchiveCatalogBootstrapper.Bootstrap(
                archiveCatalog,
                ["I:/fake/game"],
                tempFile);

            Assert.Equal(["I:/fake/game"], archiveCatalog.LoadedRoots);
            Assert.Equal(tempFile, archiveCatalog.LoadedListfilePath);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth.wdt", result.AllFiles);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth.wdt", result.KnownFiles);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth_0_0.adt", result.AllFiles);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth_0_0_tex0.adt", result.AllFiles);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Bootstrap_ListfileCache_WritesAndReloadsTrustedAndSupplementalEntries()
    {
        string tempDirectory = Path.Combine(Path.GetTempPath(), $"wowviewer-listfile-cache-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDirectory);

        string tempListfile = Path.Combine(tempDirectory, "community-listfile.txt");
        File.WriteAllLines(tempListfile,
        [
            "42;World\\Maps\\Azeroth\\Azeroth_0_0_tex0.adt",
        ]);

        try
        {
            ArchiveCatalogBootstrapOptions writeOptions = new(
                ExternalListfilePath: tempListfile,
                ListfileCacheKey: "3.3.5.12340",
                ListfileCacheDirectoryPath: tempDirectory);

            FakeArchiveCatalog writerCatalog = new(
                internalFiles: ["World\\Maps\\Azeroth\\Azeroth.wdt"],
                knownFiles: []);

            ArchiveCatalogBootstrapResult written = ArchiveCatalogBootstrapper.Bootstrap(
                writerCatalog,
                ["I:/fake/wrath/Data"],
                writeOptions);

            Assert.NotNull(written.ListfileCachePath);
            Assert.True(File.Exists(written.ListfileCachePath));

            FakeArchiveCatalog readerCatalog = new(
                internalFiles: [],
                knownFiles: []);

            ArchiveCatalogBootstrapResult loaded = ArchiveCatalogBootstrapper.Bootstrap(
                readerCatalog,
                ["I:/fake/wrath/Data"],
                new ArchiveCatalogBootstrapOptions(
                    ListfileCacheKey: "3.3.5.12340",
                    ListfileCacheDirectoryPath: tempDirectory,
                    PersistListfileCache: false));

            Assert.Contains("World\\Maps\\Azeroth\\Azeroth.wdt", loaded.CachedListfileEntries);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth_0_0_tex0.adt", loaded.CachedListfileEntries);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth.wdt", loaded.KnownFiles);
            Assert.Contains("World\\Maps\\Azeroth\\Azeroth_0_0_tex0.adt", loaded.KnownFiles);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    private sealed class FakeArchiveCatalog : IArchiveCatalog
    {
        private readonly IReadOnlyList<string> _internalFiles;
        private readonly IReadOnlyList<string> _knownFiles;

        public FakeArchiveCatalog(IReadOnlyList<string> internalFiles, IReadOnlyList<string> knownFiles)
        {
            _internalFiles = internalFiles;
            _knownFiles = knownFiles;
        }

        public List<string> LoadedRoots { get; } = [];

        public string? LoadedListfilePath { get; private set; }

        public List<string> LoadedEntries { get; } = [];

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
            LoadedEntries.AddRange(entries);
        }

        public IReadOnlyList<string> ExtractInternalListfiles() => _internalFiles;

        public IReadOnlyList<string> GetAllKnownFiles()
        {
            return _knownFiles
                .Concat(LoadedEntries)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        public bool FileExists(string virtualPath) => false;

        public byte[]? ReadFile(string virtualPath) => null;

        public void Dispose()
        {
        }
    }
}