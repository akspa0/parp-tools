using System.Text;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class AreaIdMapperTests
{
    [Fact]
    public void Constructor_DoesNotRequireEmbeddedCrosswalkDefaults()
    {
        AreaIdMapper mapper = new();

        Assert.True(mapper.CrosswalkCount >= 0);
        Assert.NotNull(mapper.LastLoadMessage);
    }

    [Fact]
    public void LoadCrosswalkCsv_LoadsMatchingReportFormatMappings()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(tempDirectory, "azeroth.csv"),
                "src_mapId,src_areaId,src_parentId,src_isZone,src_name,match_count,matches\n" +
                "0,500,0,1,Dun Morogh,1,0:7:1:map_name:active:Dun Morogh\n");

            AreaIdMapper mapper = new();
            mapper.LoadCrosswalkCsv(tempDirectory);

            Assert.Equal(7, mapper.MapAreaId(500, 0));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void LoadDbcs_PrefersContinentSpecificNameMatchWhenHintProvided()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            string alphaAreaPath = Path.Combine(tempDirectory, "AreaTableAlpha.dbc");
            string lkAreaPath = Path.Combine(tempDirectory, "AreaTableLk.dbc");

            File.WriteAllBytes(
                alphaAreaPath,
                BuildDbc(
                    fieldCount: 13,
                    rows:
                    [
                        [500u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u],
                    ],
                    stringBlockEntries: ["Echo Isles"]));

            File.WriteAllBytes(
                lkAreaPath,
                BuildDbc(
                    fieldCount: 12,
                    rows:
                    [
                        [10u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u],
                        [11u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u],
                    ],
                    stringBlockEntries: ["Echo Isles"]));

            AreaIdMapper mapper = new();
            mapper.LoadDbcs(alphaAreaPath, lkAreaPath);

            Assert.Equal(11, mapper.MapAreaId(500, 1));
            Assert.Equal(10, mapper.MapAreaId(500, 0));
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void TryLoadKnownTestDataFromRoot_MissingTrees_ReportsExplicitWarning()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            AreaIdMapper mapper = new();

            bool loaded = mapper.TryLoadKnownTestDataFromRoot(tempDirectory);

            Assert.False(loaded);
            Assert.False(mapper.LastLoadUsedSchemaDefinitions);
            Assert.NotNull(mapper.LastLoadMessage);
            Assert.Contains("Missing extracted AreaTable/Map DBC trees", mapper.LastLoadMessage);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void TryLoadKnownTestDataFromRoot_UsesDbcdWhenDefinitionsAndTablesExist()
    {
        string tempDirectory = CreateTempDirectory();
        try
        {
            string definitionsDirectory = Path.Combine(tempDirectory, "gillijimproject_refactor", "lib", "WoWDBDefs", "definitions");
            Directory.CreateDirectory(definitionsDirectory);
            File.Copy(
                Path.Combine(WorkspaceRoot(), "gillijimproject_refactor", "lib", "WoWDBDefs", "definitions", "AreaTable.dbd"),
                Path.Combine(definitionsDirectory, "AreaTable.dbd"));
            File.Copy(
                Path.Combine(WorkspaceRoot(), "gillijimproject_refactor", "lib", "WoWDBDefs", "definitions", "Map.dbd"),
                Path.Combine(definitionsDirectory, "Map.dbd"));

            string alphaDbcDir = Path.Combine(tempDirectory, "gillijimproject_refactor", "test_data", "0.5.3", "tree", "DBFilesClient");
            string lkDbcDir = Path.Combine(tempDirectory, "gillijimproject_refactor", "test_data", "3.3.5", "tree", "DBFilesClient");
            Directory.CreateDirectory(alphaDbcDir);
            Directory.CreateDirectory(lkDbcDir);

            File.WriteAllBytes(Path.Combine(alphaDbcDir, "AreaTable.dbc"), BuildAlphaAreaTableDbc());
            File.WriteAllBytes(Path.Combine(alphaDbcDir, "Map.dbc"), BuildAlphaMapDbc());
            File.WriteAllBytes(Path.Combine(lkDbcDir, "AreaTable.dbc"), BuildLkAreaTableDbc());
            File.WriteAllBytes(Path.Combine(lkDbcDir, "Map.dbc"), BuildLkMapDbc());

            AreaIdMapper mapper = new();

            bool loaded = mapper.TryLoadKnownTestDataFromRoot(tempDirectory);

            Assert.True(loaded);
            Assert.True(mapper.LastLoadUsedSchemaDefinitions);
            Assert.Equal(1, mapper.AlphaAreaCount);
            Assert.Equal(2, mapper.LkAreaCount);
            Assert.Equal(0, mapper.MapContinent(5));
            Assert.Equal(11, mapper.MapAreaId(500, 0));
            Assert.Contains("DBCD+WoWDBDefs", mapper.LastLoadMessage);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void TryLoadFromArchives_UsesArchiveBackedDbcdTables()
    {
        AreaIdMapper mapper = new();
        FakeArchiveReader alphaArchive = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\AreaTable.dbc"] = BuildAlphaAreaTableDbc(),
                ["DBFilesClient\\Map.dbc"] = BuildAlphaMapDbc(),
            });
        FakeArchiveReader lkArchive = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\AreaTable.dbc"] = BuildLkAreaTableDbc(),
                ["DBFilesClient\\Map.dbc"] = BuildLkMapDbc(),
            });

        Assert.NotNull(DbClientFileReader.TryReadTable(alphaArchive, "AreaTable"));
        Assert.NotNull(DbClientFileReader.TryReadTable(lkArchive, "AreaTable"));

        bool loaded = mapper.TryLoadFromArchives(alphaArchive, "0.5.3", lkArchive, "3.3.5.12340");

        Assert.True(loaded, mapper.LastLoadMessage);
        Assert.True(mapper.LastLoadUsedSchemaDefinitions);
        Assert.Equal(1, mapper.AlphaAreaCount);
        Assert.Equal(2, mapper.LkAreaCount);
        Assert.Equal(11, mapper.MapAreaId(500, 0));
        Assert.Contains("archive-backed sources", mapper.LastLoadMessage);
    }

    [Fact]
    public void TryLoadFromArchives_MissingTables_ReportsExplicitWarning()
    {
        AreaIdMapper mapper = new();

        bool loaded = mapper.TryLoadFromArchives(new FakeArchiveReader(), "0.5.3", new FakeArchiveReader(), "3.3.5.12340");

        Assert.False(loaded);
        Assert.False(mapper.LastLoadUsedSchemaDefinitions);
        Assert.NotNull(mapper.LastLoadMessage);
        Assert.Contains("archive-backed DBC sources", mapper.LastLoadMessage);
    }

    private static string CreateTempDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"wowviewer-area-mapper-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static byte[] BuildDbc(uint fieldCount, IReadOnlyList<uint[]> rows, IReadOnlyList<string> stringBlockEntries)
    {
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);

        foreach (string entry in stringBlockEntries)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(entry);
            stringStream.Write(bytes, 0, bytes.Length);
            stringStream.WriteByte(0);
        }

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(0x43424457u);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(fieldCount * 4u);
        writer.Write(checked((uint)stringStream.Length));

        foreach (uint[] row in rows)
        {
            Assert.Equal((int)fieldCount, row.Length);
            foreach (uint value in row)
            {
                writer.Write(value);
            }
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] BuildAlphaAreaTableDbc()
    {
        return BuildDbc(
            fieldCount: 14,
            rows:
            [
                [500u, 0u, 5u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u],
            ],
            stringBlockEntries: ["Echo Isles"]);
    }

    private static byte[] BuildLkAreaTableDbc()
    {
        return BuildDbc(
            fieldCount: 20,
            rows:
            [
                [11u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, FloatBits(0f), FloatBits(0f), 0u],
                [12u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, FloatBits(0f), FloatBits(0f), 0u],
            ],
            stringBlockEntries: ["Echo Isles"]);
    }

    private static byte[] BuildAlphaMapDbc()
    {
        return BuildDbc(
            fieldCount: 5,
            rows:
            [
                [5u, 1u, 0u, 0u, 8u],
            ],
            stringBlockEntries: ["Azeroth", "Echo Isles"]);
    }

    private static byte[] BuildLkMapDbc()
    {
        return BuildDbc(
            fieldCount: 18,
            rows:
            [
                [0u, 1u, 0u, 0u, 0u, 9u, 0u, 0u, 0u, 0u, FloatBits(1f), 0u, FloatBits(0f), FloatBits(0f), 0u, 0u, 0u, 0u],
                [1u, 2u, 0u, 0u, 0u, 9u, 0u, 0u, 0u, 0u, FloatBits(1f), 0u, FloatBits(0f), FloatBits(0f), 0u, 0u, 0u, 0u],
            ],
            stringBlockEntries: ["Azeroth", "Echo Isles"]);
    }

    private static uint FloatBits(float value)
    {
        return BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);
    }

    private static string WorkspaceRoot()
    {
        string current = AppContext.BaseDirectory;
        DirectoryInfo? directory = new(current);
        while (directory != null)
        {
            string wowViewerRoot = Path.Combine(directory.FullName, "wow-viewer");
            if (Directory.Exists(wowViewerRoot))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate workspace root for WoWDBDefs test fixtures.");
    }

    private sealed class FakeArchiveReader(Dictionary<string, byte[]>? files = null) : IArchiveReader
    {
        private readonly Dictionary<string, byte[]> _files = files ?? new(StringComparer.OrdinalIgnoreCase);

        public bool FileExists(string virtualPath)
        {
            return _files.ContainsKey(virtualPath);
        }

        public byte[]? ReadFile(string virtualPath)
        {
            return _files.TryGetValue(virtualPath, out byte[]? data) ? data : null;
        }
    }
}