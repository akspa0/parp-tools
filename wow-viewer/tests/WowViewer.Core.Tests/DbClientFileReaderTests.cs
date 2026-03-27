using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class DbClientFileReaderTests
{
    [Fact]
    public void EnumerateTablePaths_ReturnsExpectedPreferenceOrder()
    {
        IReadOnlyList<string> paths = DbClientFileReader.EnumerateTablePaths("AreaTable");

        Assert.Equal(
            [
                "DBFilesClient\\AreaTable.dbc",
                "DBFilesClient\\AreaTable.db2",
                "DBFilesClient/AreaTable.dbc",
                "DBFilesClient/AreaTable.db2",
                "DBC\\AreaTable.dbc",
                "DBC\\AreaTable.db2",
                "DBC/AreaTable.dbc",
                "DBC/AreaTable.db2",
                "AreaTable.dbc",
                "AreaTable.db2",
            ],
            paths);
    }

    [Fact]
    public void TryReadTable_ReturnsFirstMatchingPayload()
    {
        FakeArchiveReader archiveReader = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["AreaTable.db2"] = [1, 2, 3],
            });

        byte[]? data = DbClientFileReader.TryReadTable(archiveReader, "AreaTable");

        Assert.Equal([1, 2, 3], data);
        Assert.Equal(
            [
                "DBFilesClient\\AreaTable.dbc",
                "DBFilesClient\\AreaTable.db2",
                "DBFilesClient/AreaTable.dbc",
                "DBFilesClient/AreaTable.db2",
                "DBC\\AreaTable.dbc",
                "DBC\\AreaTable.db2",
                "DBC/AreaTable.dbc",
                "DBC/AreaTable.db2",
                "AreaTable.dbc",
                "AreaTable.db2",
            ],
            archiveReader.RequestedPaths);
    }

    private sealed class FakeArchiveReader : IArchiveReader
    {
        private readonly Dictionary<string, byte[]> _files;

        public FakeArchiveReader(Dictionary<string, byte[]> files)
        {
            _files = files;
        }

        public List<string> RequestedPaths { get; } = [];

        public bool FileExists(string virtualPath)
        {
            RequestedPaths.Add(virtualPath);
            return _files.ContainsKey(virtualPath);
        }

        public byte[]? ReadFile(string virtualPath)
        {
            RequestedPaths.Add(virtualPath);
            return _files.TryGetValue(virtualPath, out byte[]? data) ? data : null;
        }
    }
}