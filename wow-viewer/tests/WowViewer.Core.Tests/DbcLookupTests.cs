using System.Text;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class DbcLookupTests
{
    [Fact]
    public void MapDirectoryLookup_LoadsArchiveMapTableAndResolvesDirectory()
    {
        FakeArchiveReader archiveReader = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\Map.dbc"] = BuildDbc(
                    fieldCount: 5,
                    rows:
                    [
                        [1u, 1u, 0u, 0u, 9u],
                    ],
                    stringBlockEntries: ["Azeroth", "Eastern Kingdoms"]),
            });

        MapDirectoryLookup lookup = new();
        lookup.Load(Array.Empty<string>(), archiveReader);

        Assert.True(lookup.IsLoaded);
        Assert.Equal("Azeroth", lookup.ResolveDirectory("1"));
        Assert.Equal("Azeroth", lookup.ResolveDirectory("Eastern Kingdoms"));
    }

    [Fact]
    public void GroundEffectLookup_LoadsArchiveTablesAndReturnsModels()
    {
        FakeArchiveReader archiveReader = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\GroundEffectDoodad.dbc"] = BuildDbc(
                    fieldCount: 3,
                    rows:
                    [
                        [7u, 0u, 13u],
                    ],
                    stringBlockEntries: ["World\\NoExt", "World\\Plants\\Grass.mdx"]),
                ["DBFilesClient\\GroundEffectTexture.dbc"] = BuildDbc(
                    fieldCount: 5,
                    rows:
                    [
                        [12u, 7u, 0u, 0u, 0u],
                    ],
                    stringBlockEntries: []),
            });

        GroundEffectLookup lookup = new();
        lookup.Load(Array.Empty<string>(), archiveReader);

        Assert.True(lookup.IsLoaded);
        string[]? models = lookup.GetDoodadsEffect(12);
        Assert.NotNull(models);
        Assert.Equal(["World\\Plants\\Grass.mdx"], models);
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
                writer.Write(value);
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }

    private sealed class FakeArchiveReader : IArchiveReader
    {
        private readonly IReadOnlyDictionary<string, byte[]> _files;

        public FakeArchiveReader(IReadOnlyDictionary<string, byte[]> files)
        {
            _files = files;
        }

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