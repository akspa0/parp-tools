using System.Text;
using WowViewer.Core.IO.Dbc;

namespace WowViewer.Core.Tests;

public sealed class AreaIdMapperTests
{
    [Fact]
    public void Constructor_LoadsEmbeddedCrosswalkDefaults()
    {
        AreaIdMapper mapper = new();

        Assert.True(mapper.CrosswalkCount > 0);
        Assert.True(mapper.MapAreaId(1048576, 0) > 0);
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
}