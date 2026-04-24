using WowViewer.Core.IO.Maps;
using System.Text;

namespace WowViewer.Core.Tests;

public sealed class LooseWorldMapDiscoveryTests
{
    [Fact]
    public void Discover_FindsMapsUnderClientAndOverlayRoots()
    {
        string tempRoot = CreateTempDirectory();
        try
        {
            string clientRoot = Path.Combine(tempRoot, "client");
            string overlayRoot = Path.Combine(tempRoot, "overlay");
            Directory.CreateDirectory(Path.Combine(clientRoot, "DBFilesClient"));
            Directory.CreateDirectory(Path.Combine(overlayRoot, "DBFilesClient"));

            File.WriteAllBytes(
                Path.Combine(clientRoot, "DBFilesClient", "Map.dbc"),
                BuildMapDbc(
                [
                    (1u, "Azeroth", "Eastern Kingdoms"),
                    (2u, "Kalimdor", "Kalimdor"),
                ]));

            File.WriteAllBytes(
                Path.Combine(overlayRoot, "DBFilesClient", "Map.dbc"),
                BuildMapDbc(
                [
                    (1u, "Azeroth", "Eastern Kingdoms"),
                    (999u, "development", "development"),
                ]));

            Directory.CreateDirectory(Path.Combine(clientRoot, "Data", "World", "Maps", "Azeroth"));
            File.WriteAllBytes(Path.Combine(clientRoot, "Data", "World", "Maps", "Azeroth", "Azeroth.wdt"), [0x57, 0x44, 0x54]);
            File.WriteAllBytes(Path.Combine(clientRoot, "Data", "World", "Maps", "Azeroth", "Azeroth.wdl"), [0x57, 0x44, 0x4C]);

            Directory.CreateDirectory(Path.Combine(overlayRoot, "World", "Maps", "development"));
            File.WriteAllBytes(Path.Combine(overlayRoot, "World", "Maps", "development", "development.wdt"), [0x57, 0x44, 0x54]);

            IReadOnlyList<DiscoveredLooseWorldMap> maps = LooseWorldMapDiscovery.Discover(clientRoot, overlayRoot);

            DiscoveredLooseWorldMap azeroth = Assert.Single(maps, static map => map.Directory == "Azeroth");
            Assert.Equal("Eastern Kingdoms", azeroth.Name);
            Assert.True(azeroth.HasLooseWdt);
            Assert.True(azeroth.HasLooseWdl);

            DiscoveredLooseWorldMap development = Assert.Single(maps, static map => map.Directory == "development");
            Assert.Equal(999, development.Id);
            Assert.True(development.HasLooseWdt);
            Assert.False(development.HasLooseWdl);

            Assert.DoesNotContain(maps, static map => map.Directory == "Kalimdor");
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void Discover_UsesClientMapDbc_WhenNoLooseOverlayMapDbcExists()
    {
        string tempRoot = CreateTempDirectory();
        try
        {
            string clientRoot = Path.Combine(tempRoot, "client");
            Directory.CreateDirectory(Path.Combine(clientRoot, "DBFilesClient"));
            File.WriteAllBytes(
                Path.Combine(clientRoot, "DBFilesClient", "Map.dbc"),
                BuildMapDbc(
                [
                    (1u, "Azeroth", "Eastern Kingdoms"),
                    (530u, "Expansion01", "Outland"),
                ]));

            IReadOnlyList<DiscoveredLooseWorldMap> maps = LooseWorldMapDiscovery.Discover(clientRoot);

            Assert.Equal(2, maps.Count);
            DiscoveredLooseWorldMap outland = Assert.Single(maps, static map => map.Directory == "Expansion01");
            Assert.Equal("Outland", outland.Name);
            Assert.False(outland.HasLooseFiles);
            Assert.Null(outland.LooseSourceDirectory);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static string CreateTempDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), $"wowviewer-loose-maps-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private static byte[] BuildMapDbc(IReadOnlyList<(uint Id, string Directory, string Name)> rows)
    {
        Dictionary<string, uint> offsets = new(StringComparer.Ordinal);
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);

        uint GetOffset(string value)
        {
            if (offsets.TryGetValue(value, out uint existing))
                return existing;

            uint offset = checked((uint)stringStream.Length);
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            stringStream.Write(bytes, 0, bytes.Length);
            stringStream.WriteByte(0);
            offsets[value] = offset;
            return offset;
        }

        using MemoryStream recordsStream = new();
        using BinaryWriter recordsWriter = new(recordsStream, Encoding.UTF8, leaveOpen: true);
        foreach ((uint id, string directory, string name) in rows)
        {
            recordsWriter.Write(id);
            recordsWriter.Write(GetOffset(directory));
            recordsWriter.Write(0u);
            recordsWriter.Write(0u);
            recordsWriter.Write(GetOffset(name));
        }

        recordsWriter.Flush();

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(0x43424457u);
        writer.Write(checked((uint)rows.Count));
        writer.Write(5u);
        writer.Write(20u);
        writer.Write(checked((uint)stringStream.Length));

        recordsStream.Position = 0;
        recordsStream.CopyTo(stream);
        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }
}
