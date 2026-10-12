using WoWViewer.Terrain;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 247 US5: DAT folders as Cartography layer donors. A layer names its donor with one string, so a DAT
/// folder is encoded into that slot as "dat:&lt;folder&gt;". These pin the locator round trip and that an
/// ordinary map name is never mistaken for one — the seam must be inert for every existing layer.
/// </summary>
public sealed class DatLayerSourceTests
{
    [Theory]
    [InlineData("Azeroth")]
    [InlineData("Kalimdor")]
    [InlineData("DeadminesInstance")]
    [InlineData("")]
    [InlineData(null)]
    public void OrdinaryMapNames_AreNotDatSources(string? mapName)
    {
        Assert.False(DatLayerSource.IsDatSource(mapName));
        Assert.False(DatLayerSource.TryGetFolder(mapName, out _));
        Assert.Null(DatLayerSource.Resolve(mapName, 36f));
        Assert.Empty(DatLayerSource.OccupiedTiles(mapName, 36f));
    }

    [Fact]
    public void ForFolder_RoundTripsThroughTheLocator()
    {
        string folder = Path.Combine(Path.GetTempPath(), "dat-layer-round-trip");
        string locator = DatLayerSource.ForFolder(folder);

        Assert.True(DatLayerSource.IsDatSource(locator));
        Assert.True(DatLayerSource.TryGetFolder(locator, out string? parsed));
        Assert.Equal(Path.GetFullPath(folder), parsed);
    }

    [Fact]
    public void DisplayName_UsesTheFolderLeaf_AndLeavesMapNamesAlone()
    {
        string locator = DatLayerSource.ForFolder(Path.Combine("E:", "WC2", "wrat2", "world", "maps", "Expansion01"));

        Assert.Equal("DAT: Expansion01", DatLayerSource.DisplayName(locator));
        Assert.Equal("Azeroth", DatLayerSource.DisplayName("Azeroth"));
    }

    [Fact]
    public void Resolve_OnAMissingFolder_IsADisplayableStateNotAThrow()
    {
        string locator = DatLayerSource.ForFolder(Path.Combine(Path.GetTempPath(), "dat-layer-does-not-exist-" + Guid.NewGuid().ToString("N")));

        Assert.Null(DatLayerSource.Resolve(locator, 36f));
        Assert.Empty(DatLayerSource.OccupiedTiles(locator, 36f));
    }

    [Fact]
    public void Resolve_OnAFolderWithNoDatFiles_ReturnsNull()
    {
        string folder = Path.Combine(Path.GetTempPath(), "dat-layer-empty-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            File.WriteAllText(Path.Combine(folder, "notes.txt"), "not a DAT file");
            string locator = DatLayerSource.ForFolder(folder);

            Assert.Null(DatLayerSource.Resolve(locator, 36f));
            Assert.Empty(DatLayerSource.OccupiedTiles(locator, 36f));
        }
        finally
        {
            DatLayerSource.Forget(DatLayerSource.ForFolder(folder));
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void OccupiedTiles_DecodesTheAdapterTileKeys()
    {
        // The adapter keys tiles as tileX * 64 + tileY, and the footprint overlay needs (x, y) pairs back.
        string folder = Path.Combine(Path.GetTempPath(), "dat-layer-footprint-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        string locator = DatLayerSource.ForFolder(folder);
        try
        {
            // Filenames carry the tile location when there is no ALOC, which is the v22/v23 case.
            File.WriteAllBytes(Path.Combine(folder, "area_24_38.dat"), BuildMinimalAhdrFile());
            File.WriteAllBytes(Path.Combine(folder, "area_25_37.dat"), BuildMinimalAhdrFile());
            DatLayerSource.Forget(locator);

            var tiles = DatLayerSource.OccupiedTiles(locator, 36f).ToHashSet();

            // The adapter places renderer TileX = ALOC tile Y and TileY = ALOC tile X.
            Assert.Equal(2, tiles.Count);
            Assert.Contains((38, 24), tiles);
            Assert.Contains((37, 25), tiles);
        }
        finally
        {
            DatLayerSource.Forget(locator);
            Directory.Delete(folder, recursive: true);
        }
    }

    /// <summary>MVER 22 + AHDR, enough for the folder scan's content sniff and version read.</summary>
    private static byte[] BuildMinimalAhdrFile()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("REVM"u8.ToArray());
        w.Write(4);
        w.Write(22u);
        w.Write("RDHA"u8.ToArray());
        w.Write(64);
        w.Write(22u);
        w.Write(129u);
        w.Write(129u);
        w.Write(16u);
        w.Write(16u);
        for (int i = 0; i < 11; i++)
            w.Write(0u);
        w.Flush();
        return ms.ToArray();
    }
}
