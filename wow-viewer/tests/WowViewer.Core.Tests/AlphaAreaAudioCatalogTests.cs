using WowViewer.Core.Audio;

namespace WowViewer.Core.Tests;

public sealed class AlphaAreaAudioCatalogTests
{
    [Fact]
    public void TryResolveWithParents_InheritsZoneMusicWhenChildHasNoAssignment()
    {
        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [10] = new(10, 0, 0, "Zone", 0, 0, 321, 0, 0),
                [11] = new(11, 0, 10, "Subzone", 0, 0, 0, 0, 0),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>());

        AlphaAreaAudioBinding? binding = catalog.TryResolveWithParents(11);

        Assert.NotNull(binding);
        Assert.Equal(10, binding!.Area.Id);
        Assert.Equal(321, binding.Area.ZoneMusicId);
    }

    [Fact]
    public void TryResolveWithParents_StopsOnParentCycles()
    {
        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [10] = new(10, 0, 11, "A", 0, 0, 0, 0, 0),
                [11] = new(11, 0, 10, "B", 0, 0, 0, 0, 0),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>());

        Assert.Null(catalog.TryResolveWithParents(10));
    }

    [Fact]
    public void TryResolveWithParents_ResolvesAlphaPackedAreaNumberAndParentAreaNumber()
    {
        const int continentId = 0;
        const int zoneAreaNumber = 0x000A0000;
        const int subzoneAreaNumber = 0x000A0001;

        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [100] = new(
                    100,
                    continentId,
                    0,
                    "Zone",
                    0,
                    0,
                    321,
                    0,
                    0,
                    zoneAreaNumber,
                    zoneAreaNumber),
                [101] = new(
                    101,
                    continentId,
                    0,
                    "Subzone",
                    0,
                    0,
                    0,
                    0,
                    0,
                    subzoneAreaNumber,
                    zoneAreaNumber),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>());

        AlphaAreaAudioBinding? binding = catalog.TryResolveWithParents(subzoneAreaNumber, continentId);

        Assert.NotNull(binding);
        Assert.Equal(100, binding!.Area.Id);
        Assert.Equal(zoneAreaNumber, binding.Area.AreaNumber);
        Assert.Equal(321, binding.Area.ZoneMusicId);
    }

    [Fact]
    public void TryResolve_RequiresContinentWhenAlphaAreaNumberIsAmbiguous()
    {
        const int areaNumber = 0x000A0001;

        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [100] = new(100, 0, 0, "Eastern", 0, 0, 321, 0, 0, areaNumber, 0),
                [200] = new(200, 1, 0, "Kalimdor", 0, 0, 654, 0, 0, areaNumber, 0),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>());

        Assert.Null(catalog.TryResolve(areaNumber));
        Assert.Equal(321, catalog.TryResolve(areaNumber, 0)!.Area.ZoneMusicId);
        Assert.Equal(654, catalog.TryResolve(areaNumber, 1)!.Area.ZoneMusicId);
    }

    [Fact]
    public void TryResolve_UsesHighAndLowWordsForUnsignedPackedAreaNumbers()
    {
        const int areaNumber = unchecked((int)0x80010002);

        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [areaNumber] = new(areaNumber, 0, 0, "Unsigned packed area", 0, 0, 777, 0, 0, areaNumber, 0)
            },
            new Dictionary<int, AlphaAreaMidiAmbience>());

        AlphaAreaAudioBinding? binding = catalog.TryResolve(areaNumber, 0);

        Assert.NotNull(binding);
        Assert.Equal(777, binding!.Area.ZoneMusicId);
    }
}
