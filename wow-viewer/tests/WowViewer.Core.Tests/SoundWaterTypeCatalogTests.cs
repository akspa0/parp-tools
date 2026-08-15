using WowViewer.Core.Audio;

namespace WowViewer.Core.Tests;

public sealed class SoundWaterTypeCatalogTests
{
    [Fact]
    public void TryResolve_UsesExactSoundTypeAndSubtype()
    {
        SoundWaterTypeCatalog catalog = new(
        [
            new SoundWaterTypeEntry(Id: 20, SoundType: 0, SoundSubtype: 4, SoundId: 104),
            new SoundWaterTypeEntry(Id: 10, SoundType: 0, SoundSubtype: 0, SoundId: 100),
            new SoundWaterTypeEntry(Id: 30, SoundType: 1, SoundSubtype: 8, SoundId: 108),
        ]);

        Assert.True(catalog.TryResolve(soundType: 0, soundSubtype: 4, out SoundWaterTypeEntry river));
        Assert.Equal(104, river.SoundId);
        Assert.True(catalog.TryResolve(soundType: 1, soundSubtype: 8, out SoundWaterTypeEntry ocean));
        Assert.Equal(108, ocean.SoundId);
        Assert.False(catalog.TryResolve(soundType: 0, soundSubtype: 8, out _));
    }

    [Fact]
    public void Constructor_DropsInvalidSoundRowsAndKeepsStableIdOrder()
    {
        SoundWaterTypeCatalog catalog = new(
        [
            new SoundWaterTypeEntry(Id: 2, SoundType: 0, SoundSubtype: 0, SoundId: 22),
            new SoundWaterTypeEntry(Id: 1, SoundType: 0, SoundSubtype: 0, SoundId: 11),
            new SoundWaterTypeEntry(Id: 3, SoundType: 0, SoundSubtype: 4, SoundId: 0),
            new SoundWaterTypeEntry(Id: 0, SoundType: 1, SoundSubtype: 0, SoundId: 33),
        ]);

        Assert.Equal(2, catalog.Entries.Count);
        Assert.Equal(1, catalog.Entries[0].Id);
        Assert.Equal(11, catalog.Entries[0].SoundId);
        Assert.Equal(22, catalog.Entries[1].SoundId);
        Assert.True(catalog.TryResolve(0, 0, out SoundWaterTypeEntry resolved));
        Assert.Equal(11, resolved.SoundId);
    }
}
