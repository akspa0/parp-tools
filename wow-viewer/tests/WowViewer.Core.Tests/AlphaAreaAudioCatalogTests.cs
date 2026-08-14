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
}
