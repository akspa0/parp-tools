using WowViewer.Core.Audio;
using WowViewer.Core.World;

namespace WowViewer.Core.Tests;

public sealed class AreaIdentityLayoutTests
{
    [Theory]
    [InlineData("0.5.3.3368", AreaIdentityLayout.PackedAreaNumber)]
    [InlineData("3.3.5.12340", AreaIdentityLayout.DirectAreaId)]
    [InlineData("1.12.1.5875", AreaIdentityLayout.DirectAreaId)]
    [InlineData(null, AreaIdentityLayout.DirectAreaId)]
    public void FromBuildSelectsTheAreaIdentityLayout(string? build, AreaIdentityLayout expected)
    {
        Assert.Equal(expected, AreaIdentityLayoutResolver.FromBuild(build));
    }

    [Fact]
    public void DirectCatalogDoesNotInterpretAreaIdAsPackedAreaNumber()
    {
        const int directAreaId = 0x000A0001;
        const int packedLookingAreaNumber = directAreaId;

        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [directAreaId] = new(directAreaId, 0, 0, "Direct area", 0, 0, 111, 0, 0),
                [42] = new(42, 0, 0, "Packed-looking alias", 0, 0, 222, 0, 0, packedLookingAreaNumber, 0),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>(),
            AreaIdentityLayout.DirectAreaId);

        AlphaAreaAudioBinding? binding = catalog.TryResolve(directAreaId, continentId: 0);

        Assert.NotNull(binding);
        Assert.Equal(directAreaId, binding!.Area.Id);
        Assert.Equal(111, binding.Area.ZoneMusicId);
    }

    [Fact]
    public void DirectCatalogFollowsParentAreaIdWithoutUsingParentAreaNumber()
    {
        const int parentAreaId = 100;
        const int childAreaId = 101;
        const int packedParentAreaNumber = 0x000A0001;

        AlphaAreaAudioCatalog catalog = new(
            new Dictionary<int, AlphaAreaRecord>
            {
                [parentAreaId] = new(parentAreaId, 0, 0, "Direct parent", 0, 0, 333, 0, 0),
                [childAreaId] = new(childAreaId, 0, parentAreaId, "Direct child", 0, 0, 0, 0, 0, 0, packedParentAreaNumber),
                [200] = new(200, 0, 0, "Packed-looking parent", 0, 0, 444, 0, 0, packedParentAreaNumber, 0),
            },
            new Dictionary<int, AlphaAreaMidiAmbience>(),
            AreaIdentityLayout.DirectAreaId);

        AlphaAreaAudioBinding? binding = catalog.TryResolveWithParents(childAreaId, continentId: 0);

        Assert.NotNull(binding);
        Assert.Equal(parentAreaId, binding!.Area.Id);
        Assert.Equal(333, binding.Area.ZoneMusicId);
    }
}
