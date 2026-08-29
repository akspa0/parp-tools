using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Procedural;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class AdaptiveLayoutPackerTests
{
    private static RosettaAssetEntry ItemAsset(string path, float size = 1f) => new(
        path,
        RosettaAssetKind.Model,
        new Vector3(-size / 2f, -size / 2f, -size / 2f),
        new Vector3(size / 2f, size / 2f, size / 2f));

    [Fact]
    public void PackExhibits_PacksMicroAssetsDenselyInSmallCells()
    {
        var weapons = new List<RosettaAssetEntry>();
        for (int i = 0; i < 64; i++)
        {
            weapons.Add(ItemAsset($"item/objectcomponents/weapon/sword_1h_{i:D2}.mdx", 1.2f));
        }

        var options = new AdaptiveLayoutOptions(Density: DensityPreset.Compact);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(weapons, options);

        Assert.Equal(64, placements.Count);

        // All 64 weapons should pack into 16.66m micro cells, fitting in a single tile (which has up to 1024 16.66m cells)!
        Assert.All(placements, p => Assert.Equal(16.66666f, p.CellSize, 2));
        Assert.All(placements, p => Assert.True(p.ComputedScale >= 2.5f));

        int uniqueTiles = placements.Select(p => (p.TileX, p.TileY)).Distinct().Count();
        Assert.Equal(1, uniqueTiles);
    }

    [Fact]
    public void PackExhibits_AssignsDistinctTiersBasedOnArchetype()
    {
        var assets = new List<RosettaAssetEntry>
        {
            ItemAsset("item/objectcomponents/weapon/dagger_01.mdx", 0.6f),
            ItemAsset("creature/rat/rat.mdx", 0.8f),
            ItemAsset("character/human/male/humanmale.mdx", 2.2f),
            ItemAsset("creature/dragon/blackdragon.mdx", 30.0f)
        };

        var options = new AdaptiveLayoutOptions(Density: DensityPreset.Balanced);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, options);

        Assert.Equal(4, placements.Count);

        var dagger = placements.Single(p => p.Asset.AssetPath.Contains("dagger"));
        var rat = placements.Single(p => p.Asset.AssetPath.Contains("rat"));
        var human = placements.Single(p => p.Asset.AssetPath.Contains("humanmale"));
        var dragon = placements.Single(p => p.Asset.AssetPath.Contains("blackdragon"));

        Assert.True(dagger.CellSize <= 33.33334f);
        Assert.True(dagger.ComputedScale >= 3.0f);

        Assert.True(rat.CellSize <= 66.66667f);
        Assert.True(rat.ComputedScale >= 2.0f);

        Assert.True(human.CellSize >= 66.66666f);
        Assert.True(dragon.CellSize >= 133.33333f);
        Assert.Equal(1.0f, dragon.ComputedScale);
    }

    [Fact]
    public void PackExhibits_PreservesNoCollisionOverlaps()
    {
        var assets = new List<RosettaAssetEntry>();
        for (int i = 0; i < 20; i++)
        {
            assets.Add(ItemAsset($"creature/critters/critter_{i:D2}.mdx", 1.5f));
        }

        var options = new AdaptiveLayoutOptions(Density: DensityPreset.Balanced);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, options);

        // Verify no two exhibits share the same exact cell on the same tile
        var cellKeys = placements.Select(p => (p.TileX, p.TileY, MathF.Round(p.CellU, 1), MathF.Round(p.CellV, 1))).ToList();
        Assert.Equal(cellKeys.Count, cellKeys.Distinct().Count());
    }
}
