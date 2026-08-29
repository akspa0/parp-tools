using System.Numerics;
using WowViewer.Core.Editor.Procedural;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class SemanticAssetClassifierTests
{
    [Theory]
    [InlineData("item/objectcomponents/weapon/sword_1h_short_a_01.mdx", AssetSemanticArchetype.Weapon, CellDensityTier.Micro, 3.0f)]
    [InlineData("item/objectcomponents/weapon/dagger_1h_small_02.mdx", AssetSemanticArchetype.Weapon, CellDensityTier.Micro, 4.0f)]
    [InlineData("item/objectcomponents/shield/shield_round_b_01.mdx", AssetSemanticArchetype.Weapon, CellDensityTier.Micro, 3.0f)]
    [InlineData("item/objectcomponents/armor/helm_leather_b_01.mdx", AssetSemanticArchetype.Armor, CellDensityTier.Micro, 3.5f)]
    [InlineData("spells/frost_crystal_missile.mdx", AssetSemanticArchetype.SpellEffect, CellDensityTier.Micro, 3.0f)]
    [InlineData("creature/rat/rat.mdx", AssetSemanticArchetype.SmallCritter, CellDensityTier.Small, 2.5f)]
    [InlineData("creature/horse/horse.mdx", AssetSemanticArchetype.Mount, CellDensityTier.Medium, 2.0f)]
    [InlineData("character/human/male/humanmale.mdx", AssetSemanticArchetype.Humanoid, CellDensityTier.Medium, 2.0f)]
    [InlineData("creature/dragon/blackdragon.mdx", AssetSemanticArchetype.GiantBoss, CellDensityTier.Large, 1.0f)]
    [InlineData("world/generic/doodads/lights/candle_sm.mdx", AssetSemanticArchetype.SmallDoodad, CellDensityTier.Small, 2.5f)]
    [InlineData("world/generic/doodads/statues/monument_large.mdx", AssetSemanticArchetype.LargeDoodad, CellDensityTier.Large, 1.5f)]
    [InlineData("world/wmo/dungeon/keep/keep01.wmo", AssetSemanticArchetype.Structure, CellDensityTier.Large, 1.0f)]
    public void Classify_CorrectlyIdentifiesArchetypeAndRecommendedScale(
        string path,
        AssetSemanticArchetype expectedArchetype,
        CellDensityTier expectedTier,
        float expectedScale)
    {
        Vector3 boundsMin = new(-1f, -1f, -1f);
        Vector3 boundsMax = new(1f, 1f, 1f);
        RosettaAssetKind kind = path.EndsWith(".wmo") ? RosettaAssetKind.WorldModel : RosettaAssetKind.Model;

        var result = SemanticAssetClassifier.Classify(path, boundsMin, boundsMax, kind);

        Assert.Equal(expectedArchetype, result.Archetype);
        Assert.Equal(expectedTier, result.RecommendedDensityTier);
        Assert.Equal(expectedScale, result.RecommendedScale);
    }

    [Fact]
    public void Classify_DisambiguatesDungeonSetPrefixesFromSizePrefixes()
    {
        // "sm_" under dungeon set should be Scarlet Monastery, not a small doodad
        string dungeonSmPath = "world/wmo/dungeon/sm_monastery/sm_altar01.wmo";
        var resultDungeon = SemanticAssetClassifier.Classify(
            dungeonSmPath, new Vector3(-20f, -20f, 0f), new Vector3(20f, 20f, 15f), RosettaAssetKind.WorldModel);

        Assert.True(resultDungeon.IsDungeonSetAsset);
        Assert.Equal("sm", resultDungeon.DungeonPrefix);
        Assert.Contains("Scarlet Monastery", resultDungeon.SuggestedPavilion);

        // "sm_" on a generic prop in doodads should be small doodad
        string genericPropSmPath = "world/generic/doodads/sm_torch.mdx";
        var resultProp = SemanticAssetClassifier.Classify(
            genericPropSmPath, new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, 0.5f, 1.5f), RosettaAssetKind.Model);

        Assert.Equal(AssetSemanticArchetype.SmallDoodad, resultProp.Archetype);
        Assert.Equal(CellDensityTier.Small, resultProp.RecommendedDensityTier);
    }
}
