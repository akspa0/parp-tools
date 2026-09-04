using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 203. These cover the two defects the phase merge previously had: a whole-chunk replacement
/// that destroyed every base field the phase did not carry, and placements appended unconditionally
/// so a restaging phase produced both the old set and the new one.
/// </summary>
public sealed class PhaseCompositionPolicyTests
{
    [Fact]
    public void ResolveChannelsToTake_PresenceGated_DropsChannelsThePhaseDoesNotCarry()
    {
        // The blank-plate case: the phase ships terrain but no texturing of its own.
        PhaseDataChannel carries = PhaseDataChannel.Heightmap | PhaseDataChannel.Normals;

        PhaseDataChannel taken = PhaseCompositionPolicy.ResolveChannelsToTake(
            PhaseDataChannel.All, carries, onlyTakeWhatThePhaseCarries: true);

        Assert.Equal(carries, taken);
        Assert.Equal(PhaseDataChannel.None, taken & PhaseDataChannel.TextureLayers);
    }

    [Fact]
    public void ResolveChannelsToTake_PresenceGateOff_LetsAnEmptyPhaseClearAChannel()
    {
        // Deliberately destructive, and the only way to express "this phase removes the trees".
        PhaseDataChannel taken = PhaseCompositionPolicy.ResolveChannelsToTake(
            PhaseDataChannel.All, PhaseDataChannel.None, onlyTakeWhatThePhaseCarries: false);

        Assert.Equal(PhaseDataChannel.All, taken);
    }

    [Fact]
    public void ResolveChannelsToTake_NeverExceedsWhatTheUserEnabled()
    {
        PhaseDataChannel taken = PhaseCompositionPolicy.ResolveChannelsToTake(
            PhaseDataChannel.Objects, PhaseDataChannel.All, onlyTakeWhatThePhaseCarries: true);

        Assert.Equal(PhaseDataChannel.Objects, taken);
        Assert.Equal(PhaseDataChannel.None, taken & PhaseDataChannel.Heightmap);
    }

    [Fact]
    public void ResolveChannelsToTake_UncheckedLayerTakesNothing()
    {
        Assert.Equal(
            PhaseDataChannel.None,
            PhaseCompositionPolicy.ResolveChannelsToTake(PhaseDataChannel.None, PhaseDataChannel.All, true));
    }

    [Fact]
    public void DescribePresence_EmptyChunkCarriesNothing()
    {
        PhaseDataChannel present = PhaseCompositionPolicy.DescribePresence(
            hasHeights: false,
            hasNormals: false,
            textureLayerCount: 0,
            hasVertexColors: false,
            hasHoleMask: false,
            hasShadowMap: false,
            hasLiquid: false,
            hasAreaId: false);

        Assert.Equal(PhaseDataChannel.None, present);
    }

    [Fact]
    public void DescribePresence_TerrainWithoutTexturing_ReportsExactlyThat()
    {
        // This is the shape that used to blank out a base chunk's texturing.
        PhaseDataChannel present = PhaseCompositionPolicy.DescribePresence(
            hasHeights: true,
            hasNormals: true,
            textureLayerCount: 0,
            hasVertexColors: false,
            hasHoleMask: false,
            hasShadowMap: false,
            hasLiquid: false,
            hasAreaId: false);

        Assert.Equal(PhaseDataChannel.Heightmap | PhaseDataChannel.Normals, present);
    }

    [Fact]
    public void DescribePresence_FullChunkCarriesEveryChunkLevelChannel()
    {
        PhaseDataChannel present = PhaseCompositionPolicy.DescribePresence(
            hasHeights: true,
            hasNormals: true,
            textureLayerCount: 4,
            hasVertexColors: true,
            hasHoleMask: true,
            hasShadowMap: true,
            hasLiquid: true,
            hasAreaId: true);

        // Placements are tile-level, not chunk-level, so they are not part of chunk presence.
        PhaseDataChannel expected = PhaseDataChannel.All & ~PhaseDataChannel.Objects;
        Assert.Equal(expected, present);
    }

    [Fact]
    public void PhaseOwnsPlacements_PhaseShippingObjects_TakesOwnership()
    {
        Assert.True(PhaseCompositionPolicy.PhaseOwnsPlacements(
            PhaseDataChannel.All, PhaseDataChannel.Doodads, phasePlacementCount: 12));
    }

    [Fact]
    public void PhaseOwnsPlacements_PhaseShippingNoObjects_LeavesTheBaseMapsAlone()
    {
        // The regression that produced "missing trees": a phase with no object companion must not
        // be treated as an authoritative empty set.
        Assert.False(PhaseCompositionPolicy.PhaseOwnsPlacements(
            PhaseDataChannel.All, PhaseDataChannel.Doodads, phasePlacementCount: 0));
    }

    [Fact]
    public void PhaseOwnsPlacements_ChannelDisabled_NeverTakesOwnership()
    {
        Assert.False(PhaseCompositionPolicy.PhaseOwnsPlacements(
            PhaseDataChannel.Terrain, PhaseDataChannel.Doodads, phasePlacementCount: 12));
    }

    [Fact]
    public void ChannelGroups_PartitionTheWayThePanelPresentsThem()
    {
        Assert.Equal(
            PhaseDataChannel.All,
            PhaseDataChannel.Terrain | PhaseDataChannel.Texturing | PhaseDataChannel.Liquid
                | PhaseDataChannel.Objects | PhaseDataChannel.AreaId);

        Assert.Equal(PhaseDataChannel.None, PhaseDataChannel.Terrain & PhaseDataChannel.Texturing);
        Assert.Equal(PhaseDataChannel.None, PhaseDataChannel.Terrain & PhaseDataChannel.Objects);
        Assert.Equal(PhaseDataChannel.None, PhaseDataChannel.Texturing & PhaseDataChannel.Objects);
    }

    [Fact]
    public void MultipleLayers_LaterLayerWinsOnlyOnChannelsItActuallyContributes()
    {
        // Two phases stacked: the first restages terrain, the second only moves objects.
        PhaseDataChannel first = PhaseCompositionPolicy.ResolveChannelsToTake(
            PhaseDataChannel.All,
            PhaseDataChannel.Heightmap | PhaseDataChannel.TextureLayers,
            onlyTakeWhatThePhaseCarries: true);

        PhaseDataChannel second = PhaseCompositionPolicy.ResolveChannelsToTake(
            PhaseDataChannel.All,
            PhaseDataChannel.None,
            onlyTakeWhatThePhaseCarries: true);

        Assert.Equal(PhaseDataChannel.Heightmap | PhaseDataChannel.TextureLayers, first);

        // The second layer contributes no chunk channels, so the first layer's terrain survives it.
        Assert.Equal(PhaseDataChannel.None, second);
    }

    [Fact]
    public void LayerSettings_DefaultsAreTheSafeComposition()
    {
        PhaseLayerSettings layer = new() { MapName = "GilneasPhase1" };

        Assert.True(layer.Enabled);
        Assert.Equal(PhaseDataChannel.All, layer.Channels);
        Assert.True(layer.OnlyTakeWhatThePhaseCarries);
    }

    [Fact]
    public void LayerSettings_CloneIsIndependent()
    {
        PhaseLayerSettings layer = new() { MapName = "GilneasPhase1" };
        PhaseLayerSettings copy = layer.Clone();
        copy.Channels = PhaseDataChannel.Objects;
        copy.Enabled = false;

        Assert.Equal(PhaseDataChannel.All, layer.Channels);
        Assert.True(layer.Enabled);
        Assert.Equal("GilneasPhase1", copy.MapName);
    }

    [Fact]
    public void TileOffsetToWorldTranslation_IsNegativeInBothAxes()
    {
        // Renderer X and Y both DECREASE as tile X/Y increase
        // (worldX = MapOrigin - tileX * ChunkSize - ...), so a positive tile shift translates negative.
        // The span passed in MUST be ONE ADT tile in the 64x64 grid: WoWConstants.ChunkSize
        // (533.33 — misnamed, but it is the ADT span; TileSize = 16 ADTs would overshoot 16x).
        const float adtTileSpan = 533.33333f;
        (float x, float y) = PhaseCompositionPolicy.TileOffsetToWorldTranslation(1, 2, adtTileSpan);

        Assert.Equal(-adtTileSpan, x, 3);
        Assert.Equal(-2f * adtTileSpan, y, 3);
    }

    [Fact]
    public void TileOffsetToWorldTranslation_ZeroOffsetDoesNotMoveAnything()
    {
        (float x, float y) = PhaseCompositionPolicy.TileOffsetToWorldTranslation(0, 0, 533.33333f);

        Assert.Equal(0f, x);
        Assert.Equal(0f, y);
    }

    [Fact]
    public void LayerSettings_TileOffsetDefaultsToNoShift()
    {
        PhaseLayerSettings layer = new() { MapName = "RazorfenDowns" };

        Assert.Equal(0, layer.TileOffsetX);
        Assert.Equal(0, layer.TileOffsetY);
        Assert.False(layer.HasTileOffset);
    }

    [Fact]
    public void LayerSettings_CloneCarriesTheTileOffset()
    {
        PhaseLayerSettings layer = new() { MapName = "RazorfenDowns", TileOffsetX = -3, TileOffsetY = 7 };
        PhaseLayerSettings copy = layer.Clone();

        Assert.Equal(-3, copy.TileOffsetX);
        Assert.Equal(7, copy.TileOffsetY);
        Assert.True(copy.HasTileOffset);
    }
}
