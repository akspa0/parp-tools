using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 203/207 — phase composition expressed over the harvest signal names, which is the unit the
/// Python store ingests. Storage stays Python's job; this only decides which arrays a composed tile
/// carries.
/// </summary>
public sealed class PhaseSignalChannelMapTests
{
    [Theory]
    [InlineData("height_257", PhaseDataChannel.Heightmap)]
    [InlineData("mcnr_normal_xyz", PhaseDataChannel.Normals)]
    [InlineData("mcal_alpha_pack_256", PhaseDataChannel.TextureLayers)]
    [InlineData("mccv_rgb", PhaseDataChannel.VertexColors)]
    [InlineData("mh2o_surface_height", PhaseDataChannel.Liquid)]
    [InlineData("mddf_placements", PhaseDataChannel.Doodads)]
    [InlineData("modf_placements", PhaseDataChannel.WorldObjects)]
    public void ChannelFor_ClassifiesTheSignalsTheStreamEmits(string signal, PhaseDataChannel expected)
        => Assert.Equal(expected, PhaseSignalChannelMap.ChannelFor(signal));

    [Fact]
    public void ShouldTakeFromPhase_HonoursTheChannelSelection()
    {
        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("height_257", PhaseDataChannel.Heightmap, out _));
        Assert.False(PhaseSignalChannelMap.ShouldTakeFromPhase("height_257", PhaseDataChannel.Objects, out _));
    }

    [Fact]
    public void ShouldTakeFromPhase_TextureLayersAndTheirAlphaMoveTogether()
    {
        // An alpha map indexes the layer list it was authored against, so taking one without the
        // other paints the phase's alpha onto the base map's textures.
        const PhaseDataChannel take = PhaseDataChannel.TextureLayers;

        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("mcly_texture_ids", take, out _));
        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("mcal_alpha_pack_256", take, out _));
        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("mcly_layer_mask", take, out _));
    }

    [Fact]
    public void ShouldTakeFromPhase_ReferenceListsFollowTheirPlacements()
    {
        // MCRF/MCRD indices point into the placement table; keeping base indices alongside phase
        // placements would dereference the wrong objects.
        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("mcrf_doodad_ref_indices", PhaseDataChannel.Doodads, out _));
        Assert.True(PhaseSignalChannelMap.ShouldTakeFromPhase("mcrw_ref_counts_16", PhaseDataChannel.WorldObjects, out _));
        Assert.False(PhaseSignalChannelMap.ShouldTakeFromPhase("mcrf_doodad_ref_indices", PhaseDataChannel.WorldObjects, out _));
    }

    [Fact]
    public void IdentitySignals_NeverComeFromAPhaseLayer()
    {
        // Even when every data channel came from the donor, the composed tile is still the base
        // tile. For an offset layer the donor's coordinates are different outright.
        foreach (string signal in new[] { "tile_x", "tile_y", "map_name", "build" })
        {
            Assert.True(PhaseSignalChannelMap.IsBaseOnly(signal));
            Assert.False(PhaseSignalChannelMap.ShouldTakeFromPhase(signal, PhaseDataChannel.All, out bool unclassified));
            Assert.False(unclassified);
        }
    }

    [Fact]
    public void UnclassifiedSignal_IsReported_NotSilentlyDefaulted()
    {
        // The failure mode this guard exists for: a signal added to the emitter but not to the map
        // would otherwise pick a side silently and produce a composed tile that is wrong with no
        // indication anywhere.
        bool taken = PhaseSignalChannelMap.ShouldTakeFromPhase("some_new_signal_v3", PhaseDataChannel.All, out bool unclassified);

        Assert.True(unclassified);
        Assert.False(taken);
        Assert.Null(PhaseSignalChannelMap.ChannelFor("some_new_signal_v3"));
    }

    [Fact]
    public void EveryKnownSignal_ResolvesToExactlyOneChannel()
    {
        foreach (string signal in PhaseSignalChannelMap.KnownSignals)
        {
            PhaseDataChannel? channel = PhaseSignalChannelMap.ChannelFor(signal);
            Assert.NotNull(channel);
            Assert.NotEqual(PhaseDataChannel.None, channel!.Value);

            // A signal owned by two channels would be taken or dropped inconsistently depending on
            // which channel the user unticked.
            int bits = System.Numerics.BitOperations.PopCount((uint)channel.Value);
            Assert.True(bits == 1, $"Signal '{signal}' maps to {bits} channels ({channel.Value}); it must map to exactly one.");
        }
    }

    [Fact]
    public void KnownSignals_CoverEveryChannelThePanelOffers()
    {
        // If the panel offers a checkbox for a channel no signal belongs to, that checkbox does
        // nothing and the user has no way to tell.
        PhaseDataChannel covered = PhaseDataChannel.None;
        foreach (string signal in PhaseSignalChannelMap.KnownSignals)
            covered |= PhaseSignalChannelMap.ChannelFor(signal)!.Value;

        Assert.Equal(PhaseDataChannel.All, covered);
    }
}
