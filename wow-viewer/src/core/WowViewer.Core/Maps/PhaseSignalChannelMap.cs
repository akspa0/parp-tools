namespace WowViewer.Core.Maps;

/// <summary>
/// Maps a harvest signal name to the phase channel that owns it.
/// </summary>
/// <remarks>
/// <para>
/// Phase composition is expressed over the <b>named arrays</b> the harvest stream already emits
/// (<c>height_257</c>, <c>mcnr_normal_xyz</c>, <c>mcal_alpha_pack_256</c>, …) rather than over
/// <see cref="TerrainTileTensorPack"/> objects. Three reasons:
/// </para>
/// <list type="number">
/// <item>The pack is a <c>sealed class</c> with <c>init</c> properties, so a field-by-field copy
/// would <b>silently drop</b> any array added to it later — exactly the class of defect this project
/// keeps paying for.</item>
/// <item>Named arrays are what <c>RawArraySerializer.SerializeGeneric</c> writes and what the Python
/// store ingests, so composing at that level means no new format and no second representation.</item>
/// <item>A name-keyed map can be <b>tested for completeness</b>: an emitted signal with no channel
/// fails the guard rather than quietly defaulting.</item>
/// </list>
/// <para>
/// Storage remains Python's job. This only decides <em>which</em> arrays a composed tile carries.
/// </para>
/// </remarks>
public static class PhaseSignalChannelMap
{
    private static readonly Dictionary<string, PhaseDataChannel> ChannelBySignal = new(StringComparer.Ordinal)
    {
        // Terrain geometry.
        ["height_257"] = PhaseDataChannel.Heightmap,
        ["height_65"] = PhaseDataChannel.Heightmap,
        ["height_17"] = PhaseDataChannel.Heightmap,
        ["mcnk_flags_16"] = PhaseDataChannel.Heightmap,
        ["mcnr_normal_xyz"] = PhaseDataChannel.Normals,
        ["mcnr_mask_257"] = PhaseDataChannel.Normals,
        ["hole_mask_16"] = PhaseDataChannel.Holes,

        // Texturing.
        ["mcly_texture_ids"] = PhaseDataChannel.TextureLayers,
        ["mcly_layer_mask"] = PhaseDataChannel.TextureLayers,
        ["mcmt_material_ids"] = PhaseDataChannel.TextureLayers,
        ["mcal_alpha_pack"] = PhaseDataChannel.TextureLayers,
        ["mcal_alpha_pack_256"] = PhaseDataChannel.TextureLayers,
        ["mccv_rgb"] = PhaseDataChannel.VertexColors,
        ["mclv_lighting_bytes"] = PhaseDataChannel.VertexColors,
        ["mcsh_shadow_mask_256"] = PhaseDataChannel.Shadows,

        // Liquid, in all four representations the pack carries.
        ["mh2o_surface_height"] = PhaseDataChannel.Liquid,
        ["mh2o_depth"] = PhaseDataChannel.Liquid,
        ["mh2o_type_mask"] = PhaseDataChannel.Liquid,
        ["mh2o_presence_mask"] = PhaseDataChannel.Liquid,
        ["mclq_surface_height"] = PhaseDataChannel.Liquid,
        ["mclq_type_mask"] = PhaseDataChannel.Liquid,
        ["mclq_presence_mask"] = PhaseDataChannel.Liquid,
        ["wl_liquid_mask"] = PhaseDataChannel.Liquid,
        ["wl_liquid_height"] = PhaseDataChannel.Liquid,
        ["unified_liquid_mask"] = PhaseDataChannel.Liquid,
        ["unified_liquid_height"] = PhaseDataChannel.Liquid,
        ["liquid_basic_type_257"] = PhaseDataChannel.Liquid,

        // Placements and the per-chunk reference lists that index them.
        ["mddf_placements"] = PhaseDataChannel.Doodads,
        ["mcrf_doodad_ref_counts_16"] = PhaseDataChannel.Doodads,
        ["mcrf_doodad_ref_indices"] = PhaseDataChannel.Doodads,
        ["mcrd_ref_counts_16"] = PhaseDataChannel.Doodads,
        ["mcrd_ref_indices"] = PhaseDataChannel.Doodads,
        ["modf_placements"] = PhaseDataChannel.WorldObjects,
        ["mcrf_wmo_ref_counts_16"] = PhaseDataChannel.WorldObjects,
        ["mcrf_wmo_ref_indices"] = PhaseDataChannel.WorldObjects,
        ["mcrw_ref_counts_16"] = PhaseDataChannel.WorldObjects,
        ["mcrw_ref_indices"] = PhaseDataChannel.WorldObjects,

        ["area_id_16"] = PhaseDataChannel.AreaId,
    };

    /// <summary>
    /// Signals that always come from the base tile regardless of channel selection.
    /// </summary>
    /// <remarks>
    /// Identity and provenance describe <em>which</em> tile this is. Taking them from a phase layer
    /// would relabel the composed tile as the donor's, which is wrong even when every data channel
    /// came from that donor — and is doubly wrong for an offset layer, whose tile coordinates differ.
    /// </remarks>
    private static readonly HashSet<string> AlwaysFromBase = new(StringComparer.Ordinal)
    {
        "tile_x",
        "tile_y",
        "map_name",
        "build",
        "tile_id",
        "source_path",
    };

    /// <summary>Every signal name this map classifies.</summary>
    public static IReadOnlyCollection<string> KnownSignals => ChannelBySignal.Keys;

    /// <summary>True when the signal is identity/provenance and never comes from a phase layer.</summary>
    public static bool IsBaseOnly(string signalName) => AlwaysFromBase.Contains(signalName);

    /// <summary>
    /// The channel that owns <paramref name="signalName"/>, or null when it is unclassified.
    /// </summary>
    /// <remarks>
    /// Null is deliberately <b>not</b> "take from base": an unclassified signal is a gap in this map,
    /// and <see cref="ShouldTakeFromPhase"/> reports it rather than guessing a default that would
    /// silently pick a side.
    /// </remarks>
    public static PhaseDataChannel? ChannelFor(string signalName)
        => ChannelBySignal.TryGetValue(signalName, out PhaseDataChannel channel) ? channel : null;

    /// <summary>
    /// Whether a composed tile takes <paramref name="signalName"/> from the phase layer.
    /// </summary>
    /// <param name="signalName">The harvest signal name.</param>
    /// <param name="channelsToTake">Channels this layer contributes, already presence-gated.</param>
    /// <param name="unclassified">Set when the signal has no channel; the caller must report it.</param>
    public static bool ShouldTakeFromPhase(
        string signalName,
        PhaseDataChannel channelsToTake,
        out bool unclassified)
    {
        unclassified = false;

        if (IsBaseOnly(signalName))
            return false;

        PhaseDataChannel? channel = ChannelFor(signalName);
        if (channel is null)
        {
            // Never guess. An unmapped signal means this map is out of date with the emitter, and
            // silently defaulting either way produces a composed tile that is wrong in a way nothing
            // reports.
            unclassified = true;
            return false;
        }

        return (channelsToTake & channel.Value) != 0;
    }
}
