using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Composes one phase chunk onto a base chunk, channel by channel.
/// </summary>
/// <remarks>
/// Replaces the previous whole-chunk assignment in <c>MergePhaseTile</c>, which kept only
/// <c>Liquid</c> and therefore discarded every base field the phase chunk happened not to carry.
/// </remarks>
internal static class PhaseChunkMerger
{
    /// <summary>What this chunk actually carries, in <see cref="PhaseDataChannel"/> terms.</summary>
    public static PhaseDataChannel DescribePresence(TerrainChunkData chunk)
    {
        ArgumentNullException.ThrowIfNull(chunk);

        return PhaseCompositionPolicy.DescribePresence(
            hasHeights: chunk.Heights.Length > 0,
            hasNormals: chunk.Normals.Length > 0,
            textureLayerCount: chunk.Layers.Length,
            hasVertexColors: chunk.MccvColors is { Length: > 0 },
            hasHoleMask: chunk.HoleMask != 0,
            hasShadowMap: chunk.ShadowMap is { Length: > 0 },
            hasLiquid: chunk.Liquid is not null,
            hasAreaId: chunk.AreaId != 0);
    }

    /// <summary>
    /// Produce the composed chunk. Every channel in <paramref name="channelsToTake"/> comes from
    /// <paramref name="phaseChunk"/>; everything else is the base chunk's, unchanged.
    /// </summary>
    public static TerrainChunkData Merge(
        TerrainChunkData baseChunk,
        TerrainChunkData phaseChunk,
        PhaseDataChannel channelsToTake)
    {
        ArgumentNullException.ThrowIfNull(baseChunk);
        ArgumentNullException.ThrowIfNull(phaseChunk);

        bool TakeChannel(PhaseDataChannel channel) => (channelsToTake & channel) != 0;

        // Texture layers and their alpha maps are one channel: an alpha map indexes into the layer
        // list it was authored against, so taking one without the other produces paint applied to
        // the wrong texture.
        bool takeTexturing = TakeChannel(PhaseDataChannel.TextureLayers);

        return new TerrainChunkData
        {
            // Identity always comes from the base: this chunk occupies the base map's slot.
            McinIndex = baseChunk.McinIndex,
            TileX = baseChunk.TileX,
            TileY = baseChunk.TileY,
            ChunkX = baseChunk.ChunkX,
            ChunkY = baseChunk.ChunkY,
            WorldPosition = baseChunk.WorldPosition,

            Heights = TakeChannel(PhaseDataChannel.Heightmap) ? phaseChunk.Heights : baseChunk.Heights,
            Normals = TakeChannel(PhaseDataChannel.Normals) ? phaseChunk.Normals : baseChunk.Normals,
            HoleMask = TakeChannel(PhaseDataChannel.Holes) ? phaseChunk.HoleMask : baseChunk.HoleMask,

            Layers = takeTexturing ? phaseChunk.Layers : baseChunk.Layers,
            AlphaMaps = takeTexturing ? phaseChunk.AlphaMaps : baseChunk.AlphaMaps,
            AlphaSourceFlags = takeTexturing ? phaseChunk.AlphaSourceFlags : baseChunk.AlphaSourceFlags,

            MccvColors = TakeChannel(PhaseDataChannel.VertexColors) ? phaseChunk.MccvColors : baseChunk.MccvColors,
            ShadowMap = TakeChannel(PhaseDataChannel.Shadows) ? phaseChunk.ShadowMap : baseChunk.ShadowMap,
            AreaId = TakeChannel(PhaseDataChannel.AreaId) ? phaseChunk.AreaId : baseChunk.AreaId,

            // McnkFlags describe the chunk's own terrain payload, so they follow the heightmap.
            McnkFlags = TakeChannel(PhaseDataChannel.Heightmap) ? phaseChunk.McnkFlags : baseChunk.McnkFlags,

            // Reference lists belong with the placements they index.
            McrdReferences = TakeChannel(PhaseDataChannel.Doodads) ? phaseChunk.McrdReferences : baseChunk.McrdReferences,
            McrwReferences = TakeChannel(PhaseDataChannel.WorldObjects) ? phaseChunk.McrwReferences : baseChunk.McrwReferences,

            Liquid = TakeChannel(PhaseDataChannel.Liquid) ? phaseChunk.Liquid : baseChunk.Liquid,
        };
    }
}
