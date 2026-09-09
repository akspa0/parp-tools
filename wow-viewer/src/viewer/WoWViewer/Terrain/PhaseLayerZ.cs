using System.Numerics;
using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Operator directive 2026-09-09: per-layer world-Z adjustment for composed phase content.
/// A cut tile (e.g. Teldrassil out of the Kalidar map) has to be raised or lowered to meet the
/// tiles it lands beside; <c>z' = z * ZScale + ZOffset</c> applies the same affine transform to
/// the terrain heights, the liquid surfaces, and the placement Z so the tile moves as one object.
/// </summary>
/// <remarks>
/// Viewer heights are absolute world-space Z (the mesh builders sample
/// <c>chunk.Heights[i]</c> directly), so the affine form is valid for both the Alpha and the
/// Standard adapters.
/// </remarks>
internal static class PhaseLayerZ
{
    public static bool IsIdentity(PhaseLayerSettings layer)
        => layer.ZScale == 1f && layer.ZOffset == 0f;

    /// <summary>Applies the layer's Z transform to a phase tile in place, before the merge.</summary>
    public static void Apply(PhaseLayerSettings layer, TileLoadResult phase)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(phase);

        if (IsIdentity(layer))
            return;

        foreach (TerrainChunkData chunk in phase.Chunks)
            ApplyToChunk(chunk, layer);

        for (int i = 0; i < phase.MddfPlacements.Count; i++)
        {
            MddfPlacement placement = phase.MddfPlacements[i];
            placement.Position = TransformZ(placement.Position, layer);
            phase.MddfPlacements[i] = placement;
        }

        for (int i = 0; i < phase.ModfPlacements.Count; i++)
        {
            ModfPlacement placement = phase.ModfPlacements[i];
            placement.Position = TransformZ(placement.Position, layer);
            phase.ModfPlacements[i] = placement;
        }
    }

    private static void ApplyToChunk(TerrainChunkData chunk, PhaseLayerSettings layer)
    {
        if (chunk.Heights is { Length: > 0 } heights)
        {
            for (int i = 0; i < heights.Length; i++)
                heights[i] = (heights[i] * layer.ZScale) + layer.ZOffset;
        }

        // LiquidChunkData's Min/Max are init-only; the visible surface is the mutable Heights
        // array, which carries the same world-Z convention as the terrain heights.
        if (chunk.Liquid?.Heights is { Length: > 0 } liquidHeights)
        {
            for (int i = 0; i < liquidHeights.Length; i++)
                liquidHeights[i] = (liquidHeights[i] * layer.ZScale) + layer.ZOffset;
        }
    }

    private static Vector3 TransformZ(Vector3 position, PhaseLayerSettings layer)
        => new(position.X, position.Y, (position.Z * layer.ZScale) + layer.ZOffset);
}
