using System;
using WowViewer.Core.Maps;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Viewport chunk editing for donor placements. When enabled, clicking a terrain chunk in the
/// 3D view toggles that donor chunk's inclusion in the placement that claims the target tile.
/// This is the in-renderer counterpart to the donor picker's 16×16 chunk grid.
/// </summary>
internal sealed class DonorChunkEditService
{
    private readonly IViewerAppHost _host;

    public DonorChunkEditService(IViewerAppHost host)
    {
        _host = host;
    }

    public bool Enabled { get; set; }
    public string? StatusMessage { get; private set; }

    public bool TryToggleAt(TerrainRenderer.TerrainChunkInfo info)
    {
        if (!Enabled)
            return false;

        TerrainManager? terrain = _host.TerrainManager;
        if (terrain == null)
            return false;

        for (int layerIndex = terrain.PhaseLayers.Count - 1; layerIndex >= 0; layerIndex--)
        {
            PhaseLayerSettings layer = terrain.PhaseLayers[layerIndex];
            if (!layer.Enabled || layer.Channels == PhaseDataChannel.None)
                continue;

            for (int placementIndex = layer.TilePlacements.Count - 1; placementIndex >= 0; placementIndex--)
            {
                PhaseTilePlacement placement = layer.TilePlacements[placementIndex];
                if (!placement.Contributes)
                    continue;

                int targetX = layer.UsePlacedTilesOnly
                    ? placement.TargetTileX + layer.TileOffsetX
                    : placement.TargetTileX;
                int targetY = layer.UsePlacedTilesOnly
                    ? placement.TargetTileY + layer.TileOffsetY
                    : placement.TargetTileY;
                if (targetX != info.TileX || targetY != info.TileY)
                    continue;

                IReadOnlyList<TileTransformKind> transforms = PhaseCompositionPolicy.ComposeTileTransforms(layer);
                (int donorChunkX, int donorChunkY) = PhaseCompositionPolicy.InverseTransformChunkSlot(
                    info.ChunkX, info.ChunkY, transforms);

                PhaseChunkMask mask = placement.ChunkMask ?? PhaseChunkMask.All;
                bool included = !mask.IsIncluded(donorChunkX, donorChunkY);
                mask = mask.With(donorChunkX, donorChunkY, included);
                layer.TilePlacements[placementIndex] = placement with
                {
                    ChunkMask = mask.IsAllIncluded ? null : mask,
                };

                terrain.RefreshPhaseLayers();
                StatusMessage = $"Donor chunk {donorChunkX:D2}_{donorChunkY:D2} "
                    + $"{(included ? "included" : "excluded")} for {layer.MapName}.";
                return true;
            }
        }

        StatusMessage = "No donor placement claims this target chunk.";
        return false;
    }
}
