using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Wdl;

namespace WowViewer.Core.Runtime.World;

public static class WorldRenderCompositionBuilder
{
    public static WorldRenderCompositionFrame Build(
        WorldFramePassOptions passOptions,
        WorldWdlTileData wdlTileData,
        WorldTerrainTileData terrainTileData,
        WorldLiquidTileData liquidTileData,
        int wmoSourceCount,
        int mdxSourceCount,
        WorldRenderFrameStats stats,
        int skyboxBackdropSourceCount = 0)
    {
        ArgumentNullException.ThrowIfNull(wdlTileData);
        ArgumentNullException.ThrowIfNull(terrainTileData);
        ArgumentNullException.ThrowIfNull(liquidTileData);
        ArgumentOutOfRangeException.ThrowIfNegative(wmoSourceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(mdxSourceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(skyboxBackdropSourceCount);

        WorldRenderLayerState[] layers =
        [
            new(
                WorldRenderLayerKind.Sky,
                "Spherical Sky",
                passOptions.SkyVisible,
                passOptions.SkyVisible,
                passOptions.SkyVisible ? 1 : 0,
                passOptions.SkyVisible ? 1 : 0,
                "Procedural camera-centered backdrop until client sky records are decoded."),
            new(
                WorldRenderLayerKind.SkyboxBackdrop,
                "Skybox Backdrop",
                passOptions.SkyVisible,
                passOptions.SkyVisible && skyboxBackdropSourceCount > 0,
                skyboxBackdropSourceCount,
                stats.SkyboxBackdrop.SubmittedCount,
                stats.SkyboxBackdrop.SubmittedCount > 0
                    ? "Backdrop-like model placements feed the current procedural spherical backdrop layer; decoded client skybox geometry is still future work."
                    : !passOptions.SkyVisible
                    ? "Backdrop-like model placements were classified but sky rendering is disabled for this frame."
                    : skyboxBackdropSourceCount > 0
                    ? "Backdrop-like model placements were classified; procedural backdrop submission is still pending for this frame."
                    : "Reserved for placed or zone-selected backdrop models."),
            new(
                WorldRenderLayerKind.Wdl,
                "Far Terrain (WDL)",
                passOptions.WdlVisible,
                wdlTileData.HasData,
                wdlTileData.HasData ? 1 : 0,
                stats.Wdl.SubmittedCount,
                wdlTileData.HasData ? "Low-detail terrain source found for the selected tile." : "No WDL tile data submitted for this frame."),
            new(
                WorldRenderLayerKind.Terrain,
                "ADT Terrain Quilt",
                passOptions.TerrainVisible,
                terrainTileData.ChunkCount > 0,
                terrainTileData.ChunkCount,
                stats.Terrain.SubmittedCount,
                "Rigid Z-axis ADT chunk mesh."),
            new(
                WorldRenderLayerKind.Liquid,
                "Liquid",
                passOptions.LiquidVisible,
                liquidTileData.ActiveChunkCount > 0,
                liquidTileData.ActiveChunkCount,
                stats.Liquid.SubmittedCount,
                liquidTileData.ActiveChunkCount > 0 ? "Tile carries liquid chunks." : "No active liquid chunks for this tile."),
            new(
                WorldRenderLayerKind.Wmo,
                "World Models",
                passOptions.WmosVisible,
                wmoSourceCount > 0,
                wmoSourceCount,
                stats.WmoSubmission.SubmittedCount,
                "WMO geometry is still represented by placement markers in the current preview."),
            new(
                WorldRenderLayerKind.Doodad,
                "Doodads",
                passOptions.DoodadsVisible,
                mdxSourceCount > 0,
                mdxSourceCount,
                stats.MdxOpaqueSubmission.SubmittedCount + stats.MdxTransparentSubmission.SubmittedCount,
                "MDX/M2 geometry is still represented by placement markers in the current preview."),
            new(
                WorldRenderLayerKind.Overlay,
                "Overlays",
                passOptions.OverlayVisible,
                passOptions.OverlayVisible,
                0,
                stats.Overlay.SubmittedCount,
                "Debug/editor overlay layer."),
        ];

        return new WorldRenderCompositionFrame(layers);
    }
}
