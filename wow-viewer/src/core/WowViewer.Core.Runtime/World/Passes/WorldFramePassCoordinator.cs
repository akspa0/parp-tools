namespace WowViewer.Core.Runtime.World.Passes;

public readonly struct WorldFramePassOptions
{
    public WorldFramePassOptions(
        bool objectsVisible,
        bool wmosVisible,
        bool doodadsVisible,
        bool skyVisible = true,
        bool wdlVisible = false,
        bool terrainVisible = true,
        bool liquidVisible = true,
        bool overlayVisible = true)
    {
        ObjectsVisible = objectsVisible;
        WmosVisible = wmosVisible;
        DoodadsVisible = doodadsVisible;
        SkyVisible = skyVisible;
        WdlVisible = wdlVisible;
        TerrainVisible = terrainVisible;
        LiquidVisible = liquidVisible;
        OverlayVisible = overlayVisible;
    }

    public bool ObjectsVisible { get; }

    public bool WmosVisible { get; }

    public bool DoodadsVisible { get; }

    public bool SkyVisible { get; }

    public bool WdlVisible { get; }

    public bool TerrainVisible { get; }

    public bool LiquidVisible { get; }

    public bool OverlayVisible { get; }
}

public readonly record struct WorldFramePasses(
    Action RenderLighting,
    Action RenderSky,
    Action RenderSkyboxBackdrop,
    Action RenderWdl,
    Action RenderTerrain,
    Action PrepareObjectPhase,
    Action RenderWmoOpaque,
    Action RenderMdxOpaque,
    Action RenderLiquid,
    Action RenderMdxTransparent,
    Action RenderOverlay);

public static class WorldFramePassCoordinator
{
    public static bool Execute(WorldFramePassOptions options, in WorldFramePasses passes)
    {
        passes.RenderLighting();

        if (options.SkyVisible)
        {
            passes.RenderSky();
            passes.RenderSkyboxBackdrop();
        }

        if (options.WdlVisible)
            passes.RenderWdl();

        if (options.TerrainVisible)
            passes.RenderTerrain();

        if (!options.ObjectsVisible)
            return false;

        passes.PrepareObjectPhase();

        if (options.WmosVisible)
            passes.RenderWmoOpaque();

        if (options.DoodadsVisible)
            passes.RenderMdxOpaque();

        if (options.LiquidVisible)
            passes.RenderLiquid();

        if (options.DoodadsVisible)
            passes.RenderMdxTransparent();

        if (options.OverlayVisible)
            passes.RenderOverlay();

        return true;
    }
}
