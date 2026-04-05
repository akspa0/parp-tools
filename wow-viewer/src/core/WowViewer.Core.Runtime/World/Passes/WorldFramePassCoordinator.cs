namespace WowViewer.Core.Runtime.World.Passes;

public readonly record struct WorldFramePassOptions(
    bool ObjectsVisible,
    bool WmosVisible,
    bool DoodadsVisible);

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
        passes.RenderSky();
        passes.RenderSkyboxBackdrop();
        passes.RenderWdl();
        passes.RenderTerrain();

        if (!options.ObjectsVisible)
            return false;

        passes.PrepareObjectPhase();

        if (options.WmosVisible)
            passes.RenderWmoOpaque();

        if (options.DoodadsVisible)
            passes.RenderMdxOpaque();

        passes.RenderLiquid();

        if (options.DoodadsVisible)
            passes.RenderMdxTransparent();

        passes.RenderOverlay();
        return true;
    }
}