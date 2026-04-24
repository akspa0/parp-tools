namespace WowViewer.Core.Runtime.World;

public enum WorldRenderLayerKind
{
    Sky = 0,
    SkyboxBackdrop = 1,
    Wdl = 2,
    Terrain = 3,
    Liquid = 4,
    Wmo = 5,
    Doodad = 6,
    Overlay = 7,
}

public readonly record struct WorldRenderLayerState(
    WorldRenderLayerKind Kind,
    string DisplayName,
    bool Enabled,
    bool Ready,
    int SourceCount,
    int SubmittedCount,
    string Note);

public sealed class WorldRenderCompositionFrame
{
    public WorldRenderCompositionFrame(IReadOnlyList<WorldRenderLayerState> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        Layers = layers;
    }

    public IReadOnlyList<WorldRenderLayerState> Layers { get; }

    public IEnumerable<WorldRenderLayerState> EnabledLayers => Layers.Where(static layer => layer.Enabled);

    public bool HasSubmittedSkyLayer => Layers.Any(static layer =>
        layer.Kind is WorldRenderLayerKind.Sky or WorldRenderLayerKind.SkyboxBackdrop
        && layer.Enabled
        && layer.SubmittedCount > 0);
}
