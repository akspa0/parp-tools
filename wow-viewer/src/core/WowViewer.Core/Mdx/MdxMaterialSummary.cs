namespace WowViewer.Core.Mdx;

public sealed class MdxMaterialSummary
{
    public MdxMaterialSummary(int index, int priorityPlane, IReadOnlyList<MdxMaterialLayerSummary> layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentNullException.ThrowIfNull(layers);

        Index = index;
        PriorityPlane = priorityPlane;
        Layers = layers;
        LayerCount = layers.Count;
    }

    public int Index { get; }

    public int PriorityPlane { get; }

    public IReadOnlyList<MdxMaterialLayerSummary> Layers { get; }

    public int LayerCount { get; }
}