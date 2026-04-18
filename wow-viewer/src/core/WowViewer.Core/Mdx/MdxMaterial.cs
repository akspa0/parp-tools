namespace WowViewer.Core.Mdx;

public sealed class MdxMaterial
{
    public MdxMaterial(int index, int priorityPlane, IReadOnlyList<MdxMaterialLayer> layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentNullException.ThrowIfNull(layers);

        Index = index;
        PriorityPlane = priorityPlane;
        Layers = layers;
    }

    public int Index { get; }

    public int PriorityPlane { get; }

    public IReadOnlyList<MdxMaterialLayer> Layers { get; }

    public int LayerCount => Layers.Count;
}