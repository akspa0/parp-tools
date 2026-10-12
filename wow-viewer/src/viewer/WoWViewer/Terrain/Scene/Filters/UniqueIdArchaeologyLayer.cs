namespace WoWViewer.Terrain;

public readonly struct UniqueIdArchaeologyLayer
{
    public UniqueIdArchaeologyLayer(int layerNumber, int minUniqueId, int maxUniqueId, int placementCount, int wmoCount, int mdxCount)
    {
        LayerNumber = layerNumber;
        MinUniqueId = minUniqueId;
        MaxUniqueId = maxUniqueId;
        PlacementCount = placementCount;
        WmoCount = wmoCount;
        MdxCount = mdxCount;
    }

    public int LayerNumber { get; }
    public int MinUniqueId { get; }
    public int MaxUniqueId { get; }
    public int PlacementCount { get; }
    public int WmoCount { get; }
    public int MdxCount { get; }
}
