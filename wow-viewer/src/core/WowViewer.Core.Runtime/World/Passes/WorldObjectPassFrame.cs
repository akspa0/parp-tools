namespace WowViewer.Core.Runtime.World.Passes;

public sealed class WorldObjectPassFrame
{
    public List<WorldVisibleMdxPassRoute> OpaqueVisibleMdxRoutes { get; } = new();
    public List<WorldVisibleMdxPassRoute> TransparentVisibleMdxRoutes { get; } = new();
    public HashSet<int> UnbatchedVisibleMdxIndices { get; } = new();
    public HashSet<string> UpdatedMdxModelKeys { get; } = new(StringComparer.OrdinalIgnoreCase);
    public int FirstOpaqueBatchedVisibleMdxIndex { get; set; } = -1;

    public void Reset()
    {
        OpaqueVisibleMdxRoutes.Clear();
        TransparentVisibleMdxRoutes.Clear();
        UnbatchedVisibleMdxIndices.Clear();
        UpdatedMdxModelKeys.Clear();
        FirstOpaqueBatchedVisibleMdxIndex = -1;
    }
}