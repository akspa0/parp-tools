namespace WowViewer.Core.Runtime.World.Visibility;

public sealed class WorldVisibilityFrame
{
    public List<WorldVisibleWmoEntry> VisibleWmos { get; } = new();
    public List<WorldVisibleMdxEntry> VisibleMdx { get; } = new();

    public int VisibleTaxiMdxCount { get; set; }

    public void Reset()
    {
        VisibleWmos.Clear();
        VisibleMdx.Clear();
        VisibleTaxiMdxCount = 0;
    }
}