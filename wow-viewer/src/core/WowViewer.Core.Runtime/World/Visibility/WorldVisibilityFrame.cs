namespace WowViewer.Core.Runtime.World.Visibility;

public sealed class WorldVisibilityFrame
{
    public List<WorldVisibleWmoEntry> VisibleWmos { get; } = new();
    public List<WorldVisibleMdxEntry> VisibleMdx { get; } = new();

    /// <summary>
    /// Instances dropped because they had faded past visibility. Counted rather than silently
    /// discarded, so the saving is measurable and a fade-band regression is visible.
    /// </summary>
    public int FullyFadedMdxCount { get; set; }

    public int VisibleTaxiMdxCount { get; set; }

    public void Reset()
    {
        VisibleWmos.Clear();
        VisibleMdx.Clear();
        FullyFadedMdxCount = 0;
        VisibleTaxiMdxCount = 0;
    }
}