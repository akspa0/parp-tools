namespace WowViewer.Core.Mdx;

public sealed class MdxGlobalSequenceSummary
{
    public MdxGlobalSequenceSummary(int index, uint duration)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        Duration = duration;
    }

    public int Index { get; }

    public uint Duration { get; }
}