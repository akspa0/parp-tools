namespace WowViewer.Core.Maps;

public sealed class WdtMainFlagValueSummary
{
    public WdtMainFlagValueSummary(uint value, int tileCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tileCount);

        Value = value;
        TileCount = tileCount;
    }

    public uint Value { get; }

    public int TileCount { get; }
}