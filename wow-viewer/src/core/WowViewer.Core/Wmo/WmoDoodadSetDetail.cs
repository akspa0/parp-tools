namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadSetDetail
{
    public WmoDoodadSetDetail(int setIndex, string name, int startIndex, int count, int rangeEnd, uint rawFlags)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(setIndex);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegative(rangeEnd);

        SetIndex = setIndex;
        Name = name;
        StartIndex = startIndex;
        Count = count;
        RangeEnd = rangeEnd;
        RawFlags = rawFlags;
    }

    public int SetIndex { get; }

    public string Name { get; }

    public int StartIndex { get; }

    public int Count { get; }

    public int RangeEnd { get; }

    public uint RawFlags { get; }
}