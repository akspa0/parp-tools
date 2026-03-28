using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxPivotPointSummary
{
    public MdxPivotPointSummary(int index, Vector3 position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        Position = position;
    }

    public int Index { get; }

    public Vector3 Position { get; }
}