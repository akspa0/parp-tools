namespace WowViewer.Core.Mdx;

public sealed class MdxIntKeyframe
{
    public MdxIntKeyframe(int time, int value, int? inTangent, int? outTangent)
    {
        Time = time;
        Value = value;
        InTangent = inTangent;
        OutTangent = outTangent;
    }

    public int Time { get; }

    public int Value { get; }

    public int? InTangent { get; }

    public int? OutTangent { get; }
}
