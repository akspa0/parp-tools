namespace WowViewer.Core.Mdx;

public sealed class MdxScalarKeyframe
{
    public MdxScalarKeyframe(int time, float value, float? inTangent, float? outTangent)
    {
        Time = time;
        Value = value;
        InTangent = inTangent;
        OutTangent = outTangent;
    }

    public int Time { get; }

    public float Value { get; }

    public float? InTangent { get; }

    public float? OutTangent { get; }
}
