using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxQuaternionKeyframe
{
    public MdxQuaternionKeyframe(int time, Quaternion value, Quaternion? inTangent, Quaternion? outTangent)
    {
        Time = time;
        Value = value;
        InTangent = inTangent;
        OutTangent = outTangent;
    }

    public int Time { get; }

    public Quaternion Value { get; }

    public Quaternion? InTangent { get; }

    public Quaternion? OutTangent { get; }
}