using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxVector3Keyframe
{
    public MdxVector3Keyframe(int time, Vector3 value, Vector3? inTangent, Vector3? outTangent)
    {
        Time = time;
        Value = value;
        InTangent = inTangent;
        OutTangent = outTangent;
    }

    public int Time { get; }

    public Vector3 Value { get; }

    public Vector3? InTangent { get; }

    public Vector3? OutTangent { get; }
}