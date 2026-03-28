using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxSequenceSummary
{
    public MdxSequenceSummary(
        int index,
        string? name,
        int startTime,
        int endTime,
        float moveSpeed,
        uint flags,
        float frequency,
        int replayStart,
        int replayEnd,
        uint? blendTime,
        Vector3? boundsMin,
        Vector3? boundsMax,
        float? boundsRadius)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        Name = string.IsNullOrWhiteSpace(name) ? null : name;
        StartTime = startTime;
        EndTime = endTime;
        MoveSpeed = moveSpeed;
        Flags = flags;
        Frequency = frequency;
        ReplayStart = replayStart;
        ReplayEnd = replayEnd;
        BlendTime = blendTime;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        BoundsRadius = boundsRadius;
    }

    public int Index { get; }

    public string? Name { get; }

    public int StartTime { get; }

    public int EndTime { get; }

    public int Duration => EndTime - StartTime;

    public float MoveSpeed { get; }

    public uint Flags { get; }

    public float Frequency { get; }

    public int ReplayStart { get; }

    public int ReplayEnd { get; }

    public uint? BlendTime { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }

    public float? BoundsRadius { get; }
}