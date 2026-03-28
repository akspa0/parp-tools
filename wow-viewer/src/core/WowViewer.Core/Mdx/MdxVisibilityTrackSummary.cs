namespace WowViewer.Core.Mdx;

public sealed class MdxVisibilityTrackSummary
{
    public MdxVisibilityTrackSummary(string tag, int keyCount, uint interpolationType, int globalSequenceId, int? firstKeyTime, int? lastKeyTime)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tag);
        ArgumentOutOfRangeException.ThrowIfNegative(keyCount);

        if (keyCount == 0 && (firstKeyTime is not null || lastKeyTime is not null))
            throw new ArgumentException("Track times must be null when the track has no keys.", nameof(firstKeyTime));

        if (firstKeyTime is not null && lastKeyTime is not null && lastKeyTime < firstKeyTime)
            throw new ArgumentException("Track end time must be greater than or equal to the start time.", nameof(lastKeyTime));

        Tag = tag;
        KeyCount = keyCount;
        InterpolationType = interpolationType;
        GlobalSequenceId = globalSequenceId;
        FirstKeyTime = firstKeyTime;
        LastKeyTime = lastKeyTime;
    }

    public string Tag { get; }

    public int KeyCount { get; }

    public uint InterpolationType { get; }

    public int GlobalSequenceId { get; }

    public int? FirstKeyTime { get; }

    public int? LastKeyTime { get; }
}