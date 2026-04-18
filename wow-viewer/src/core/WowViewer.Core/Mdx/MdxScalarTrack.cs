namespace WowViewer.Core.Mdx;

public sealed class MdxScalarTrack
{
    public MdxScalarTrack(string tag, MdxTrackInterpolationType interpolationType, int globalSequenceId, IReadOnlyList<MdxScalarKeyframe> keys)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tag);
        ArgumentNullException.ThrowIfNull(keys);

        Tag = tag;
        InterpolationType = interpolationType;
        GlobalSequenceId = globalSequenceId;
        Keys = keys;
    }

    public string Tag { get; }

    public MdxTrackInterpolationType InterpolationType { get; }

    public int GlobalSequenceId { get; }

    public IReadOnlyList<MdxScalarKeyframe> Keys { get; }

    public int KeyCount => Keys.Count;

    public int? FirstKeyTime => Keys.Count == 0 ? null : Keys[0].Time;

    public int? LastKeyTime => Keys.Count == 0 ? null : Keys[^1].Time;
}
