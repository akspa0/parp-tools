namespace WowViewer.Core.Mdx;

public sealed class MdxIntTrack
{
    public MdxIntTrack(string tag, MdxTrackInterpolationType interpolationType, int globalSequenceId, IReadOnlyList<MdxIntKeyframe> keys)
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

    public IReadOnlyList<MdxIntKeyframe> Keys { get; }

    public int KeyCount => Keys.Count;

    public int? FirstKeyTime => Keys.Count == 0 ? null : Keys[0].Time;

    public int? LastKeyTime => Keys.Count == 0 ? null : Keys[^1].Time;
}
