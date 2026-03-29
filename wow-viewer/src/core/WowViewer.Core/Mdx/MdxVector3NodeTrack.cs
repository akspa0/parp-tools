namespace WowViewer.Core.Mdx;

public sealed class MdxVector3NodeTrack
{
    public MdxVector3NodeTrack(string tag, MdxTrackInterpolationType interpolationType, int globalSequenceId, IReadOnlyList<MdxVector3Keyframe> keys)
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

    public IReadOnlyList<MdxVector3Keyframe> Keys { get; }

    public int KeyCount => Keys.Count;

    public int? FirstKeyTime => Keys.Count == 0 ? null : Keys[0].Time;

    public int? LastKeyTime => Keys.Count == 0 ? null : Keys[^1].Time;
}