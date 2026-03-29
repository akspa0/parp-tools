namespace WowViewer.Core.Mdx;

public sealed class MdxQuaternionNodeTrack
{
    public MdxQuaternionNodeTrack(string tag, MdxTrackInterpolationType interpolationType, int globalSequenceId, IReadOnlyList<MdxQuaternionKeyframe> keys)
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

    public IReadOnlyList<MdxQuaternionKeyframe> Keys { get; }

    public int KeyCount => Keys.Count;

    public int? FirstKeyTime => Keys.Count == 0 ? null : Keys[0].Time;

    public int? LastKeyTime => Keys.Count == 0 ? null : Keys[^1].Time;
}