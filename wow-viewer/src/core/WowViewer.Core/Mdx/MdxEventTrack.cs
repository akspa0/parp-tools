namespace WowViewer.Core.Mdx;

public sealed class MdxEventTrack
{
    public MdxEventTrack(string tag, int globalSequenceId, IReadOnlyList<int> keyTimes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tag);
        ArgumentNullException.ThrowIfNull(keyTimes);

        Tag = tag;
        GlobalSequenceId = globalSequenceId;
        KeyTimes = keyTimes;
    }

    public string Tag { get; }

    public int GlobalSequenceId { get; }

    public IReadOnlyList<int> KeyTimes { get; }

    public int KeyCount => KeyTimes.Count;

    public int? FirstKeyTime => KeyTimes.Count == 0 ? null : KeyTimes[0];

    public int? LastKeyTime => KeyTimes.Count == 0 ? null : KeyTimes[^1];
}
