namespace WowViewer.Core.Mdx;

public sealed class MdxEventFile
{
    public MdxEventFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxEvent> events)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(events);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Events = events;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxEvent> Events { get; }

    public int EventCount => Events.Count;
}
