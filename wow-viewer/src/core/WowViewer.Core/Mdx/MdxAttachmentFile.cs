namespace WowViewer.Core.Mdx;

public sealed class MdxAttachmentFile
{
    public MdxAttachmentFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxAttachment> attachments)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(attachments);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Attachments = attachments;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxAttachment> Attachments { get; }

    public int AttachmentCount => Attachments.Count;
}
