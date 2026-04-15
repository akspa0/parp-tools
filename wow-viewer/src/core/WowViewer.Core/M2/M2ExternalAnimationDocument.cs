namespace WowViewer.Core.M2;

public sealed class M2ExternalAnimationDocument
{
    public M2ExternalAnimationDocument(
        string sourcePath,
        byte[] payload,
        bool isChunkedContainer,
        string? containerSignature)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(payload);

        SourcePath = M2ModelIdentity.NormalizePath(sourcePath);
        Payload = payload;
        IsChunkedContainer = isChunkedContainer;
        ContainerSignature = containerSignature;
    }

    public string SourcePath { get; }

    public byte[] Payload { get; }

    public int PayloadSizeBytes => Payload.Length;

    public bool IsChunkedContainer { get; }

    public string? ContainerSignature { get; }
}
