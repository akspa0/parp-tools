using WowViewer.Core.M2;

namespace WowViewer.Core.Anim;

public sealed class M2AnimationPoseSource
{
    public M2AnimationPoseSource(
        M2ModelDocument model,
        string sourcePath,
        string sourceFormat,
        string contentHash)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceFormat);
        ArgumentException.ThrowIfNullOrWhiteSpace(contentHash);

        Model = model;
        SourcePath = sourcePath;
        SourceFormat = sourceFormat;
        ContentHash = contentHash;
    }

    public M2ModelDocument Model { get; }

    public string SourcePath { get; }

    public string SourceFormat { get; }

    public string ContentHash { get; }
}
