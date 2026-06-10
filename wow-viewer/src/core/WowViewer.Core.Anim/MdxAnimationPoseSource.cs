using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.Anim;

public sealed class MdxAnimationPoseSource
{
    public MdxAnimationPoseSource(
        MdxFile mdx,
        string sourcePath,
        string contentHash)
    {
        ArgumentNullException.ThrowIfNull(mdx);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(contentHash);

        Mdx = mdx;
        SourcePath = sourcePath;
        ContentHash = contentHash;
    }

    public MdxFile Mdx { get; }

    public string SourcePath { get; }

    public string ContentHash { get; }
}
