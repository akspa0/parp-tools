namespace WowViewer.Core.Mdx;

public sealed class MdxTextureAnimationFile
{
    public MdxTextureAnimationFile(string sourcePath, string signature, uint? version, string? modelName, IReadOnlyList<MdxTextureAnimation> textureAnimations)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(textureAnimations);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        TextureAnimations = textureAnimations;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxTextureAnimation> TextureAnimations { get; }

    public int TextureAnimationCount => TextureAnimations.Count;
}