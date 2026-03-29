namespace WowViewer.Core.Mdx;

public sealed class MdxCollisionFile
{
    public MdxCollisionFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        MdxCollisionMesh? collision)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Collision = collision;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public MdxCollisionMesh? Collision { get; }

    public bool HasCollision => Collision is not null;
}