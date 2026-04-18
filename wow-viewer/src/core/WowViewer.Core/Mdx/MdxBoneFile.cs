namespace WowViewer.Core.Mdx;

public sealed class MdxBoneFile
{
    public MdxBoneFile(string sourcePath, string signature, uint? version, string? modelName, IReadOnlyList<MdxBone> bones)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(bones);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Bones = bones;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxBone> Bones { get; }

    public int BoneCount => Bones.Count;
}
