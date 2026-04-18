namespace WowViewer.Core.Mdx;

public sealed class MdxMaterialFile
{
    public MdxMaterialFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxMaterial> materials)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(materials);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Materials = materials;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxMaterial> Materials { get; }

    public int MaterialCount => Materials.Count;
}