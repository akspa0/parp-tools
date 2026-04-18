namespace WowViewer.Core.Mdx;

public sealed class MdxRibbonEmitterFile
{
    public MdxRibbonEmitterFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxRibbonEmitter> ribbons)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(ribbons);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Ribbons = ribbons;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxRibbonEmitter> Ribbons { get; }

    public int RibbonCount => Ribbons.Count;
}
