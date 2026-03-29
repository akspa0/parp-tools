namespace WowViewer.Core.Mdx;

public sealed class MdxHitTestFile
{
    public MdxHitTestFile(string sourcePath, string signature, uint? version, string? modelName, IReadOnlyList<MdxHitTestShape> shapes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(shapes);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Shapes = shapes;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxHitTestShape> Shapes { get; }

    public int ShapeCount => Shapes.Count;
}