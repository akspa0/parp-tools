namespace WowViewer.Core.Mdx;

public sealed class MdxGeometryFile
{
    public MdxGeometryFile(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        IReadOnlyList<MdxGeosetGeometry> geosets)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(geosets);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        Geosets = geosets;
        GeosetCount = geosets.Count;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxGeosetGeometry> Geosets { get; }

    public int GeosetCount { get; }
}