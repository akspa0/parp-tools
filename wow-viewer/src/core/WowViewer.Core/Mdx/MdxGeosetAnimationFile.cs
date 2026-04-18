namespace WowViewer.Core.Mdx;

public sealed class MdxGeosetAnimationFile
{
    public MdxGeosetAnimationFile(string sourcePath, string signature, uint? version, string? modelName, IReadOnlyList<MdxGeosetAnimation> geosetAnimations)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(geosetAnimations);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        GeosetAnimations = geosetAnimations;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public IReadOnlyList<MdxGeosetAnimation> GeosetAnimations { get; }

    public int GeosetAnimationCount => GeosetAnimations.Count;
}
