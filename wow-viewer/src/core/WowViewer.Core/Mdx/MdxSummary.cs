using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxSummary
{
    public MdxSummary(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        uint? blendTime,
        Vector3? boundsMin,
        Vector3? boundsMax,
        IReadOnlyList<MdxSequenceSummary> sequences,
        IReadOnlyList<MdxGeosetSummary> geosets,
        IReadOnlyList<MdxPivotPointSummary> pivotPoints,
        IReadOnlyList<MdxTextureSummary> textures,
        IReadOnlyList<MdxMaterialSummary> materials,
        IReadOnlyList<MdxChunkSummary> chunks,
        int knownChunkCount,
        int unknownChunkCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(geosets);
        ArgumentNullException.ThrowIfNull(pivotPoints);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentNullException.ThrowIfNull(materials);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentOutOfRangeException.ThrowIfNegative(knownChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(unknownChunkCount);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        BlendTime = blendTime;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        Sequences = sequences;
        SequenceCount = sequences.Count;
        Geosets = geosets;
        GeosetCount = geosets.Count;
        PivotPoints = pivotPoints;
        PivotPointCount = pivotPoints.Count;
        Textures = textures;
        TextureCount = textures.Count;
        ReplaceableTextureCount = textures.Count(static texture => texture.IsReplaceable);
        Materials = materials;
        MaterialCount = materials.Count;
        MaterialLayerCount = materials.Sum(static material => material.LayerCount);
        Chunks = chunks;
        ChunkCount = chunks.Count;
        KnownChunkCount = knownChunkCount;
        UnknownChunkCount = unknownChunkCount;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public uint? BlendTime { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }

    public IReadOnlyList<MdxSequenceSummary> Sequences { get; }

    public int SequenceCount { get; }

    public IReadOnlyList<MdxGeosetSummary> Geosets { get; }

    public int GeosetCount { get; }

    public IReadOnlyList<MdxPivotPointSummary> PivotPoints { get; }

    public int PivotPointCount { get; }

    public IReadOnlyList<MdxTextureSummary> Textures { get; }

    public int TextureCount { get; }

    public int ReplaceableTextureCount { get; }

    public IReadOnlyList<MdxMaterialSummary> Materials { get; }

    public int MaterialCount { get; }

    public int MaterialLayerCount { get; }

    public IReadOnlyList<MdxChunkSummary> Chunks { get; }

    public int ChunkCount { get; }

    public int KnownChunkCount { get; }

    public int UnknownChunkCount { get; }
}