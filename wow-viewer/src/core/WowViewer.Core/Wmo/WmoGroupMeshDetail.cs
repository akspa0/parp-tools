using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoGroupMeshDetail
{
    public WmoGroupMeshDetail(
        string sourcePath,
        uint? version,
        int headerSizeBytes,
        string? indexChunkId,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<Vector3> normals,
        IReadOnlyList<ushort> indices,
        IReadOnlyList<Vector2> primaryUvs,
        IReadOnlyList<IReadOnlyList<Vector2>> additionalUvSets,
        IReadOnlyList<uint> primaryVertexColorsBgra,
        IReadOnlyList<IReadOnlyList<uint>> additionalVertexColorSetsBgra,
        IReadOnlyList<WmoGroupFaceMaterialDetail> faceMaterials,
        IReadOnlyList<WmoGroupBatchDetail> batches)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(headerSizeBytes);
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(normals);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(primaryUvs);
        ArgumentNullException.ThrowIfNull(additionalUvSets);
        ArgumentNullException.ThrowIfNull(primaryVertexColorsBgra);
        ArgumentNullException.ThrowIfNull(additionalVertexColorSetsBgra);
        ArgumentNullException.ThrowIfNull(faceMaterials);
        ArgumentNullException.ThrowIfNull(batches);

        SourcePath = sourcePath;
        Version = version;
        HeaderSizeBytes = headerSizeBytes;
        IndexChunkId = indexChunkId;
        Vertices = vertices;
        Normals = normals;
        Indices = indices;
        PrimaryUvs = primaryUvs;
        AdditionalUvSets = additionalUvSets;
        PrimaryVertexColorsBgra = primaryVertexColorsBgra;
        AdditionalVertexColorSetsBgra = additionalVertexColorSetsBgra;
        FaceMaterials = faceMaterials;
        Batches = batches;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int HeaderSizeBytes { get; }

    public string? IndexChunkId { get; }

    public IReadOnlyList<Vector3> Vertices { get; }

    public IReadOnlyList<Vector3> Normals { get; }

    public IReadOnlyList<ushort> Indices { get; }

    public IReadOnlyList<Vector2> PrimaryUvs { get; }

    public IReadOnlyList<IReadOnlyList<Vector2>> AdditionalUvSets { get; }

    public IReadOnlyList<uint> PrimaryVertexColorsBgra { get; }

    public IReadOnlyList<IReadOnlyList<uint>> AdditionalVertexColorSetsBgra { get; }

    public IReadOnlyList<WmoGroupFaceMaterialDetail> FaceMaterials { get; }

    public IReadOnlyList<WmoGroupBatchDetail> Batches { get; }
}