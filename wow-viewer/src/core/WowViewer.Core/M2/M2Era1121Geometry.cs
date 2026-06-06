using System.Numerics;

namespace WowViewer.Core.M2;

public sealed class M2Era1121Geometry
{
    public M2Era1121Geometry(
        IReadOnlyList<M2Era1121VertexIndex> vertices,
        IReadOnlyList<Vector3> positions,
        IReadOnlyList<Vector3> normals,
        IReadOnlyList<Vector2> uvs,
        IReadOnlyList<ushort> triangles,
        IReadOnlyList<M2Era1121Batch> batches)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentNullException.ThrowIfNull(normals);
        ArgumentNullException.ThrowIfNull(uvs);
        ArgumentNullException.ThrowIfNull(triangles);
        ArgumentNullException.ThrowIfNull(batches);

        Vertices = vertices;
        Positions = positions;
        Normals = normals;
        Uvs = uvs;
        Triangles = triangles;
        Batches = batches;
    }

    public IReadOnlyList<M2Era1121VertexIndex> Vertices { get; }
    public IReadOnlyList<Vector3> Positions { get; }
    public IReadOnlyList<Vector3> Normals { get; }
    public IReadOnlyList<Vector2> Uvs { get; }
    public IReadOnlyList<ushort> Triangles { get; }
    public IReadOnlyList<M2Era1121Batch> Batches { get; }
}

public readonly record struct M2Era1121VertexIndex(ushort PositionIndex, ushort NormalIndex);

public sealed class M2Era1121Batch
{
    public M2Era1121Batch(
        ushort flags,
        ushort priorityPlane,
        ushort shaderId,
        ushort skinSectionIndex,
        ushort geosetIndex,
        ushort colorIndex,
        ushort materialIndex,
        ushort materialLayer,
        ushort textureCount,
        ushort textureComboIndex,
        ushort textureCoordComboIndex,
        ushort textureWeightComboIndex,
        ushort textureTransformComboIndex,
        ushort indexStart,
        ushort indexCount)
    {
        Flags = flags;
        PriorityPlane = priorityPlane;
        ShaderId = shaderId;
        SkinSectionIndex = skinSectionIndex;
        GeosetIndex = geosetIndex;
        ColorIndex = colorIndex;
        MaterialIndex = materialIndex;
        MaterialLayer = materialLayer;
        TextureCount = textureCount;
        TextureComboIndex = textureComboIndex;
        TextureCoordComboIndex = textureCoordComboIndex;
        TextureWeightComboIndex = textureWeightComboIndex;
        TextureTransformComboIndex = textureTransformComboIndex;
        IndexStart = indexStart;
        IndexCount = indexCount;
    }

    public ushort Flags { get; }
    public ushort PriorityPlane { get; }
    public ushort ShaderId { get; }
    public ushort SkinSectionIndex { get; }
    public ushort GeosetIndex { get; }
    public ushort ColorIndex { get; }
    public ushort MaterialIndex { get; }
    public ushort MaterialLayer { get; }
    public ushort TextureCount { get; }
    public ushort TextureComboIndex { get; }
    public ushort TextureCoordComboIndex { get; }
    public ushort TextureWeightComboIndex { get; }
    public ushort TextureTransformComboIndex { get; }
    public ushort IndexStart { get; }
    public ushort IndexCount { get; }
}
