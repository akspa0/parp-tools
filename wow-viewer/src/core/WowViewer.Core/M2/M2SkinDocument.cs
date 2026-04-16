namespace WowViewer.Core.M2;

public sealed class M2SkinDocument
{
    public M2SkinDocument(
        string sourcePath,
        string signature,
        IReadOnlyList<ushort> vertexLookup,
        uint vertexLookupOffset,
        IReadOnlyList<ushort> triangleIndices,
        uint triangleIndexOffset,
        IReadOnlyList<M2SkinBoneEntry> boneEntries,
        uint boneEntryOffset,
        IReadOnlyList<M2SkinSubmesh> submeshes,
        uint submeshOffset,
        IReadOnlyList<M2SkinBatch> batches,
        uint batchOffset,
        uint globalVertexOffset,
        uint shadowBatchCount,
        uint shadowBatchOffset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(vertexLookup);
        ArgumentNullException.ThrowIfNull(triangleIndices);
        ArgumentNullException.ThrowIfNull(boneEntries);
        ArgumentNullException.ThrowIfNull(submeshes);
        ArgumentNullException.ThrowIfNull(batches);

        SourcePath = sourcePath;
        Signature = signature;
        VertexLookup = vertexLookup;
        VertexLookupCount = vertexLookup.Count;
        VertexLookupOffset = vertexLookupOffset;
        TriangleIndices = triangleIndices;
        TriangleIndexCount = triangleIndices.Count;
        TriangleIndexOffset = triangleIndexOffset;
        BoneEntries = boneEntries;
        BoneEntryCount = boneEntries.Count;
        BoneEntryOffset = boneEntryOffset;
        Submeshes = submeshes;
        SubmeshCount = submeshes.Count;
        SubmeshOffset = submeshOffset;
        Batches = batches;
        BatchCount = batches.Count;
        BatchOffset = batchOffset;
        GlobalVertexOffset = globalVertexOffset;
        ShadowBatchCount = shadowBatchCount;
        ShadowBatchOffset = shadowBatchOffset;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public IReadOnlyList<ushort> VertexLookup { get; }

    public int VertexLookupCount { get; }

    public uint VertexLookupOffset { get; }

    public IReadOnlyList<ushort> TriangleIndices { get; }

    public int TriangleIndexCount { get; }

    public uint TriangleIndexOffset { get; }

    public IReadOnlyList<M2SkinBoneEntry> BoneEntries { get; }

    public int BoneEntryCount { get; }

    public uint BoneEntryOffset { get; }

    public IReadOnlyList<M2SkinSubmesh> Submeshes { get; }

    public int SubmeshCount { get; }

    public uint SubmeshOffset { get; }

    public IReadOnlyList<M2SkinBatch> Batches { get; }

    public int BatchCount { get; }

    public uint BatchOffset { get; }

    public uint GlobalVertexOffset { get; }

    public uint ShadowBatchCount { get; }

    public uint ShadowBatchOffset { get; }

    public bool HasShadowBatches => ShadowBatchCount > 0;
}

public readonly record struct M2SkinBoneEntry(
    byte Bone0,
    byte Bone1,
    byte Bone2,
    byte Bone3);