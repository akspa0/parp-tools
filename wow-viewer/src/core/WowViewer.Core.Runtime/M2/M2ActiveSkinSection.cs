using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ActiveSkinSection
{
    public M2ActiveSkinSection(int sectionIndex, M2SkinSubmesh submesh, IReadOnlyList<M2ActiveSkinBatch> batches)
    {
        ArgumentNullException.ThrowIfNull(submesh);
        ArgumentNullException.ThrowIfNull(batches);

        SectionIndex = sectionIndex;
        SkinSectionId = submesh.SkinSectionId;
        Level = submesh.Level;
        VertexStart = submesh.VertexStart;
        VertexCount = submesh.VertexCount;
        IndexStart = submesh.IndexStart;
        IndexCount = submesh.IndexCount;
        Batches = batches;
    }

    public int SectionIndex { get; }

    public ushort SkinSectionId { get; }

    public ushort Level { get; }

    public ushort VertexStart { get; }

    public ushort VertexCount { get; }

    public ushort IndexStart { get; }

    public ushort IndexCount { get; }

    public IReadOnlyList<M2ActiveSkinBatch> Batches { get; }

    public int ActiveBatchCount => Batches.Count;
}