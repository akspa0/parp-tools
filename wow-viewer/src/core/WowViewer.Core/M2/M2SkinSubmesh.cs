namespace WowViewer.Core.M2;

public sealed class M2SkinSubmesh
{
    public M2SkinSubmesh(
        ushort skinSectionId,
        ushort level,
        ushort vertexStart,
        ushort vertexCount,
        ushort indexStart,
        ushort indexCount)
    {
        SkinSectionId = skinSectionId;
        Level = level;
        VertexStart = vertexStart;
        VertexCount = vertexCount;
        IndexStart = indexStart;
        IndexCount = indexCount;
    }

    public ushort SkinSectionId { get; }

    public ushort Level { get; }

    public ushort VertexStart { get; }

    public ushort VertexCount { get; }

    public ushort IndexStart { get; }

    public ushort IndexCount { get; }
}