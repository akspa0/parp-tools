namespace WowViewer.Core.M2;

public sealed class M2SkinSubmesh
{
    public M2SkinSubmesh(
        ushort skinSectionId,
        ushort level,
        ushort vertexStart,
        ushort vertexCount,
        ushort indexStart,
        ushort indexCount,
        ushort boneCount = 0,
        ushort boneComboIndex = 0,
        ushort boneInfluences = 0,
        ushort centerBoneIndex = 0)
    {
        SkinSectionId = skinSectionId;
        Level = level;
        VertexStart = vertexStart;
        VertexCount = vertexCount;
        IndexStart = indexStart;
        IndexCount = indexCount;
        BoneCount = boneCount;
        BoneComboIndex = boneComboIndex;
        BoneInfluences = boneInfluences;
        CenterBoneIndex = centerBoneIndex;
    }

    public ushort SkinSectionId { get; }

    public ushort Level { get; }

    public ushort VertexStart { get; }

    public ushort VertexCount { get; }

    public ushort IndexStart { get; }

    public ushort IndexCount { get; }

    public ushort BoneCount { get; }

    public ushort BoneComboIndex { get; }

    public ushort BoneInfluences { get; }

    public ushort CenterBoneIndex { get; }
}
