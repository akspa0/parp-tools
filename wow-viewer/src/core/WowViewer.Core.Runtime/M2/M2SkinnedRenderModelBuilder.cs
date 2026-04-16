using System.Numerics;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2SkinnedRenderModel
{
    public M2SkinnedRenderModel(M2StaticRenderModel source, M2BonePoseState pose, IReadOnlyList<M2SkinnedRenderSection> sections)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(pose);
        ArgumentNullException.ThrowIfNull(sections);

        Source = source;
        Pose = pose;
        Sections = sections;
    }

    public M2StaticRenderModel Source { get; }

    public M2BonePoseState Pose { get; }

    public IReadOnlyList<M2SkinnedRenderSection> Sections { get; }

    public int VertexCount => Sections.Sum(static section => section.Vertices.Count);
}

public sealed class M2SkinnedRenderSection
{
    public M2SkinnedRenderSection(M2StructuredRenderSection source, IReadOnlyList<M2SkinnedRenderVertex> vertices)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(vertices);

        Source = source;
        Vertices = vertices;
    }

    public M2StructuredRenderSection Source { get; }

    public IReadOnlyList<M2SkinnedRenderVertex> Vertices { get; }
}

public readonly record struct M2SkinnedRenderVertex(
    Vector3 Position,
    Vector3 Normal,
    Vector2 TextureCoords,
    Vector4 BoneIndices,
    Vector4 BoneWeights);

public static class M2SkinnedRenderModelBuilder
{
    public static M2SkinnedRenderModel ApplyPose(M2StaticRenderModel renderModel, M2BonePoseState pose)
    {
        ArgumentNullException.ThrowIfNull(renderModel);
        ArgumentNullException.ThrowIfNull(pose);

        List<M2SkinnedRenderSection> sections = new(renderModel.StructuredSections.Count);
        foreach (M2StructuredRenderSection section in renderModel.StructuredSections)
        {
            List<M2SkinnedRenderVertex> vertices = new(section.Vertices.Count);
            foreach (M2StaticRenderVertex vertex in section.Vertices)
                vertices.Add(ApplyVertex(renderModel, section, vertex, pose));

            sections.Add(new M2SkinnedRenderSection(section, vertices));
        }

        return new M2SkinnedRenderModel(renderModel, pose, sections);
    }

    private static M2SkinnedRenderVertex ApplyVertex(
        M2StaticRenderModel renderModel,
        M2StructuredRenderSection section,
        M2StaticRenderVertex vertex,
        M2BonePoseState pose)
    {
        Vector3 skinnedPosition = Vector3.Zero;
        Vector3 skinnedNormal = Vector3.Zero;
        float totalWeight = 0.0f;

        for (int influence = 0; influence < 4; influence++)
        {
            float weight = GetComponent(vertex.BoneWeights, influence);
            if (weight <= 0.0f)
                continue;

            int boneIndex = ResolveBoneIndex(renderModel, section, (int)GetComponent(vertex.BoneIndices, influence));
            if (boneIndex < 0 || boneIndex >= pose.Matrices.Count)
                continue;

            Matrix4x4 matrix = pose.Matrices[boneIndex];
            skinnedPosition += Vector3.Transform(vertex.Position, matrix) * weight;
            skinnedNormal += Vector3.TransformNormal(vertex.Normal, matrix) * weight;
            totalWeight += weight;
        }

        if (totalWeight <= 0.0f)
        {
            skinnedPosition = vertex.Position;
            skinnedNormal = vertex.Normal;
        }
        else if (Math.Abs(totalWeight - 1.0f) > 0.0001f)
        {
            skinnedPosition /= totalWeight;
            skinnedNormal /= totalWeight;
        }

        if (skinnedNormal.LengthSquared() > 0.000001f)
            skinnedNormal = Vector3.Normalize(skinnedNormal);

        return new M2SkinnedRenderVertex(skinnedPosition, skinnedNormal, vertex.TextureCoords0, vertex.BoneIndices, vertex.BoneWeights);
    }

    private static int ResolveBoneIndex(M2StaticRenderModel renderModel, M2StructuredRenderSection section, int sectionBoneIndex)
    {
        if (sectionBoneIndex < 0)
            return -1;

        int scopedLookupIndex = section.BoneComboIndex + sectionBoneIndex;
        if (scopedLookupIndex >= 0 && scopedLookupIndex < renderModel.BoneLookup.Count)
            return renderModel.BoneLookup[scopedLookupIndex];

        if (sectionBoneIndex < renderModel.BoneLookup.Count)
            return renderModel.BoneLookup[sectionBoneIndex];

        return sectionBoneIndex;
    }

    private static float GetComponent(Vector4 value, int index)
    {
        return index switch
        {
            0 => value.X,
            1 => value.Y,
            2 => value.Z,
            3 => value.W,
            _ => 0.0f,
        };
    }
}
