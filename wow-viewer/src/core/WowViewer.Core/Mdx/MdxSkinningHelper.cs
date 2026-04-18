using System.Numerics;

namespace WowViewer.Core.Mdx;

public static class MdxSkinningHelper
{
    public static (Vector4[] Indices, Vector4[] Weights) BuildBoneWeights(MdxGeosetGeometry geoset, IReadOnlyList<MdxBone> bones)
    {
        ArgumentNullException.ThrowIfNull(geoset);
        ArgumentNullException.ThrowIfNull(bones);

        int vertexCount = geoset.VertexCount;
        Vector4[] indices = new Vector4[vertexCount];
        Vector4[] weights = new Vector4[vertexCount];

        if (vertexCount == 0)
            return (indices, weights);

        if (geoset.VertexGroupCount == 0 || geoset.MatrixGroupCount == 0)
        {
            for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
            {
                indices[vertexIndex] = Vector4.Zero;
                weights[vertexIndex] = new Vector4(1.0f, 0.0f, 0.0f, 0.0f);
            }

            return (indices, weights);
        }

        Dictionary<uint, int> objectIdToBoneIndex = new(bones.Count);
        for (int boneIndex = 0; boneIndex < bones.Count; boneIndex++)
            objectIdToBoneIndex[(uint)bones[boneIndex].ObjectId] = boneIndex;

        int[] groupOffsets = new int[geoset.MatrixGroupCount];
        int offset = 0;
        for (int groupIndex = 0; groupIndex < geoset.MatrixGroupCount; groupIndex++)
        {
            groupOffsets[groupIndex] = offset;
            offset += checked((int)geoset.MatrixGroups[groupIndex]);
        }

        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
        {
            byte groupIndex = geoset.VertexGroups[vertexIndex];
            if (groupIndex >= geoset.MatrixGroupCount)
            {
                indices[vertexIndex] = Vector4.Zero;
                weights[vertexIndex] = new Vector4(1.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }

            uint boneCount = geoset.MatrixGroups[groupIndex];
            int matrixOffset = groupOffsets[groupIndex];
            float[] vertexBoneIndices = new float[4];
            float[] vertexBoneWeights = new float[4];
            float weight = boneCount == 0 ? 1.0f : 1.0f / boneCount;

            for (int boneSlot = 0; boneSlot < Math.Min(boneCount, 4); boneSlot++)
            {
                if (matrixOffset + boneSlot >= geoset.MatrixIndexCount)
                    continue;

                uint matrixValue = geoset.MatrixIndices[matrixOffset + boneSlot];
                if (objectIdToBoneIndex.TryGetValue(matrixValue, out int remappedBoneIndex))
                    vertexBoneIndices[boneSlot] = remappedBoneIndex;
                else if (matrixValue < bones.Count)
                    vertexBoneIndices[boneSlot] = matrixValue;

                vertexBoneWeights[boneSlot] = weight;
            }

            indices[vertexIndex] = new Vector4(vertexBoneIndices[0], vertexBoneIndices[1], vertexBoneIndices[2], vertexBoneIndices[3]);
            weights[vertexIndex] = new Vector4(vertexBoneWeights[0], vertexBoneWeights[1], vertexBoneWeights[2], vertexBoneWeights[3]);
        }

        return (indices, weights);
    }

    public static Vector3 ApplySkinning(Vector3 position, Vector4 boneIndices, Vector4 boneWeights, IReadOnlyList<Matrix4x4> boneMatrices)
    {
        return ApplyWeightedTransform(new Vector4(position, 1.0f), position, boneIndices, boneWeights, boneMatrices, normalizeResult: false);
    }

    public static Vector3 ApplySkinningNormal(Vector3 normal, Vector4 boneIndices, Vector4 boneWeights, IReadOnlyList<Matrix4x4> boneMatrices)
    {
        Vector3 transformed = ApplyWeightedTransform(new Vector4(normal, 0.0f), normal, boneIndices, boneWeights, boneMatrices, normalizeResult: true);
        return transformed.LengthSquared() <= 0.000001f ? Vector3.UnitZ : transformed;
    }

    public static float[] BuildSkinningVertexData(IReadOnlyList<Vector4> boneIndices, IReadOnlyList<Vector4> boneWeights, int vertexCount)
    {
        ArgumentNullException.ThrowIfNull(boneIndices);
        ArgumentNullException.ThrowIfNull(boneWeights);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);

        float[] skinningVertexData = new float[checked(vertexCount * 8)];
        for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
        {
            int offset = vertexIndex * 8;
            Vector4 indices = vertexIndex < boneIndices.Count ? boneIndices[vertexIndex] : Vector4.Zero;
            Vector4 weights = vertexIndex < boneWeights.Count ? boneWeights[vertexIndex] : Vector4.Zero;

            skinningVertexData[offset + 0] = indices.X;
            skinningVertexData[offset + 1] = indices.Y;
            skinningVertexData[offset + 2] = indices.Z;
            skinningVertexData[offset + 3] = indices.W;
            skinningVertexData[offset + 4] = weights.X;
            skinningVertexData[offset + 5] = weights.Y;
            skinningVertexData[offset + 6] = weights.Z;
            skinningVertexData[offset + 7] = weights.W;
        }

        return skinningVertexData;
    }

    private static Vector3 ApplyWeightedTransform(
        Vector4 source,
        Vector3 fallback,
        Vector4 boneIndices,
        Vector4 boneWeights,
        IReadOnlyList<Matrix4x4> boneMatrices,
        bool normalizeResult)
    {
        float totalWeight = boneWeights.X + boneWeights.Y + boneWeights.Z + boneWeights.W;
        if (totalWeight <= 0.0001f)
            return fallback;

        Vector4 weighted = Vector4.Zero;
        bool appliedMatrix = false;

        appliedMatrix |= TryAccumulate(source, boneIndices.X, boneWeights.X / totalWeight, boneMatrices, ref weighted);
        appliedMatrix |= TryAccumulate(source, boneIndices.Y, boneWeights.Y / totalWeight, boneMatrices, ref weighted);
        appliedMatrix |= TryAccumulate(source, boneIndices.Z, boneWeights.Z / totalWeight, boneMatrices, ref weighted);
        appliedMatrix |= TryAccumulate(source, boneIndices.W, boneWeights.W / totalWeight, boneMatrices, ref weighted);

        if (!appliedMatrix)
            return fallback;

        Vector3 result = new(weighted.X, weighted.Y, weighted.Z);
        if (normalizeResult && result.LengthSquared() > 0.000001f)
            result = Vector3.Normalize(result);

        return result;
    }

    private static bool TryAccumulate(Vector4 source, float boneIndexValue, float weight, IReadOnlyList<Matrix4x4> boneMatrices, ref Vector4 destination)
    {
        if (weight <= 0.0f)
            return false;

        int boneIndex = (int)boneIndexValue;
        if ((uint)boneIndex >= (uint)boneMatrices.Count)
            return false;

        destination += Vector4.Transform(source, boneMatrices[boneIndex]) * weight;
        return true;
    }
}
