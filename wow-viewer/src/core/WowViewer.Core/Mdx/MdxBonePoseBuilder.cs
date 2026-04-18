using System.Numerics;

namespace WowViewer.Core.Mdx;

public static class MdxBonePoseBuilder
{
    public static Matrix4x4[] Build(MdxBoneFile boneFile, MdxSummary summary, int sequenceIndex, int timeMs)
        => Build(boneFile, summary, sequenceIndex, timeMs, cameraPosition: null);

    public static Matrix4x4[] Build(MdxBoneFile boneFile, MdxSummary summary, int sequenceIndex, int timeMs, Vector3? cameraPosition)
    {
        ArgumentNullException.ThrowIfNull(boneFile);
        ArgumentNullException.ThrowIfNull(summary);

        if (boneFile.BoneCount == 0)
            return [];

        Matrix4x4[] matrices = new Matrix4x4[boneFile.BoneCount];
        for (int index = 0; index < matrices.Length; index++)
            matrices[index] = Matrix4x4.Identity;

        Dictionary<int, int> objectIdToIndex = new(boneFile.BoneCount);
        Dictionary<int, List<int>> childrenByParent = [];
        List<int> rootIndices = [];

        for (int index = 0; index < boneFile.Bones.Count; index++)
            objectIdToIndex[boneFile.Bones[index].ObjectId] = index;

        for (int index = 0; index < boneFile.Bones.Count; index++)
        {
            MdxBone bone = boneFile.Bones[index];
            if (!bone.HasParent || !objectIdToIndex.TryGetValue(bone.ParentId, out int parentIndex))
            {
                rootIndices.Add(index);
                continue;
            }

            if (!childrenByParent.TryGetValue(parentIndex, out List<int>? children))
            {
                children = [];
                childrenByParent[parentIndex] = children;
            }

            children.Add(index);
        }

        foreach (int rootIndex in rootIndices)
            UpdateBoneRecursive(rootIndex, Matrix4x4.Identity, boneFile.Bones, summary, sequenceIndex, timeMs, cameraPosition, childrenByParent, matrices);

        return matrices;
    }

    private static void UpdateBoneRecursive(
        int boneIndex,
        Matrix4x4 parentMatrix,
        IReadOnlyList<MdxBone> bones,
        MdxSummary summary,
        int sequenceIndex,
        int timeMs,
        Vector3? cameraPosition,
        IReadOnlyDictionary<int, List<int>> childrenByParent,
        Matrix4x4[] matrices)
    {
        MdxBone bone = bones[boneIndex];

        Vector3 pivot = bone.PivotPoint;
        Vector3 translation = MdxAnimationSampler.SampleVector3Track(bone.TranslationTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
        Quaternion rotation = MdxAnimationSampler.SampleQuaternionTrack(bone.RotationTrack, summary, sequenceIndex, timeMs, Quaternion.Identity);
        Vector3 scaling = MdxAnimationSampler.SampleVector3Track(bone.ScalingTrack, summary, sequenceIndex, timeMs, Vector3.One);

        Matrix4x4 parentRotationScaleMatrix = parentMatrix;
        if (bone.IgnoresParentTranslation)
        {
            parentRotationScaleMatrix.M41 = 0.0f;
            parentRotationScaleMatrix.M42 = 0.0f;
            parentRotationScaleMatrix.M43 = 0.0f;
        }

        Quaternion effectiveRotation = rotation;
        if (cameraPosition is Vector3 camera && bone.IsBillboard)
            effectiveRotation = Quaternion.Normalize(rotation * BuildBillboardRotation(bone, pivot + translation, camera));

        Matrix4x4 localMatrix = Matrix4x4.CreateTranslation(-pivot)
            * Matrix4x4.CreateScale(scaling)
            * Matrix4x4.CreateFromQuaternion(effectiveRotation)
            * Matrix4x4.CreateTranslation(pivot)
            * Matrix4x4.CreateTranslation(translation);

        Matrix4x4 worldMatrix = localMatrix * parentRotationScaleMatrix;
        matrices[boneIndex] = worldMatrix;

        if (!childrenByParent.TryGetValue(boneIndex, out List<int>? children))
            return;

        foreach (int childIndex in children)
            UpdateBoneRecursive(childIndex, worldMatrix, bones, summary, sequenceIndex, timeMs, cameraPosition, childrenByParent, matrices);
    }

    private static Quaternion BuildBillboardRotation(MdxBone bone, Vector3 bonePosition, Vector3 cameraPosition)
    {
        Vector3 toCamera = cameraPosition - bonePosition;
        if (toCamera.LengthSquared() <= 0.000001f)
            return Quaternion.Identity;

        toCamera = Vector3.Normalize(toCamera);

        if (bone.IsSphericalBillboard)
            return CreateLookRotation(toCamera, Vector3.UnitZ);

        Vector3 axis = bone.IsCylindricalBillboardLockX
            ? Vector3.UnitX
            : bone.IsCylindricalBillboardLockY
                ? Vector3.UnitY
                : Vector3.UnitZ;

        Vector3 projected = toCamera - (Vector3.Dot(toCamera, axis) * axis);
        if (projected.LengthSquared() <= 0.000001f)
            return Quaternion.Identity;

        projected = Vector3.Normalize(projected);
        return CreateLookRotation(projected, axis);
    }

    private static Quaternion CreateLookRotation(Vector3 forward, Vector3 upHint)
    {
        Vector3 safeForward = forward.LengthSquared() > 0.000001f ? Vector3.Normalize(forward) : Vector3.UnitY;
        Vector3 up = upHint.LengthSquared() > 0.000001f ? Vector3.Normalize(upHint) : Vector3.UnitZ;
        if (MathF.Abs(Vector3.Dot(safeForward, up)) > 0.999f)
            up = MathF.Abs(Vector3.Dot(safeForward, Vector3.UnitX)) < 0.999f ? Vector3.UnitX : Vector3.UnitY;

        Vector3 right = Vector3.Normalize(Vector3.Cross(up, safeForward));
        Vector3 recalculatedUp = Vector3.Normalize(Vector3.Cross(safeForward, right));

        Matrix4x4 matrix = new(
            right.X, right.Y, right.Z, 0.0f,
            recalculatedUp.X, recalculatedUp.Y, recalculatedUp.Z, 0.0f,
            safeForward.X, safeForward.Y, safeForward.Z, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        return Quaternion.CreateFromRotationMatrix(matrix);
    }
}
