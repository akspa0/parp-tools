using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2BonePoseState
{
    public M2BonePoseState(
        int requestedSequenceIndex,
        int resolvedSequenceIndex,
        int timeMs,
        bool usesExternalPayload,
        IReadOnlyList<M2BonePose> bones)
    {
        ArgumentNullException.ThrowIfNull(bones);

        RequestedSequenceIndex = requestedSequenceIndex;
        ResolvedSequenceIndex = resolvedSequenceIndex;
        TimeMs = timeMs;
        UsesExternalPayload = usesExternalPayload;
        Bones = bones;
        Matrices = bones.Select(static bone => bone.WorldTransform).ToArray();
    }

    public int RequestedSequenceIndex { get; }

    public int ResolvedSequenceIndex { get; }

    public int TimeMs { get; }

    public bool UsesExternalPayload { get; }

    public IReadOnlyList<M2BonePose> Bones { get; }

    public IReadOnlyList<Matrix4x4> Matrices { get; }

    public int BoneCount => Bones.Count;
}

public sealed class M2BonePose
{
    public M2BonePose(
        int boneIndex,
        short parentBone,
        Vector3 pivot,
        Vector3 translation,
        Quaternion rotation,
        Vector3 scaling,
        Matrix4x4 localTransform,
        Matrix4x4 worldTransform)
    {
        BoneIndex = boneIndex;
        ParentBone = parentBone;
        Pivot = pivot;
        Translation = translation;
        Rotation = rotation;
        Scaling = scaling;
        LocalTransform = localTransform;
        WorldTransform = worldTransform;
    }

    public int BoneIndex { get; }

    public short ParentBone { get; }

    public Vector3 Pivot { get; }

    public Vector3 Translation { get; }

    public Quaternion Rotation { get; }

    public Vector3 Scaling { get; }

    public Matrix4x4 LocalTransform { get; }

    public Matrix4x4 WorldTransform { get; }
}

public static class M2BonePoseEvaluator
{
    public static M2BonePoseState Evaluate(
        M2ModelDocument model,
        int sequenceIndex,
        int timeMs,
        M2ExternalAnimationRuntimeState? externalAnimationState = null)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (sequenceIndex < 0 || sequenceIndex >= model.Sequences.Count)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndex), $"Sequence index {sequenceIndex} is out of range for model '{model.Identity.CanonicalModelPath}'.");

        if (externalAnimationState is not null && externalAnimationState.RequestedSequenceIndex != sequenceIndex)
        {
            throw new ArgumentException(
                $"External animation state targets sequence {externalAnimationState.RequestedSequenceIndex} but pose evaluation requested sequence {sequenceIndex}.",
                nameof(externalAnimationState));
        }

        int resolvedSequenceIndex = externalAnimationState?.ResolvedSequenceIndex ?? sequenceIndex;
        byte[] payload = ResolvePayload(model, externalAnimationState);
        bool usesExternalPayload = !ReferenceEquals(payload, model.RawBytes);
        Matrix4x4[] worldMatrices = new Matrix4x4[model.Bones.Count];
        bool[] solved = new bool[model.Bones.Count];
        M2BonePose?[] poses = new M2BonePose?[model.Bones.Count];

        for (int boneIndex = 0; boneIndex < model.Bones.Count; boneIndex++)
            SolveBone(model, payload, resolvedSequenceIndex, timeMs, boneIndex, worldMatrices, solved, poses);

        return new M2BonePoseState(sequenceIndex, resolvedSequenceIndex, timeMs, usesExternalPayload, poses.Select(static pose => pose!).ToArray());
    }

    private static void SolveBone(
        M2ModelDocument model,
        byte[] payload,
        int sequenceIndex,
        int timeMs,
        int boneIndex,
        Matrix4x4[] worldMatrices,
        bool[] solved,
        M2BonePose?[] poses)
    {
        if (solved[boneIndex])
            return;

        M2BoneDefinition bone = model.Bones[boneIndex];
        Matrix4x4 parentWorld = Matrix4x4.Identity;
        if (bone.ParentBone >= 0 && bone.ParentBone < model.Bones.Count)
        {
            SolveBone(model, payload, sequenceIndex, timeMs, bone.ParentBone, worldMatrices, solved, poses);
            parentWorld = worldMatrices[bone.ParentBone];
        }

        Vector3 translation = M2TrackSampler.SampleVector3(payload, model, sequenceIndex, timeMs, bone.TranslationTrack, Vector3.Zero);
        Quaternion rotation = M2TrackSampler.SampleCompressedQuaternion(payload, model, sequenceIndex, timeMs, bone.RotationTrack, Quaternion.Identity);
        Vector3 scaling = M2TrackSampler.SampleVector3(payload, model, sequenceIndex, timeMs, bone.ScalingTrack, Vector3.One);

        Matrix4x4 local = Matrix4x4.CreateTranslation(-bone.Pivot)
            * Matrix4x4.CreateScale(scaling)
            * Matrix4x4.CreateFromQuaternion(rotation)
            * Matrix4x4.CreateTranslation(bone.Pivot)
            * Matrix4x4.CreateTranslation(translation);
        Matrix4x4 world = local * parentWorld;

        worldMatrices[boneIndex] = world;
        solved[boneIndex] = true;
        poses[boneIndex] = new M2BonePose(bone.Index, bone.ParentBone, bone.Pivot, translation, rotation, scaling, local, world);
    }

    private static byte[] ResolvePayload(M2ModelDocument model, M2ExternalAnimationRuntimeState? externalAnimationState)
    {
        if (externalAnimationState?.UsesExternalFile == true && externalAnimationState.LoadedAnimation is not null)
            return externalAnimationState.LoadedAnimation.Payload;

        return model.RawBytes;
    }
}
