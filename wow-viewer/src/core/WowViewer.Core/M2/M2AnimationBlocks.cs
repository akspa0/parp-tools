using System.Numerics;

namespace WowViewer.Core.M2;

public enum M2TrackInterpolation : ushort
{
    None = 0,
    Linear = 1,
    Hermite = 2,
    Bezier = 3,
}

public readonly record struct M2TrackSequenceSlice(uint TimestampCount, uint TimestampOffset, uint ValueCount, uint ValueOffset)
{
    public bool HasData => TimestampCount > 0 && ValueCount > 0;
}

public readonly record struct M2TrackArrayReference(uint Count, uint Offset)
{
    public bool HasData => Count > 0;
}

public sealed class M2TrackDefinition<T>
{
    public M2TrackDefinition(
        M2TrackInterpolation interpolation,
        int globalSequenceIndex,
        M2TrackArrayReference timestampArray,
        M2TrackArrayReference valueArray)
    {
        Interpolation = interpolation;
        GlobalSequenceIndex = globalSequenceIndex;
        TimestampArray = timestampArray;
        ValueArray = valueArray;
    }

    public M2TrackInterpolation Interpolation { get; }

    public int GlobalSequenceIndex { get; }

    public M2TrackArrayReference TimestampArray { get; }

    public M2TrackArrayReference ValueArray { get; }

    public bool UsesGlobalSequence => GlobalSequenceIndex >= 0;
}

public sealed class M2ColorDefinition
{
    public M2ColorDefinition(int index, M2TrackDefinition<Vector3> colorTrack, M2TrackDefinition<short> alphaTrack)
    {
        ArgumentNullException.ThrowIfNull(colorTrack);
        ArgumentNullException.ThrowIfNull(alphaTrack);

        Index = index;
        ColorTrack = colorTrack;
        AlphaTrack = alphaTrack;
    }

    public int Index { get; }

    public M2TrackDefinition<Vector3> ColorTrack { get; }

    public M2TrackDefinition<short> AlphaTrack { get; }
}

public sealed class M2TextureWeightDefinition
{
    public M2TextureWeightDefinition(int index, M2TrackDefinition<short> weightTrack)
    {
        ArgumentNullException.ThrowIfNull(weightTrack);

        Index = index;
        WeightTrack = weightTrack;
    }

    public int Index { get; }

    public M2TrackDefinition<short> WeightTrack { get; }
}

public sealed class M2TextureTransformDefinition
{
    public M2TextureTransformDefinition(
        int index,
        M2TrackDefinition<Vector3> translationTrack,
        M2TrackDefinition<Quaternion> rotationTrack,
        M2TrackDefinition<Vector3> scalingTrack)
    {
        ArgumentNullException.ThrowIfNull(translationTrack);
        ArgumentNullException.ThrowIfNull(rotationTrack);
        ArgumentNullException.ThrowIfNull(scalingTrack);

        Index = index;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
    }

    public int Index { get; }

    public M2TrackDefinition<Vector3> TranslationTrack { get; }

    public M2TrackDefinition<Quaternion> RotationTrack { get; }

    public M2TrackDefinition<Vector3> ScalingTrack { get; }
}

public sealed class M2LightDefinition
{
    public M2LightDefinition(
        int index,
        ushort type,
        short boneIndex,
        Vector3 position,
        M2TrackDefinition<Vector3> ambientColorTrack,
        M2TrackDefinition<float> ambientIntensityTrack,
        M2TrackDefinition<Vector3> diffuseColorTrack,
        M2TrackDefinition<float> diffuseIntensityTrack,
        M2TrackDefinition<float> attenuationStartTrack,
        M2TrackDefinition<float> attenuationEndTrack,
        M2TrackDefinition<byte> visibilityTrack)
    {
        ArgumentNullException.ThrowIfNull(ambientColorTrack);
        ArgumentNullException.ThrowIfNull(ambientIntensityTrack);
        ArgumentNullException.ThrowIfNull(diffuseColorTrack);
        ArgumentNullException.ThrowIfNull(diffuseIntensityTrack);
        ArgumentNullException.ThrowIfNull(attenuationStartTrack);
        ArgumentNullException.ThrowIfNull(attenuationEndTrack);
        ArgumentNullException.ThrowIfNull(visibilityTrack);

        Index = index;
        Type = type;
        BoneIndex = boneIndex;
        Position = position;
        AmbientColorTrack = ambientColorTrack;
        AmbientIntensityTrack = ambientIntensityTrack;
        DiffuseColorTrack = diffuseColorTrack;
        DiffuseIntensityTrack = diffuseIntensityTrack;
        AttenuationStartTrack = attenuationStartTrack;
        AttenuationEndTrack = attenuationEndTrack;
        VisibilityTrack = visibilityTrack;
    }

    public int Index { get; }

    public ushort Type { get; }

    public short BoneIndex { get; }

    public Vector3 Position { get; }

    public M2TrackDefinition<Vector3> AmbientColorTrack { get; }

    public M2TrackDefinition<float> AmbientIntensityTrack { get; }

    public M2TrackDefinition<Vector3> DiffuseColorTrack { get; }

    public M2TrackDefinition<float> DiffuseIntensityTrack { get; }

    public M2TrackDefinition<float> AttenuationStartTrack { get; }

    public M2TrackDefinition<float> AttenuationEndTrack { get; }

    public M2TrackDefinition<byte> VisibilityTrack { get; }
}

public sealed class M2CameraDefinition
{
    public M2CameraDefinition(
        int index,
        int type,
        float? staticFieldOfView,
        float farClip,
        float nearClip,
        M2TrackDefinition<Vector3> positionTrack,
        Vector3 positionBase,
        M2TrackDefinition<Vector3> targetPositionTrack,
        Vector3 targetPositionBase,
        M2TrackDefinition<float> rollTrack,
        M2TrackDefinition<float>? fieldOfViewTrack = null)
    {
        ArgumentNullException.ThrowIfNull(positionTrack);
        ArgumentNullException.ThrowIfNull(targetPositionTrack);
        ArgumentNullException.ThrowIfNull(rollTrack);

        Index = index;
        Type = type;
        StaticFieldOfView = staticFieldOfView;
        FarClip = farClip;
        NearClip = nearClip;
        PositionTrack = positionTrack;
        PositionBase = positionBase;
        TargetPositionTrack = targetPositionTrack;
        TargetPositionBase = targetPositionBase;
        RollTrack = rollTrack;
        FieldOfViewTrack = fieldOfViewTrack;
    }

    public int Index { get; }

    public int Type { get; }

    public float? StaticFieldOfView { get; }

    public float FarClip { get; }

    public float NearClip { get; }

    public M2TrackDefinition<Vector3> PositionTrack { get; }

    public Vector3 PositionBase { get; }

    public M2TrackDefinition<Vector3> TargetPositionTrack { get; }

    public Vector3 TargetPositionBase { get; }

    public M2TrackDefinition<float> RollTrack { get; }

    public M2TrackDefinition<float>? FieldOfViewTrack { get; }

    public bool HasAnimatedFieldOfView => FieldOfViewTrack != null;
}

public readonly record struct M2CompQuaternion(ushort X, ushort Y, ushort Z, ushort W)
{
    public static M2CompQuaternion Identity { get; } = new(32767, 32767, 32767, 65535);

    public Quaternion ToQuaternion()
    {
        static float Decode(ushort value)
        {
            return Math.Clamp(((int)value - 32767) / 32767.0f, -1.0f, 1.0f);
        }

        Quaternion quaternion = new(Decode(X), Decode(Y), Decode(Z), Decode(W));
        return quaternion.LengthSquared() > 0.000001f
            ? Quaternion.Normalize(quaternion)
            : Quaternion.Identity;
    }
}

public sealed class M2BoneDefinition
{
    public M2BoneDefinition(
        int index,
        int keyBoneId,
        uint flags,
        short parentBone,
        ushort submeshId,
        uint boneNameCrc,
        M2TrackDefinition<Vector3> translationTrack,
        M2TrackDefinition<M2CompQuaternion> rotationTrack,
        M2TrackDefinition<Vector3> scalingTrack,
        Vector3 pivot)
    {
        ArgumentNullException.ThrowIfNull(translationTrack);
        ArgumentNullException.ThrowIfNull(rotationTrack);
        ArgumentNullException.ThrowIfNull(scalingTrack);

        Index = index;
        KeyBoneId = keyBoneId;
        Flags = flags;
        ParentBone = parentBone;
        SubmeshId = submeshId;
        BoneNameCrc = boneNameCrc;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        Pivot = pivot;
    }

    public int Index { get; }

    public int KeyBoneId { get; }

    public uint Flags { get; }

    public short ParentBone { get; }

    public ushort SubmeshId { get; }

    public uint BoneNameCrc { get; }

    public M2TrackDefinition<Vector3> TranslationTrack { get; }

    public M2TrackDefinition<M2CompQuaternion> RotationTrack { get; }

    public M2TrackDefinition<Vector3> ScalingTrack { get; }

    public Vector3 Pivot { get; }

    public bool HasParent => ParentBone >= 0;
}

public sealed class M2RibbonDefinition
{
    public M2RibbonDefinition(
        int index,
        uint ribbonId,
        uint boneIndex,
        Vector3 position,
        IReadOnlyList<ushort> textureIndices,
        IReadOnlyList<ushort> materialIndices,
        M2TrackDefinition<Vector3> colorTrack,
        M2TrackDefinition<short> alphaTrack,
        M2TrackDefinition<float> heightAboveTrack,
        M2TrackDefinition<float> heightBelowTrack,
        float edgesPerSecond,
        float edgeLifetime,
        float gravity,
        ushort textureRows,
        ushort textureColumns,
        M2TrackDefinition<ushort> textureSlotTrack,
        M2TrackDefinition<byte> visibilityTrack,
        short priorityPlane,
        sbyte ribbonColorIndex,
        sbyte textureTransformLookupIndex)
    {
        ArgumentNullException.ThrowIfNull(textureIndices);
        ArgumentNullException.ThrowIfNull(materialIndices);
        ArgumentNullException.ThrowIfNull(colorTrack);
        ArgumentNullException.ThrowIfNull(alphaTrack);
        ArgumentNullException.ThrowIfNull(heightAboveTrack);
        ArgumentNullException.ThrowIfNull(heightBelowTrack);
        ArgumentNullException.ThrowIfNull(textureSlotTrack);
        ArgumentNullException.ThrowIfNull(visibilityTrack);

        Index = index;
        RibbonId = ribbonId;
        BoneIndex = boneIndex;
        Position = position;
        TextureIndices = textureIndices;
        MaterialIndices = materialIndices;
        ColorTrack = colorTrack;
        AlphaTrack = alphaTrack;
        HeightAboveTrack = heightAboveTrack;
        HeightBelowTrack = heightBelowTrack;
        EdgesPerSecond = edgesPerSecond;
        EdgeLifetime = edgeLifetime;
        Gravity = gravity;
        TextureRows = textureRows;
        TextureColumns = textureColumns;
        TextureSlotTrack = textureSlotTrack;
        VisibilityTrack = visibilityTrack;
        PriorityPlane = priorityPlane;
        RibbonColorIndex = ribbonColorIndex;
        TextureTransformLookupIndex = textureTransformLookupIndex;
    }

    public int Index { get; }

    public uint RibbonId { get; }

    public uint BoneIndex { get; }

    public Vector3 Position { get; }

    public IReadOnlyList<ushort> TextureIndices { get; }

    public IReadOnlyList<ushort> MaterialIndices { get; }

    public M2TrackDefinition<Vector3> ColorTrack { get; }

    public M2TrackDefinition<short> AlphaTrack { get; }

    public M2TrackDefinition<float> HeightAboveTrack { get; }

    public M2TrackDefinition<float> HeightBelowTrack { get; }

    public float EdgesPerSecond { get; }

    public float EdgeLifetime { get; }

    public float Gravity { get; }

    public ushort TextureRows { get; }

    public ushort TextureColumns { get; }

    public M2TrackDefinition<ushort> TextureSlotTrack { get; }

    public M2TrackDefinition<byte> VisibilityTrack { get; }

    public short PriorityPlane { get; }

    public sbyte RibbonColorIndex { get; }

    public sbyte TextureTransformLookupIndex { get; }
}

public sealed class M2ParticleDefinition
{
    public M2ParticleDefinition(
        int index,
        uint particleId,
        uint flags,
        Vector3 position,
        ushort boneIndex,
        ushort textureIndex,
        string? geometryModelPath,
        string? recursionModelPath,
        ushort blendingType,
        ushort emitterType,
        ushort particleColorIndex,
        byte particleType,
        byte headOrTail,
        short textureTileRotation,
        ushort textureRows,
        ushort textureColumns,
        M2TrackDefinition<float> emissionSpeedTrack,
        M2TrackDefinition<float> speedVariationTrack,
        M2TrackDefinition<float> verticalRangeTrack,
        M2TrackDefinition<float> horizontalRangeTrack,
        M2TrackDefinition<float> gravityTrack,
        M2TrackDefinition<float> lifespanTrack,
        M2TrackDefinition<float> emissionRateTrack,
        M2TrackDefinition<float> emissionAreaLengthTrack,
        M2TrackDefinition<float> emissionAreaWidthTrack,
        M2TrackDefinition<float> zSourceTrack,
        M2TrackDefinition<byte> enabledTrack)
    {
        ArgumentNullException.ThrowIfNull(emissionSpeedTrack);
        ArgumentNullException.ThrowIfNull(speedVariationTrack);
        ArgumentNullException.ThrowIfNull(verticalRangeTrack);
        ArgumentNullException.ThrowIfNull(horizontalRangeTrack);
        ArgumentNullException.ThrowIfNull(gravityTrack);
        ArgumentNullException.ThrowIfNull(lifespanTrack);
        ArgumentNullException.ThrowIfNull(emissionRateTrack);
        ArgumentNullException.ThrowIfNull(emissionAreaLengthTrack);
        ArgumentNullException.ThrowIfNull(emissionAreaWidthTrack);
        ArgumentNullException.ThrowIfNull(zSourceTrack);
        ArgumentNullException.ThrowIfNull(enabledTrack);

        Index = index;
        ParticleId = particleId;
        Flags = flags;
        Position = position;
        BoneIndex = boneIndex;
        TextureIndex = textureIndex;
        GeometryModelPath = geometryModelPath;
        RecursionModelPath = recursionModelPath;
        BlendingType = blendingType;
        EmitterType = emitterType;
        ParticleColorIndex = particleColorIndex;
        ParticleType = particleType;
        HeadOrTail = headOrTail;
        TextureTileRotation = textureTileRotation;
        TextureRows = textureRows;
        TextureColumns = textureColumns;
        EmissionSpeedTrack = emissionSpeedTrack;
        SpeedVariationTrack = speedVariationTrack;
        VerticalRangeTrack = verticalRangeTrack;
        HorizontalRangeTrack = horizontalRangeTrack;
        GravityTrack = gravityTrack;
        LifespanTrack = lifespanTrack;
        EmissionRateTrack = emissionRateTrack;
        EmissionAreaLengthTrack = emissionAreaLengthTrack;
        EmissionAreaWidthTrack = emissionAreaWidthTrack;
        ZSourceTrack = zSourceTrack;
        EnabledTrack = enabledTrack;
    }

    public int Index { get; }

    public uint ParticleId { get; }

    public uint Flags { get; }

    public Vector3 Position { get; }

    public ushort BoneIndex { get; }

    public ushort TextureIndex { get; }

    public string? GeometryModelPath { get; }

    public string? RecursionModelPath { get; }

    public ushort BlendingType { get; }

    public ushort EmitterType { get; }

    public ushort ParticleColorIndex { get; }

    public byte ParticleType { get; }

    public byte HeadOrTail { get; }

    public short TextureTileRotation { get; }

    public ushort TextureRows { get; }

    public ushort TextureColumns { get; }

    public M2TrackDefinition<float> EmissionSpeedTrack { get; }

    public M2TrackDefinition<float> SpeedVariationTrack { get; }

    public M2TrackDefinition<float> VerticalRangeTrack { get; }

    public M2TrackDefinition<float> HorizontalRangeTrack { get; }

    public M2TrackDefinition<float> GravityTrack { get; }

    public M2TrackDefinition<float> LifespanTrack { get; }

    public M2TrackDefinition<float> EmissionRateTrack { get; }

    public M2TrackDefinition<float> EmissionAreaLengthTrack { get; }

    public M2TrackDefinition<float> EmissionAreaWidthTrack { get; }

    public M2TrackDefinition<float> ZSourceTrack { get; }

    public M2TrackDefinition<byte> EnabledTrack { get; }

    public bool UsesModelParticle => !string.IsNullOrWhiteSpace(GeometryModelPath);

    public bool UsesRecursiveParticleModel => !string.IsNullOrWhiteSpace(RecursionModelPath);
}
