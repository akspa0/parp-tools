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