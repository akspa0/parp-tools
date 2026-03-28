using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxLightSummary
{
    public MdxLightSummary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        MdxLightType lightType,
        float staticAttenuationStart,
        float staticAttenuationEnd,
        Vector3 staticColor,
        float staticIntensity,
        Vector3 staticAmbientColor,
        float staticAmbientIntensity,
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack,
        MdxTrackSummary? attenuationStartTrack,
        MdxTrackSummary? attenuationEndTrack,
        MdxTrackSummary? colorTrack,
        MdxTrackSummary? intensityTrack,
        MdxTrackSummary? ambientColorTrack,
        MdxTrackSummary? ambientIntensityTrack,
        MdxVisibilityTrackSummary? visibilityTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(objectId);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        LightType = lightType;
        StaticAttenuationStart = staticAttenuationStart;
        StaticAttenuationEnd = staticAttenuationEnd;
        StaticColor = staticColor;
        StaticIntensity = staticIntensity;
        StaticAmbientColor = staticAmbientColor;
        StaticAmbientIntensity = staticAmbientIntensity;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        AttenuationStartTrack = attenuationStartTrack;
        AttenuationEndTrack = attenuationEndTrack;
        ColorTrack = colorTrack;
        IntensityTrack = intensityTrack;
        AmbientColorTrack = ambientColorTrack;
        AmbientIntensityTrack = ambientIntensityTrack;
        VisibilityTrack = visibilityTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public MdxLightType LightType { get; }

    public float StaticAttenuationStart { get; }

    public float StaticAttenuationEnd { get; }

    public Vector3 StaticColor { get; }

    public float StaticIntensity { get; }

    public Vector3 StaticAmbientColor { get; }

    public float StaticAmbientIntensity { get; }

    public bool HasParent => ParentId >= 0;

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }

    public MdxTrackSummary? AttenuationStartTrack { get; }

    public MdxTrackSummary? AttenuationEndTrack { get; }

    public MdxTrackSummary? ColorTrack { get; }

    public MdxTrackSummary? IntensityTrack { get; }

    public MdxTrackSummary? AmbientColorTrack { get; }

    public MdxTrackSummary? AmbientIntensityTrack { get; }

    public MdxVisibilityTrackSummary? VisibilityTrack { get; }
}