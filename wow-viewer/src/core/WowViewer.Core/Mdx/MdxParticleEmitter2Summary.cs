using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxParticleEmitter2Summary
{
    public MdxParticleEmitter2Summary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        int emitterType,
        float staticSpeed,
        float staticVariation,
        float staticLatitude,
        float staticLongitude,
        float staticGravity,
        float staticZSource,
        float staticLife,
        float staticEmissionRate,
        float staticLength,
        float staticWidth,
        uint rows,
        uint columns,
        uint particleType,
        float tailLength,
        float middleTime,
        Vector3 startColor,
        Vector3 middleColor,
        Vector3 endColor,
        byte startAlpha,
        byte middleAlpha,
        byte endAlpha,
        float startScale,
        float middleScale,
        float endScale,
        uint blendMode,
        int textureId,
        int priorityPlane,
        uint replaceableId,
        string? geometryModel,
        string? recursionModel,
        uint splineCount,
        int squirts,
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack,
        MdxVisibilityTrackSummary? visibilityTrack,
        MdxTrackSummary? speedTrack,
        MdxTrackSummary? variationTrack,
        MdxTrackSummary? latitudeTrack,
        MdxTrackSummary? longitudeTrack,
        MdxTrackSummary? gravityTrack,
        MdxTrackSummary? lifeTrack,
        MdxTrackSummary? emissionRateTrack,
        MdxTrackSummary? widthTrack,
        MdxTrackSummary? lengthTrack,
        MdxTrackSummary? zSourceTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(objectId);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        EmitterType = emitterType;
        StaticSpeed = staticSpeed;
        StaticVariation = staticVariation;
        StaticLatitude = staticLatitude;
        StaticLongitude = staticLongitude;
        StaticGravity = staticGravity;
        StaticZSource = staticZSource;
        StaticLife = staticLife;
        StaticEmissionRate = staticEmissionRate;
        StaticLength = staticLength;
        StaticWidth = staticWidth;
        Rows = rows;
        Columns = columns;
        ParticleType = particleType;
        TailLength = tailLength;
        MiddleTime = middleTime;
        StartColor = startColor;
        MiddleColor = middleColor;
        EndColor = endColor;
        StartAlpha = startAlpha;
        MiddleAlpha = middleAlpha;
        EndAlpha = endAlpha;
        StartScale = startScale;
        MiddleScale = middleScale;
        EndScale = endScale;
        BlendMode = blendMode;
        TextureId = textureId;
        PriorityPlane = priorityPlane;
        ReplaceableId = replaceableId;
        GeometryModel = string.IsNullOrWhiteSpace(geometryModel) ? null : geometryModel;
        RecursionModel = string.IsNullOrWhiteSpace(recursionModel) ? null : recursionModel;
        SplineCount = splineCount;
        Squirts = squirts;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        VisibilityTrack = visibilityTrack;
        SpeedTrack = speedTrack;
        VariationTrack = variationTrack;
        LatitudeTrack = latitudeTrack;
        LongitudeTrack = longitudeTrack;
        GravityTrack = gravityTrack;
        LifeTrack = lifeTrack;
        EmissionRateTrack = emissionRateTrack;
        WidthTrack = widthTrack;
        LengthTrack = lengthTrack;
        ZSourceTrack = zSourceTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public int EmitterType { get; }

    public float StaticSpeed { get; }

    public float StaticVariation { get; }

    public float StaticLatitude { get; }

    public float StaticLongitude { get; }

    public float StaticGravity { get; }

    public float StaticZSource { get; }

    public float StaticLife { get; }

    public float StaticEmissionRate { get; }

    public float StaticLength { get; }

    public float StaticWidth { get; }

    public uint Rows { get; }

    public uint Columns { get; }

    public uint ParticleType { get; }

    public float TailLength { get; }

    public float MiddleTime { get; }

    public Vector3 StartColor { get; }

    public Vector3 MiddleColor { get; }

    public Vector3 EndColor { get; }

    public byte StartAlpha { get; }

    public byte MiddleAlpha { get; }

    public byte EndAlpha { get; }

    public float StartScale { get; }

    public float MiddleScale { get; }

    public float EndScale { get; }

    public uint BlendMode { get; }

    public int TextureId { get; }

    public int PriorityPlane { get; }

    public uint ReplaceableId { get; }

    public string? GeometryModel { get; }

    public string? RecursionModel { get; }

    public uint SplineCount { get; }

    public int Squirts { get; }

    public bool HasParent => ParentId >= 0;

    public bool HasGeometryModel => !string.IsNullOrWhiteSpace(GeometryModel);

    public bool HasRecursionModel => !string.IsNullOrWhiteSpace(RecursionModel);

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }

    public MdxVisibilityTrackSummary? VisibilityTrack { get; }

    public MdxTrackSummary? SpeedTrack { get; }

    public MdxTrackSummary? VariationTrack { get; }

    public MdxTrackSummary? LatitudeTrack { get; }

    public MdxTrackSummary? LongitudeTrack { get; }

    public MdxTrackSummary? GravityTrack { get; }

    public MdxTrackSummary? LifeTrack { get; }

    public MdxTrackSummary? EmissionRateTrack { get; }

    public MdxTrackSummary? WidthTrack { get; }

    public MdxTrackSummary? LengthTrack { get; }

    public MdxTrackSummary? ZSourceTrack { get; }
}