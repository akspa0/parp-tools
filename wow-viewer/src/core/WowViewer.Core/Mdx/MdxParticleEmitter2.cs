using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxParticleEmitter2
{
    public MdxParticleEmitter2(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        Vector3 pivotPoint,
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
        IReadOnlyList<uint> unknownIntervals,
        uint blendMode,
        int textureId,
        int priorityPlane,
        uint replaceableId,
        string? geometryModel,
        string? recursionModel,
        IReadOnlyList<float> unknownFloatBlockA,
        IReadOnlyList<float> unknownTumbleValues,
        IReadOnlyList<float> unknownFloatBlockB,
        Vector3 unknownVector,
        IReadOnlyList<float> unknownFloatBlockC,
        uint splineCount,
        IReadOnlyList<Vector3> splinePoints,
        int squirts,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack,
        MdxScalarTrack? visibilityTrack,
        MdxScalarTrack? speedTrack,
        MdxScalarTrack? variationTrack,
        MdxScalarTrack? latitudeTrack,
        MdxScalarTrack? longitudeTrack,
        MdxScalarTrack? gravityTrack,
        MdxScalarTrack? lifeTrack,
        MdxScalarTrack? emissionRateTrack,
        MdxScalarTrack? widthTrack,
        MdxScalarTrack? lengthTrack,
        MdxScalarTrack? zSourceTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(unknownIntervals);
        ArgumentNullException.ThrowIfNull(unknownFloatBlockA);
        ArgumentNullException.ThrowIfNull(unknownTumbleValues);
        ArgumentNullException.ThrowIfNull(unknownFloatBlockB);
        ArgumentNullException.ThrowIfNull(unknownFloatBlockC);
        ArgumentNullException.ThrowIfNull(splinePoints);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        PivotPoint = pivotPoint;
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
        UnknownIntervals = unknownIntervals;
        BlendMode = blendMode;
        TextureId = textureId;
        PriorityPlane = priorityPlane;
        ReplaceableId = replaceableId;
        GeometryModel = string.IsNullOrWhiteSpace(geometryModel) ? null : geometryModel;
        RecursionModel = string.IsNullOrWhiteSpace(recursionModel) ? null : recursionModel;
        UnknownFloatBlockA = unknownFloatBlockA;
        UnknownTumbleValues = unknownTumbleValues;
        UnknownFloatBlockB = unknownFloatBlockB;
        UnknownVector = unknownVector;
        UnknownFloatBlockC = unknownFloatBlockC;
        SplineCount = splineCount;
        SplinePoints = splinePoints;
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

    public Vector3 PivotPoint { get; }

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

    public IReadOnlyList<uint> UnknownIntervals { get; }

    public uint BlendMode { get; }

    public int TextureId { get; }

    public int PriorityPlane { get; }

    public uint ReplaceableId { get; }

    public string? GeometryModel { get; }

    public string? RecursionModel { get; }

    public IReadOnlyList<float> UnknownFloatBlockA { get; }

    public IReadOnlyList<float> UnknownTumbleValues { get; }

    public IReadOnlyList<float> UnknownFloatBlockB { get; }

    public Vector3 UnknownVector { get; }

    public IReadOnlyList<float> UnknownFloatBlockC { get; }

    public uint SplineCount { get; }

    public IReadOnlyList<Vector3> SplinePoints { get; }

    public int Squirts { get; }

    public bool HasParent => ParentId >= 0;

    public bool HasGeometryModel => !string.IsNullOrWhiteSpace(GeometryModel);

    public bool HasRecursionModel => !string.IsNullOrWhiteSpace(RecursionModel);

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public MdxScalarTrack? VisibilityTrack { get; }

    public MdxScalarTrack? SpeedTrack { get; }

    public MdxScalarTrack? VariationTrack { get; }

    public MdxScalarTrack? LatitudeTrack { get; }

    public MdxScalarTrack? LongitudeTrack { get; }

    public MdxScalarTrack? GravityTrack { get; }

    public MdxScalarTrack? LifeTrack { get; }

    public MdxScalarTrack? EmissionRateTrack { get; }

    public MdxScalarTrack? WidthTrack { get; }

    public MdxScalarTrack? LengthTrack { get; }

    public MdxScalarTrack? ZSourceTrack { get; }
}