using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxSummary
{
    public MdxSummary(
        string sourcePath,
        string signature,
        uint? version,
        string? modelName,
        uint? blendTime,
        Vector3? boundsMin,
        Vector3? boundsMax,
        IReadOnlyList<MdxGlobalSequenceSummary> globalSequences,
        IReadOnlyList<MdxSequenceSummary> sequences,
        IReadOnlyList<MdxGeosetSummary> geosets,
        IReadOnlyList<MdxGeosetAnimationSummary> geosetAnimations,
        IReadOnlyList<MdxBoneSummary> bones,
        IReadOnlyList<MdxHelperSummary> helpers,
        IReadOnlyList<MdxAttachmentSummary> attachments,
        IReadOnlyList<MdxParticleEmitter2Summary> particleEmitters2,
        IReadOnlyList<MdxRibbonEmitterSummary> ribbons,
        IReadOnlyList<MdxCameraSummary> cameras,
        IReadOnlyList<MdxEventSummary> events,
        IReadOnlyList<MdxHitTestShapeSummary> hitTestShapes,
        MdxCollisionSummary? collision,
        IReadOnlyList<MdxPivotPointSummary> pivotPoints,
        IReadOnlyList<MdxTextureSummary> textures,
        IReadOnlyList<MdxMaterialSummary> materials,
        IReadOnlyList<MdxChunkSummary> chunks,
        int knownChunkCount,
        int unknownChunkCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(globalSequences);
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(geosets);
        ArgumentNullException.ThrowIfNull(geosetAnimations);
        ArgumentNullException.ThrowIfNull(bones);
        ArgumentNullException.ThrowIfNull(helpers);
        ArgumentNullException.ThrowIfNull(attachments);
        ArgumentNullException.ThrowIfNull(particleEmitters2);
        ArgumentNullException.ThrowIfNull(ribbons);
        ArgumentNullException.ThrowIfNull(cameras);
        ArgumentNullException.ThrowIfNull(events);
        ArgumentNullException.ThrowIfNull(hitTestShapes);
        ArgumentNullException.ThrowIfNull(pivotPoints);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentNullException.ThrowIfNull(materials);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentOutOfRangeException.ThrowIfNegative(knownChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(unknownChunkCount);

        SourcePath = sourcePath;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        BlendTime = blendTime;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        GlobalSequences = globalSequences;
        GlobalSequenceCount = globalSequences.Count;
        Sequences = sequences;
        SequenceCount = sequences.Count;
        Geosets = geosets;
        GeosetCount = geosets.Count;
        GeosetAnimations = geosetAnimations;
        GeosetAnimationCount = geosetAnimations.Count;
        Bones = bones;
        BoneCount = bones.Count;
        Helpers = helpers;
        HelperCount = helpers.Count;
        Attachments = attachments;
        AttachmentCount = attachments.Count;
        ParticleEmitters2 = particleEmitters2;
        ParticleEmitter2Count = particleEmitters2.Count;
        Ribbons = ribbons;
        RibbonCount = ribbons.Count;
        Cameras = cameras;
        CameraCount = cameras.Count;
        Events = events;
        EventCount = events.Count;
        HitTestShapes = hitTestShapes;
        HitTestShapeCount = hitTestShapes.Count;
        Collision = collision;
        PivotPoints = pivotPoints;
        PivotPointCount = pivotPoints.Count;
        Textures = textures;
        TextureCount = textures.Count;
        ReplaceableTextureCount = textures.Count(static texture => texture.IsReplaceable);
        Materials = materials;
        MaterialCount = materials.Count;
        MaterialLayerCount = materials.Sum(static material => material.LayerCount);
        Chunks = chunks;
        ChunkCount = chunks.Count;
        KnownChunkCount = knownChunkCount;
        UnknownChunkCount = unknownChunkCount;
    }

    public string SourcePath { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public string? ModelName { get; }

    public uint? BlendTime { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }

    public IReadOnlyList<MdxGlobalSequenceSummary> GlobalSequences { get; }

    public int GlobalSequenceCount { get; }

    public IReadOnlyList<MdxSequenceSummary> Sequences { get; }

    public int SequenceCount { get; }

    public IReadOnlyList<MdxGeosetSummary> Geosets { get; }

    public int GeosetCount { get; }

    public IReadOnlyList<MdxGeosetAnimationSummary> GeosetAnimations { get; }

    public int GeosetAnimationCount { get; }

    public IReadOnlyList<MdxBoneSummary> Bones { get; }

    public int BoneCount { get; }

    public IReadOnlyList<MdxHelperSummary> Helpers { get; }

    public int HelperCount { get; }

    public IReadOnlyList<MdxAttachmentSummary> Attachments { get; }

    public int AttachmentCount { get; }

    public IReadOnlyList<MdxParticleEmitter2Summary> ParticleEmitters2 { get; }

    public int ParticleEmitter2Count { get; }

    public IReadOnlyList<MdxRibbonEmitterSummary> Ribbons { get; }

    public int RibbonCount { get; }

    public IReadOnlyList<MdxCameraSummary> Cameras { get; }

    public int CameraCount { get; }

    public IReadOnlyList<MdxEventSummary> Events { get; }

    public int EventCount { get; }

    public IReadOnlyList<MdxHitTestShapeSummary> HitTestShapes { get; }

    public int HitTestShapeCount { get; }

    public MdxCollisionSummary? Collision { get; }

    public bool HasCollision => Collision is not null;

    public IReadOnlyList<MdxPivotPointSummary> PivotPoints { get; }

    public int PivotPointCount { get; }

    public IReadOnlyList<MdxTextureSummary> Textures { get; }

    public int TextureCount { get; }

    public int ReplaceableTextureCount { get; }

    public IReadOnlyList<MdxMaterialSummary> Materials { get; }

    public int MaterialCount { get; }

    public int MaterialLayerCount { get; }

    public IReadOnlyList<MdxChunkSummary> Chunks { get; }

    public int ChunkCount { get; }

    public int KnownChunkCount { get; }

    public int UnknownChunkCount { get; }
}