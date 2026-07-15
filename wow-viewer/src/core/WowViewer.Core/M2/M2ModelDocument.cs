using System.Numerics;

namespace WowViewer.Core.M2;

public sealed class M2ModelDocument
{
    public M2ModelDocument(
        M2ModelIdentity identity,
        byte[] rawBytes,
        string signature,
        uint version,
        uint flags,
        uint viewCount,
        string? modelName,
        IReadOnlyList<uint> globalLoops,
        IReadOnlyList<M2SequenceDefinition> sequences,
        IReadOnlyList<short> sequenceLookup,
        IReadOnlyList<M2ColorDefinition> colors,
        IReadOnlyList<M2TextureWeightDefinition> textureWeights,
        IReadOnlyList<M2TextureTransformDefinition> textureTransforms,
        IReadOnlyList<M2LightDefinition> lights,
        IReadOnlyList<M2CameraDefinition>? cameras,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        uint embeddedSkinProfileCount,
        uint embeddedSkinProfileOffset,
        IReadOnlyList<M2BoneDefinition>? bones = null,
        IReadOnlyList<M2RibbonDefinition>? ribbons = null,
        IReadOnlyList<M2ParticleDefinition>? particles = null)
    {
        ArgumentNullException.ThrowIfNull(identity);
        ArgumentNullException.ThrowIfNull(rawBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentNullException.ThrowIfNull(globalLoops);
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(sequenceLookup);
        ArgumentNullException.ThrowIfNull(colors);
        ArgumentNullException.ThrowIfNull(textureWeights);
        ArgumentNullException.ThrowIfNull(textureTransforms);
        ArgumentNullException.ThrowIfNull(lights);

        Identity = identity;
        RawBytes = rawBytes;
        Signature = signature;
        Version = version;
        Flags = flags;
        ViewCount = viewCount;
        ModelName = modelName;
        GlobalLoops = globalLoops;
        Sequences = sequences;
        SequenceLookup = sequenceLookup;
        Colors = colors;
        TextureWeights = textureWeights;
        TextureTransforms = textureTransforms;
        Lights = lights;
        Cameras = cameras ?? [];
        Bones = bones ?? [];
        Ribbons = ribbons ?? [];
        Particles = particles ?? [];
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        BoundsRadius = boundsRadius;
        EmbeddedSkinProfileCount = embeddedSkinProfileCount;
        EmbeddedSkinProfileOffset = embeddedSkinProfileOffset;
        InlineEra1121Geometry = null;
    }

    public M2ModelIdentity Identity { get; }

    public byte[] RawBytes { get; }

    public string Signature { get; }

    public uint Version { get; }

    public uint Flags { get; }

    public uint ViewCount { get; }

    public string? ModelName { get; }

    public IReadOnlyList<uint> GlobalLoops { get; }

    public IReadOnlyList<M2SequenceDefinition> Sequences { get; }

    public IReadOnlyList<short> SequenceLookup { get; }

    public IReadOnlyList<M2ColorDefinition> Colors { get; }

    public IReadOnlyList<M2TextureWeightDefinition> TextureWeights { get; }

    public IReadOnlyList<M2TextureTransformDefinition> TextureTransforms { get; }

    public IReadOnlyList<M2LightDefinition> Lights { get; }

    public IReadOnlyList<M2CameraDefinition> Cameras { get; }

    public IReadOnlyList<M2BoneDefinition> Bones { get; }

    public IReadOnlyList<M2RibbonDefinition> Ribbons { get; }

    public IReadOnlyList<M2ParticleDefinition> Particles { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public float BoundsRadius { get; }

    public uint EmbeddedSkinProfileCount { get; }

    public uint EmbeddedSkinProfileOffset { get; }

    public M2Era1121Geometry? InlineEra1121Geometry { get; set; }

    /// <summary>
    /// Resolved geometry from a WoW 1.0.0 (build 3980) M2 model's embedded division.
    /// Populated by <see cref="WowViewer.Core.IO.M2Era100.M2Era100ModelReader"/>.
    /// </summary>
    public M2Era100Geometry? InlineEra100Geometry { get; set; }

    public int GlobalLoopCount => GlobalLoops.Count;

    public int SequenceCount => Sequences.Count;

    public int SequenceLookupCount => SequenceLookup.Count;

    public int ColorCount => Colors.Count;

    public int TextureWeightCount => TextureWeights.Count;

    public int TextureTransformCount => TextureTransforms.Count;

    public int LightCount => Lights.Count;

    public int CameraCount => Cameras.Count;

    public int BoneCount => Bones.Count;

    public int RibbonCount => Ribbons.Count;

    public int ParticleCount => Particles.Count;

    public bool HasEmbeddedSkinProfiles => EmbeddedSkinProfileCount > 0;

    public bool HasPhysicsSidecar => (Flags & 0x20u) != 0;
}
