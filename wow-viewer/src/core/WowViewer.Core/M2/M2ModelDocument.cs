using System.Numerics;

namespace WowViewer.Core.M2;

public sealed class M2ModelDocument
{
    public M2ModelDocument(
        M2ModelIdentity identity,
        string signature,
        uint version,
        string? modelName,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        uint embeddedSkinProfileCount,
        uint embeddedSkinProfileOffset)
    {
        ArgumentNullException.ThrowIfNull(identity);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);

        Identity = identity;
        Signature = signature;
        Version = version;
        ModelName = modelName;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        BoundsRadius = boundsRadius;
        EmbeddedSkinProfileCount = embeddedSkinProfileCount;
        EmbeddedSkinProfileOffset = embeddedSkinProfileOffset;
    }

    public M2ModelIdentity Identity { get; }

    public string Signature { get; }

    public uint Version { get; }

    public string? ModelName { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public float BoundsRadius { get; }

    public uint EmbeddedSkinProfileCount { get; }

    public uint EmbeddedSkinProfileOffset { get; }

    public bool HasEmbeddedSkinProfiles => EmbeddedSkinProfileCount > 0;
}