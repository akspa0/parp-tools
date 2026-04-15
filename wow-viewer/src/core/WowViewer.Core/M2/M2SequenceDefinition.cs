using System.Numerics;

namespace WowViewer.Core.M2;

[Flags]
public enum M2SequenceFlags : uint
{
    None = 0,
    RuntimeLoaded = 0x10,
    StoredInline = 0x20,
    Alias = 0x40,
    Unknown0x100 = 0x100,
}

public sealed class M2SequenceDefinition
{
    private const uint ExternalAnimationMask = 0x130;

    public M2SequenceDefinition(
        int index,
        ushort animationId,
        ushort variationIndex,
        uint duration,
        float moveSpeed,
        uint flags,
        short frequency,
        uint replayMinimum,
        uint replayMaximum,
        ushort blendTimeIn,
        ushort blendTimeOut,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float boundsRadius,
        short variationNext,
        ushort aliasNext)
    {
        Index = index;
        AnimationId = animationId;
        VariationIndex = variationIndex;
        Duration = duration;
        MoveSpeed = moveSpeed;
        Flags = flags;
        Frequency = frequency;
        ReplayMinimum = replayMinimum;
        ReplayMaximum = replayMaximum;
        BlendTimeIn = blendTimeIn;
        BlendTimeOut = blendTimeOut;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        BoundsRadius = boundsRadius;
        VariationNext = variationNext;
        AliasNext = aliasNext;
    }

    public int Index { get; }

    public ushort AnimationId { get; }

    public ushort VariationIndex { get; }

    public uint Duration { get; }

    public float MoveSpeed { get; }

    public uint Flags { get; }

    public M2SequenceFlags FlagsValue => (M2SequenceFlags)Flags;

    public short Frequency { get; }

    public uint ReplayMinimum { get; }

    public uint ReplayMaximum { get; }

    public ushort BlendTimeIn { get; }

    public ushort BlendTimeOut { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public float BoundsRadius { get; }

    public short VariationNext { get; }

    public ushort AliasNext { get; }

    public bool HasRuntimeLoadedFlag => (Flags & (uint)M2SequenceFlags.RuntimeLoaded) != 0;

    public bool UsesInlineAnimationData => (Flags & (uint)M2SequenceFlags.StoredInline) != 0;

    public bool IsAlias => (Flags & (uint)M2SequenceFlags.Alias) != 0;

    public bool UsesExternalAnimationFile => (Flags & ExternalAnimationMask) == 0;
}
