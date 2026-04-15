using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ExternalAnimationRuntimeState
{
    public M2ExternalAnimationRuntimeState(
        M2ModelDocument model,
        int requestedSequenceIndex,
        M2SequenceDefinition requestedSequence,
        int resolvedSequenceIndex,
        M2SequenceDefinition resolvedSequence,
        IReadOnlyList<int> aliasChain,
        M2ExternalAnimationRuntimeStage stage,
        string? companionPath,
        bool usesExternalFile,
        M2ExternalAnimationDocument? loadedAnimation,
        IReadOnlyList<int> readySequenceIndices)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(requestedSequence);
        ArgumentNullException.ThrowIfNull(resolvedSequence);
        ArgumentNullException.ThrowIfNull(aliasChain);
        ArgumentNullException.ThrowIfNull(readySequenceIndices);

        Model = model;
        RequestedSequenceIndex = requestedSequenceIndex;
        RequestedSequence = requestedSequence;
        ResolvedSequenceIndex = resolvedSequenceIndex;
        ResolvedSequence = resolvedSequence;
        AliasChain = aliasChain;
        Stage = stage;
        CompanionPath = companionPath;
        UsesExternalFile = usesExternalFile;
        LoadedAnimation = loadedAnimation;
        ReadySequenceIndices = readySequenceIndices;
    }

    public M2ModelDocument Model { get; }

    public int RequestedSequenceIndex { get; }

    public M2SequenceDefinition RequestedSequence { get; }

    public int ResolvedSequenceIndex { get; }

    public M2SequenceDefinition ResolvedSequence { get; }

    public IReadOnlyList<int> AliasChain { get; }

    public M2ExternalAnimationRuntimeStage Stage { get; }

    public string? CompanionPath { get; }

    public bool UsesExternalFile { get; }

    public M2ExternalAnimationDocument? LoadedAnimation { get; }

    public IReadOnlyList<int> ReadySequenceIndices { get; }
}
