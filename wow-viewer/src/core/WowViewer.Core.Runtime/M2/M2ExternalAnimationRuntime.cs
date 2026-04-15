using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public static class M2ExternalAnimationRuntime
{
    public static M2ExternalAnimationRuntimeState Choose(M2ModelDocument model, int sequenceIndex)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (sequenceIndex < 0 || sequenceIndex >= model.Sequences.Count)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndex), $"Sequence index {sequenceIndex} is out of range for model '{model.Identity.CanonicalModelPath}'.");

        List<int> aliasChain = [];
        HashSet<int> visited = [];
        int resolvedSequenceIndex = sequenceIndex;
        while (true)
        {
            if (!visited.Add(resolvedSequenceIndex))
            {
                throw new InvalidDataException(
                    $"Animation alias chain for '{model.Identity.CanonicalModelPath}' loops at sequence index {resolvedSequenceIndex}.");
            }

            aliasChain.Add(resolvedSequenceIndex);
            M2SequenceDefinition sequence = model.Sequences[resolvedSequenceIndex];
            if (!sequence.IsAlias)
                break;

            if (sequence.AliasNext == ushort.MaxValue || sequence.AliasNext >= model.Sequences.Count)
            {
                throw new InvalidDataException(
                    $"Animation alias chain for '{model.Identity.CanonicalModelPath}' points to invalid sequence index {sequence.AliasNext} from sequence {resolvedSequenceIndex}.");
            }

            resolvedSequenceIndex = sequence.AliasNext;
        }

        M2SequenceDefinition requestedSequence = model.Sequences[sequenceIndex];
        M2SequenceDefinition resolvedSequence = model.Sequences[resolvedSequenceIndex];
        bool usesExternalFile = resolvedSequence.UsesExternalAnimationFile;
        string? companionPath = usesExternalFile
            ? model.Identity.BuildAnimationPath(resolvedSequence.AnimationId, resolvedSequence.VariationIndex)
            : null;

        return new M2ExternalAnimationRuntimeState(
            model,
            sequenceIndex,
            requestedSequence,
            resolvedSequenceIndex,
            resolvedSequence,
            aliasChain,
            M2ExternalAnimationRuntimeStage.Chosen,
            companionPath,
            usesExternalFile,
            loadedAnimation: null,
            readySequenceIndices: []);
    }

    public static M2ExternalAnimationRuntimeState Load(M2ExternalAnimationRuntimeState state, M2ExternalAnimationDocument animation)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(animation);

        if (state.Stage != M2ExternalAnimationRuntimeStage.Chosen)
            throw new InvalidOperationException($"Cannot load an external animation from stage '{state.Stage}'. Expected '{M2ExternalAnimationRuntimeStage.Chosen}'.");

        if (!state.UsesExternalFile || string.IsNullOrWhiteSpace(state.CompanionPath))
            throw new InvalidOperationException("Cannot load an external animation for a sequence that does not require a .anim companion.");

        if (!M2ModelIdentity.PathsEqual(state.CompanionPath, animation.SourcePath))
        {
            throw new InvalidDataException(
                $"Loaded animation path '{animation.SourcePath}' does not match the exact selected companion '{state.CompanionPath}'.");
        }

        IReadOnlyList<int> readySequenceIndices = state.AliasChain.Distinct().OrderBy(static value => value).ToArray();
        return new M2ExternalAnimationRuntimeState(
            state.Model,
            state.RequestedSequenceIndex,
            state.RequestedSequence,
            state.ResolvedSequenceIndex,
            state.ResolvedSequence,
            state.AliasChain,
            M2ExternalAnimationRuntimeStage.Loaded,
            state.CompanionPath,
            usesExternalFile: true,
            animation,
            readySequenceIndices);
    }
}
