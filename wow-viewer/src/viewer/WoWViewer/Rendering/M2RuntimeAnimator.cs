using WoWViewer.DataSources;
using WoWViewer.Logging;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

internal sealed class M2RuntimeAnimator : IAnimationController
{
    private readonly M2ModelDocument _model;
    private readonly IDataSource? _dataSource;
    private readonly AnimationSequenceDescriptor[] _sequences;
    private readonly Dictionary<int, M2ExternalAnimationRuntimeState?> _externalStates = new();
    private readonly HashSet<string> _loggedAnimationPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, string> _sequenceFailures = new();
    private readonly HashSet<int> _loggedSequenceFailures = new();
    private int _sequenceIndex;
    private float _currentFrame;

    public M2RuntimeAnimator(M2ModelDocument model, IDataSource? dataSource)
    {
        ArgumentNullException.ThrowIfNull(model);

        _model = model;
        _dataSource = dataSource;
        _sequences = model.Sequences
            .Select(static sequence => new AnimationSequenceDescriptor(
                sequence.Index,
                M2AnimationNameResolver.GetSequenceDisplayName(sequence.AnimationId, sequence.VariationIndex),
                new AnimationTimeRange(0.0f, sequence.Duration)))
            .ToArray();

        if (_sequences.Length > 0 && !TryActivateSequence(0, logFailure: false))
            TryActivateFallbackSequence(excludingSequenceIndex: null, fallbackReason: "Default animation sequence is invalid.");
    }

    public bool HasAnimation => _sequences.Length > 0;

    public IReadOnlyList<AnimationSequenceDescriptor> Sequences => _sequences;

    public int CurrentSequence => _sequenceIndex;

    public float CurrentFrame
    {
        get => _currentFrame;
        set => _currentFrame = ClampFrame(value, _sequenceIndex);
    }

    public bool IsPlaying { get; set; } = true;

    public float PlaybackSpeed { get; set; } = 1.0f;

    public bool Loop { get; set; } = true;

    public void SetSequence(int index)
    {
        if ((uint)index >= (uint)_sequences.Length)
            return;

        if (TryActivateSequence(index, logFailure: true))
            return;

        TryActivateFallbackSequence(excludingSequenceIndex: index, fallbackReason: $"Rejected sequence '{GetSequenceName(index)}'.");
    }

    public float StepToNextKeyframe()
    {
        if (!HasAnimation)
            return _currentFrame;

        float step = GetKeyframeStep(_sequenceIndex);
        _currentFrame = ClampFrame(_currentFrame + step, _sequenceIndex);
        return _currentFrame;
    }

    public float StepToPrevKeyframe()
    {
        if (!HasAnimation)
            return _currentFrame;

        float step = GetKeyframeStep(_sequenceIndex);
        _currentFrame = ClampFrame(_currentFrame - step, _sequenceIndex);
        return _currentFrame;
    }

    public AnimationTrackDebugStats GetTrackDebugStatsForCurrentSequence()
    {
        if (!HasAnimation)
            return new AnimationTrackDebugStats(0, 0, 0, 0, 0, 0, null, null);

        int duration = GetDurationMs(_sequenceIndex);
        return new AnimationTrackDebugStats(0, 0, 0, 0, 0, 0, 0, duration);
    }

    public void Update(float deltaMs)
    {
        if (!HasAnimation || !IsPlaying)
            return;

        int duration = GetDurationMs(_sequenceIndex);
        if (duration <= 0)
        {
            _currentFrame = 0.0f;
            return;
        }

        _currentFrame += Math.Clamp(deltaMs, 0.0f, 100.0f) * Math.Clamp(PlaybackSpeed, 0.0f, 10.0f);
        if (_currentFrame > duration)
        {
            if (Loop)
                _currentFrame %= duration;
            else
            {
                _currentFrame = duration;
                IsPlaying = false;
            }
        }
    }

    public int GetCurrentTimeMs()
        => (int)MathF.Round(ClampFrame(_currentFrame, _sequenceIndex));

    public bool TryPrepareCurrentSequence(
        out int sequenceIndex,
        out int timeMs,
        out M2ExternalAnimationRuntimeState? externalAnimationState)
    {
        sequenceIndex = _sequenceIndex;
        timeMs = 0;
        externalAnimationState = null;

        if (!HasAnimation)
            return false;

        if (!TryResolveSequenceState(_sequenceIndex, out externalAnimationState, out string? error))
        {
            LogSequenceFailure(_sequenceIndex, error ?? "Unknown animation validation failure.");
            if (!TryActivateFallbackSequence(
                    excludingSequenceIndex: _sequenceIndex,
                    fallbackReason: $"Rejected sequence '{GetSequenceName(_sequenceIndex)}'."))
            {
                return false;
            }

            if (!TryResolveSequenceState(_sequenceIndex, out externalAnimationState, out error))
            {
                LogSequenceFailure(_sequenceIndex, error ?? "Unknown fallback animation validation failure.");
                return false;
            }
        }

        sequenceIndex = _sequenceIndex;
        timeMs = GetCurrentTimeMs();
        return true;
    }

    public bool TryHandleRuntimeFailure(Exception ex)
    {
        ArgumentNullException.ThrowIfNull(ex);

        LogSequenceFailure(_sequenceIndex, ex.Message);
        return TryActivateFallbackSequence(
            excludingSequenceIndex: _sequenceIndex,
            fallbackReason: $"Runtime animation evaluation failed for '{GetSequenceName(_sequenceIndex)}'.");
    }

    public M2ExternalAnimationRuntimeState? ResolveExternalAnimationState()
    {
        if (!HasAnimation)
            return null;

        return TryResolveSequenceState(_sequenceIndex, out M2ExternalAnimationRuntimeState? state, out _)
            ? state
            : null;
    }

    private bool TryActivateSequence(int index, bool logFailure)
    {
        if ((uint)index >= (uint)_sequences.Length)
            return false;

        if (!TryResolveSequenceState(index, out _, out string? error))
        {
            if (logFailure)
                LogSequenceFailure(index, error ?? "Unknown animation validation failure.");

            return false;
        }

        _sequenceIndex = index;
        _currentFrame = 0.0f;
        return true;
    }

    private bool TryActivateFallbackSequence(int? excludingSequenceIndex, string fallbackReason)
    {
        for (int index = 0; index < _sequences.Length; index++)
        {
            if (excludingSequenceIndex.HasValue && index == excludingSequenceIndex.Value)
                continue;

            if (!TryActivateSequence(index, logFailure: false))
                continue;

            ViewerLog.Info(
                ViewerLog.Category.Mdx,
                $"[M2] {fallbackReason} Falling back to '{GetSequenceName(index)}' for '{Path.GetFileName(_model.Identity.CanonicalModelPath)}'.");
            return true;
        }

        return false;
    }

    private bool TryResolveSequenceState(int sequenceIndex, out M2ExternalAnimationRuntimeState? state, out string? error)
    {
        if (_externalStates.TryGetValue(sequenceIndex, out state))
        {
            error = null;
            return true;
        }

        if (_sequenceFailures.TryGetValue(sequenceIndex, out error))
        {
            state = null;
            return false;
        }

        if (!M2ExternalAnimationRuntime.TryChoose(_model, sequenceIndex, out M2ExternalAnimationRuntimeState? chosenState, out error)
            || chosenState == null)
        {
            state = null;
            RecordSequenceFailure(sequenceIndex, error ?? "Unknown animation sequence resolution failure.");
            return false;
        }

        if (!chosenState.UsesExternalFile || string.IsNullOrWhiteSpace(chosenState.CompanionPath))
        {
            _externalStates[sequenceIndex] = chosenState;
            state = chosenState;
            error = null;
            return true;
        }

        if (!TryReadAnimationBytes(chosenState.CompanionPath, out byte[]? animationBytes, out string resolvedPath)
            || animationBytes == null
            || animationBytes.Length == 0)
        {
            if (_loggedAnimationPaths.Add(chosenState.CompanionPath))
            {
                ViewerLog.Debug(
                    ViewerLog.Category.Mdx,
                    $"[M2] External animation companion missing for '{_model.Identity.CanonicalModelPath}': {chosenState.CompanionPath}");
            }

            _externalStates[sequenceIndex] = chosenState;
            state = chosenState;
            error = null;
            return true;
        }

        using MemoryStream animationStream = new(animationBytes, writable: false);
        if (!M2AnimationReader.TryRead(animationStream, resolvedPath, out M2ExternalAnimationDocument? animation, out error)
            || animation == null)
        {
            state = null;
            RecordSequenceFailure(sequenceIndex, error ?? "Malformed external animation companion.");
            return false;
        }

        try
        {
            M2ExternalAnimationRuntimeState loadedState = M2ExternalAnimationRuntime.Load(chosenState, animation);
            if (_loggedAnimationPaths.Add(resolvedPath))
            {
                ViewerLog.Info(
                    ViewerLog.Category.Mdx,
                    $"[M2] Loaded external animation companion for '{Path.GetFileName(_model.Identity.CanonicalModelPath)}': {resolvedPath}");
            }

            _externalStates[sequenceIndex] = loadedState;
            state = loadedState;
            error = null;
            return true;
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or InvalidDataException)
        {
            state = null;
            error = ex.Message;
            RecordSequenceFailure(sequenceIndex, error);
            return false;
        }
    }

    private void RecordSequenceFailure(int sequenceIndex, string reason)
    {
        _sequenceFailures[sequenceIndex] = reason;
        _externalStates.Remove(sequenceIndex);

        string currentName = _sequences[sequenceIndex].Name;
        if (!currentName.EndsWith(" [invalid]", StringComparison.Ordinal))
            _sequences[sequenceIndex] = _sequences[sequenceIndex] with { Name = $"{currentName} [invalid]" };
    }

    private void LogSequenceFailure(int sequenceIndex, string reason)
    {
        RecordSequenceFailure(sequenceIndex, reason);
        if (_loggedSequenceFailures.Add(sequenceIndex))
        {
            ViewerLog.Error(
                ViewerLog.Category.Mdx,
                $"[M2] Rejected animation sequence '{GetSequenceName(sequenceIndex)}' for '{Path.GetFileName(_model.Identity.CanonicalModelPath)}': {reason}");
        }
    }

    private string GetSequenceName(int sequenceIndex)
        => (uint)sequenceIndex < (uint)_sequences.Length ? _sequences[sequenceIndex].Name : $"Sequence {sequenceIndex}";

    private int GetDurationMs(int sequenceIndex)
    {
        if ((uint)sequenceIndex >= (uint)_model.Sequences.Count)
            return 0;

        return checked((int)Math.Max(_model.Sequences[sequenceIndex].Duration, 0u));
    }

    private float GetKeyframeStep(int sequenceIndex)
    {
        int duration = GetDurationMs(sequenceIndex);
        if (duration <= 0)
            return 1.0f;

        return Math.Max(duration / 30.0f, 1.0f);
    }

    private float ClampFrame(float frame, int sequenceIndex)
    {
        int duration = GetDurationMs(sequenceIndex);
        if (duration <= 0)
            return 0.0f;

        return Math.Clamp(frame, 0.0f, duration);
    }

    private bool TryReadAnimationBytes(string animationPath, out byte[]? bytes, out string resolvedPath)
    {
        bytes = null;
        resolvedPath = animationPath.Replace('/', '\\');

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            string? actualPath = mpqDataSource.FindInFileSet(animationPath)
                ?? mpqDataSource.FindInFileSet(animationPath.Replace('\\', '/'));
            if (!string.IsNullOrWhiteSpace(actualPath))
            {
                bytes = _dataSource.ReadFile(actualPath);
                if (bytes != null && bytes.Length > 0)
                {
                    resolvedPath = actualPath.Replace('/', '\\');
                    return true;
                }
            }
        }

        if (_dataSource != null)
        {
            bytes = _dataSource.ReadFile(animationPath)
                ?? _dataSource.ReadFile(animationPath.Replace('\\', '/'));
            if (bytes != null && bytes.Length > 0)
                return true;
        }

        if (File.Exists(animationPath))
        {
            bytes = File.ReadAllBytes(animationPath);
            resolvedPath = Path.GetFullPath(animationPath);
            return true;
        }

        return false;
    }
}
