namespace WoWViewer.Rendering;

public readonly record struct AnimationTimeRange(float Start, float End);

public readonly record struct AnimationSequenceDescriptor(int Index, string Name, AnimationTimeRange Time);

public readonly record struct AnimationTrackDebugStats(
    int TranslationKeysTotal,
    int RotationKeysTotal,
    int ScalingKeysTotal,
    int TranslationKeysInSequence,
    int RotationKeysInSequence,
    int ScalingKeysInSequence,
    int? MinKeyTime,
    int? MaxKeyTime);

public interface IAnimationController
{
    bool HasAnimation { get; }

    IReadOnlyList<AnimationSequenceDescriptor> Sequences { get; }

    int CurrentSequence { get; }

    float CurrentFrame { get; set; }

    bool IsPlaying { get; set; }

    /// <summary>Playback speed multiplier. 1.0 = normal, 0.5 = half speed, 2.0 = double speed.</summary>
    float PlaybackSpeed { get; set; }

    /// <summary>When true, the current sequence loops. When false, playback stops at the end.</summary>
    bool Loop { get; set; }

    void SetSequence(int index);

    float StepToNextKeyframe();

    float StepToPrevKeyframe();

    AnimationTrackDebugStats GetTrackDebugStatsForCurrentSequence();
}
