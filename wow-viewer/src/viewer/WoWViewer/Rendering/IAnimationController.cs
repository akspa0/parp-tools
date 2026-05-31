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

    void SetSequence(int index);

    float StepToNextKeyframe();

    float StepToPrevKeyframe();

    AnimationTrackDebugStats GetTrackDebugStatsForCurrentSequence();
}
