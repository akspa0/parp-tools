namespace WowViewer.Core.Runtime.World.Validation;

public enum ValidationCaptureReadinessStatus
{
    Ready = 0,
    WaitingForSceneContent = 1,
    WaitingForFramebuffer = 2,
    WaitingForFramebufferResolution = 3,
    WaitingForWorldObjectLoads = 4,
    WaitingForTargetTile = 5,
    WaitingForSettledFrames = 6,
    TimedOut = 7,
}

public readonly record struct ValidationCaptureReadinessState(
    ValidationCaptureReadinessStatus Status,
    bool IsReady,
    bool TimedOut,
    int FramesObserved,
    int SettledFrames,
    string? Detail)
{
    public static ValidationCaptureReadinessState Ready(int framesObserved, int settledFrames)
        => new(
            ValidationCaptureReadinessStatus.Ready,
            IsReady: true,
            TimedOut: false,
            FramesObserved: framesObserved,
            SettledFrames: settledFrames,
            Detail: null);
}