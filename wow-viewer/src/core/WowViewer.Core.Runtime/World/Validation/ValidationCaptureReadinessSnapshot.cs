namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureReadinessSnapshot(
    bool HasSceneContent,
    bool HasFramebuffer,
    int FramebufferWidth,
    int FramebufferHeight,
    int RequestedResolution,
    bool WaitForSceneReady,
    bool HasTargetTile,
    bool TargetTileLoaded,
    bool TerrainStreaming,
    bool TrackPendingWorldObjectLoads,
    int PendingWorldObjectLoadCount,
    int FramesObserved,
    int SettledFrames,
    int RequiredSettledFrames,
    int MaxFramesBeforeCapture);