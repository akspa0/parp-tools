namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationWorldSceneSnapshot(
    bool HasSceneContent,
    int FramebufferWidth,
    int FramebufferHeight,
    bool TargetTileLoaded,
    bool TerrainStreaming,
    int PendingWorldObjectLoadCount);