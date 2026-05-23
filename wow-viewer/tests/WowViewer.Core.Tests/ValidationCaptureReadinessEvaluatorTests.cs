using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class ValidationCaptureReadinessEvaluatorTests
{
    [Fact]
    public void Evaluate_WaitDisabled_ReturnsReady()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: false,
            HasFramebuffer: false,
            FramebufferWidth: 0,
            FramebufferHeight: 0,
            RequestedResolution: 512,
            WaitForSceneReady: false,
            HasTargetTile: false,
            TargetTileLoaded: false,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: false,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 0,
            SettledFrames: 0,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.True(state.IsReady);
        Assert.Equal(ValidationCaptureReadinessStatus.Ready, state.Status);
    }

    [Fact]
    public void Evaluate_SmallFramebuffer_ReturnsResolutionWait()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: true,
            HasFramebuffer: true,
            FramebufferWidth: 256,
            FramebufferHeight: 512,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: false,
            TargetTileLoaded: false,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: false,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 10,
            SettledFrames: 0,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.False(state.IsReady);
        Assert.False(state.TimedOut);
        Assert.Equal(ValidationCaptureReadinessStatus.WaitingForFramebufferResolution, state.Status);
    }

    [Fact]
    public void Evaluate_PendingWorldObjectLoads_ReturnsLoadWait()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: true,
            HasFramebuffer: true,
            FramebufferWidth: 512,
            FramebufferHeight: 512,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: true,
            TargetTileLoaded: true,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: true,
            PendingWorldObjectLoadCount: 3,
            FramesObserved: 22,
            SettledFrames: 0,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.Equal(ValidationCaptureReadinessStatus.WaitingForWorldObjectLoads, state.Status);
        Assert.Contains("pending world object loads", state.Detail);
    }

    [Fact]
    public void Evaluate_TargetTileStillStreaming_ReturnsTileWait()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: true,
            HasFramebuffer: true,
            FramebufferWidth: 512,
            FramebufferHeight: 512,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: true,
            TargetTileLoaded: true,
            TerrainStreaming: true,
            TrackPendingWorldObjectLoads: false,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 22,
            SettledFrames: 0,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.Equal(ValidationCaptureReadinessStatus.WaitingForTargetTile, state.Status);
    }

    [Fact]
    public void Evaluate_SettledFramesIncomplete_ReturnsSettledWait()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: true,
            HasFramebuffer: true,
            FramebufferWidth: 512,
            FramebufferHeight: 512,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: true,
            TargetTileLoaded: true,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: true,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 100,
            SettledFrames: 47,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.Equal(ValidationCaptureReadinessStatus.WaitingForSettledFrames, state.Status);
    }

    [Fact]
    public void Evaluate_NotReadyAtTimeout_ReturnsTimedOut()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: false,
            HasFramebuffer: false,
            FramebufferWidth: 0,
            FramebufferHeight: 0,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: false,
            TargetTileLoaded: false,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: false,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 2400,
            SettledFrames: 0,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.Equal(ValidationCaptureReadinessStatus.TimedOut, state.Status);
        Assert.True(state.TimedOut);
        Assert.Equal("scene content not ready", state.Detail);
    }

    [Fact]
    public void Evaluate_AllConditionsSatisfied_ReturnsReady()
    {
        ValidationCaptureReadinessState state = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
            HasSceneContent: true,
            HasFramebuffer: true,
            FramebufferWidth: 512,
            FramebufferHeight: 512,
            RequestedResolution: 512,
            WaitForSceneReady: true,
            HasTargetTile: true,
            TargetTileLoaded: true,
            TerrainStreaming: false,
            TrackPendingWorldObjectLoads: true,
            PendingWorldObjectLoadCount: 0,
            FramesObserved: 100,
            SettledFrames: 48,
            RequiredSettledFrames: 48,
            MaxFramesBeforeCapture: 2400));

        Assert.Equal(ValidationCaptureReadinessStatus.Ready, state.Status);
        Assert.True(state.IsReady);
    }
}