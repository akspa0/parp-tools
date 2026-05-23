namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureReadinessEvaluator
{
    public static ValidationCaptureReadinessState Evaluate(
        ValidationCaptureReadinessSnapshot snapshot)
    {
        if (!snapshot.WaitForSceneReady)
            return ValidationCaptureReadinessState.Ready(snapshot.FramesObserved, snapshot.SettledFrames);

        if (!snapshot.HasSceneContent)
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForSceneContent,
                "scene content not ready");

        if (!snapshot.HasFramebuffer)
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForFramebuffer,
                "framebuffer not ready");

        if (snapshot.FramebufferWidth < snapshot.RequestedResolution
            || snapshot.FramebufferHeight < snapshot.RequestedResolution)
        {
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForFramebufferResolution,
                $"framebuffer {snapshot.FramebufferWidth}x{snapshot.FramebufferHeight} below requested {snapshot.RequestedResolution}");
        }

        if (snapshot.TrackPendingWorldObjectLoads && snapshot.PendingWorldObjectLoadCount > 0)
        {
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForWorldObjectLoads,
                $"pending world object loads: {snapshot.PendingWorldObjectLoadCount}");
        }

        if (snapshot.HasTargetTile && (!snapshot.TargetTileLoaded || snapshot.TerrainStreaming))
        {
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForTargetTile,
                $"target tile loaded={snapshot.TargetTileLoaded} terrain streaming={snapshot.TerrainStreaming}");
        }

        if (snapshot.SettledFrames < snapshot.RequiredSettledFrames)
        {
            return WaitOrTimeout(
                snapshot,
                ValidationCaptureReadinessStatus.WaitingForSettledFrames,
                $"settled frames {snapshot.SettledFrames}/{snapshot.RequiredSettledFrames}");
        }

        return ValidationCaptureReadinessState.Ready(snapshot.FramesObserved, snapshot.SettledFrames);
    }

    private static ValidationCaptureReadinessState WaitOrTimeout(
        ValidationCaptureReadinessSnapshot snapshot,
        ValidationCaptureReadinessStatus waitingStatus,
        string detail)
    {
        if (snapshot.FramesObserved >= snapshot.MaxFramesBeforeCapture)
        {
            return new ValidationCaptureReadinessState(
                ValidationCaptureReadinessStatus.TimedOut,
                IsReady: false,
                TimedOut: true,
                FramesObserved: snapshot.FramesObserved,
                SettledFrames: snapshot.SettledFrames,
                Detail: detail);
        }

        return new ValidationCaptureReadinessState(
            waitingStatus,
            IsReady: false,
            TimedOut: false,
            FramesObserved: snapshot.FramesObserved,
            SettledFrames: snapshot.SettledFrames,
            Detail: detail);
    }
}