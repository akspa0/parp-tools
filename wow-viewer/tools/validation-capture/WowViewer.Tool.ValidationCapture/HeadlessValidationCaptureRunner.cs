using System.Numerics;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Tools.ValidationCapture;

internal static class HeadlessValidationCaptureRunner
{
    private const float DefaultTileWorldSize = 533.33333f;
    private const float DefaultMapOrigin = 32f * DefaultTileWorldSize;
    private const float DefaultEyeHeightOffset = 2048f;
    private const float DefaultNearPlane = 0.1f;
    private const float DefaultFarPlane = 20000f;

    public static ValidationCaptureBatchResult Run(
        HeadlessValidationCaptureSession session,
        IValidationWorldSceneAdapter sceneAdapter)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(sceneAdapter);

        List<ValidationCaptureVariantResult> results = new(session.BatchPlan.RequestCount);
        sceneAdapter.Initialize(session);
        sceneAdapter.ApplyScenePolicy(session.ScenePolicy);

        bool batchHasSettled = false;

        foreach (ValidationCaptureTileRequest request in session.BatchPlan.TileRequests)
        {
            sceneAdapter.ApplyVariantPolicy(session.VariantPolicies[request.Variant]);
            ValidationCaptureVariantResult result = ExecuteRequest(
                session, request, sceneAdapter, batchHasSettled);
            results.Add(result);

            if (result.Succeeded && !result.TimedOut)
                batchHasSettled = true;
        }

        return new ValidationCaptureBatchResult(
            session.BatchPlan.MapName,
            session.BuildLabel,
            session.ScenePolicy.RequestedResolution,
            results);
    }

    private static ValidationCaptureVariantResult ExecuteRequest(
        HeadlessValidationCaptureSession session,
        ValidationCaptureTileRequest request,
        IValidationWorldSceneAdapter sceneAdapter,
        bool batchHasSettled)
    {
        int framesObserved = 0;
        int settledFrames = 0;

        try
        {
            while (true)
            {
                framesObserved++;
                ValidationWorldSceneSnapshot sceneSnapshot = sceneAdapter.CaptureSnapshot(request, framesObserved, settledFrames);
                settledFrames = CanAccumulateSettledFrames(sceneSnapshot, session.ScenePolicy)
                    ? settledFrames + 1
                    : 0;

                ValidationCaptureReadinessState readinessState = ValidationCaptureReadinessEvaluator.Evaluate(new ValidationCaptureReadinessSnapshot(
                    HasSceneContent: sceneSnapshot.HasSceneContent,
                    HasFramebuffer: sceneSnapshot.FramebufferWidth > 0 && sceneSnapshot.FramebufferHeight > 0,
                    FramebufferWidth: sceneSnapshot.FramebufferWidth,
                    FramebufferHeight: sceneSnapshot.FramebufferHeight,
                    RequestedResolution: session.ScenePolicy.RequestedResolution,
                    WaitForSceneReady: true,
                    HasTargetTile: true,
                    TargetTileLoaded: sceneSnapshot.TargetTileLoaded,
                    TerrainStreaming: sceneSnapshot.TerrainStreaming,
                    TrackPendingWorldObjectLoads: true,
                    PendingWorldObjectLoadCount: sceneSnapshot.PendingWorldObjectLoadCount,
                    FramesObserved: framesObserved,
                    SettledFrames: settledFrames,
                    RequiredSettledFrames: session.ScenePolicy.RequiredSettledFrames,
                    MaxFramesBeforeCapture: session.ScenePolicy.MaxFramesBeforeCapture,
                    BatchHasSettled: batchHasSettled,
                    BatchSettledFrames: session.ScenePolicy.BatchSettledFrames,
                    FastSettleAfterBatchReady: session.ScenePolicy.FastSettleAfterBatchReady));

                if (readinessState.IsReady)
                {
                    float aspectRatio = sceneSnapshot.FramebufferHeight > 0
                        ? sceneSnapshot.FramebufferWidth / (float)sceneSnapshot.FramebufferHeight
                        : 1f;
                    float groundHeight = sceneAdapter.ResolveGroundHeight(request.TileX, request.TileY);
                    ValidationCaptureCameraFrame cameraFrame = ValidationCaptureCameraSolver.SolveTopDown(new ValidationCaptureCameraInput(
                        request.TileX,
                        request.TileY,
                        aspectRatio,
                        groundHeight,
                        DefaultMapOrigin,
                        DefaultTileWorldSize,
                        DefaultTileWorldSize,
                        DefaultEyeHeightOffset,
                        DefaultNearPlane,
                        DefaultFarPlane,
                        Vector3.UnitX));

                    sceneAdapter.RenderFrame(cameraFrame);
                    byte[] rgbaPixels = sceneAdapter.ReadFramebufferRgba();
                    HeadlessValidationFramebufferExporter.WriteImage(
                        request.OutputPath,
                        sceneSnapshot.FramebufferWidth,
                        sceneSnapshot.FramebufferHeight,
                        rgbaPixels,
                        sourceOriginBottomLeft: true);
                    if (request.Variant == ValidationCaptureVariant.TerrainShade)
                    {
                        TerrainShadeCaptureManifestWriter.Write(session, request, cameraFrame);
                        if (sceneAdapter is ValidationWorldSceneAdapter concreteAdapter
                            && concreteAdapter.LastRuntimeFrame is { } runtimeFrame)
                        {
                            TerrainShadeSignalAuditor.Write(
                                request.OutputPath,
                                runtimeFrame,
                                cameraFrame,
                                rgbaPixels,
                                sceneSnapshot.FramebufferWidth,
                                sceneSnapshot.FramebufferHeight,
                                sourceOriginBottomLeft: true);
                        }
                    }

                    return new ValidationCaptureVariantResult(
                        request.Variant,
                        request.TileName,
                        request.TileX,
                        request.TileY,
                        request.OutputPath,
                        readinessState,
                        succeeded: true,
                        timedOut: false,
                        framesObserved,
                        settledFrames,
                        failureReason: null);
                }

                if (readinessState.TimedOut)
                {
                    return new ValidationCaptureVariantResult(
                        request.Variant,
                        request.TileName,
                        request.TileX,
                        request.TileY,
                        request.OutputPath,
                        readinessState,
                        succeeded: false,
                        timedOut: true,
                        framesObserved,
                        settledFrames,
                        failureReason: readinessState.Detail);
                }
            }
        }
        catch (Exception ex)
        {
            return new ValidationCaptureVariantResult(
                request.Variant,
                request.TileName,
                request.TileX,
                request.TileY,
                request.OutputPath,
                new ValidationCaptureReadinessState(
                    ValidationCaptureReadinessStatus.WaitingForSceneContent,
                    IsReady: false,
                    TimedOut: false,
                    FramesObserved: framesObserved,
                    SettledFrames: settledFrames,
                    Detail: ex.Message),
                succeeded: false,
                timedOut: false,
                framesObserved,
                settledFrames,
                failureReason: $"{ex.GetType().Name}: {ex.Message}");
        }
    }

    private static bool CanAccumulateSettledFrames(
        ValidationWorldSceneSnapshot sceneSnapshot,
        ValidationCaptureScenePolicy scenePolicy)
    {
        return sceneSnapshot.HasSceneContent
            && sceneSnapshot.FramebufferWidth >= scenePolicy.RequestedResolution
            && sceneSnapshot.FramebufferHeight >= scenePolicy.RequestedResolution
            && sceneSnapshot.TargetTileLoaded
            && !sceneSnapshot.TerrainStreaming
            && sceneSnapshot.PendingWorldObjectLoadCount <= 0;
    }
}
