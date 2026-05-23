namespace WowViewer.Core.Runtime.World.Validation;

public interface IValidationWorldSceneAdapter : IDisposable
{
    void Initialize(HeadlessValidationCaptureSession session);

    void ApplyScenePolicy(ValidationCaptureScenePolicy scenePolicy);

    void ApplyVariantPolicy(ValidationCaptureVariantPolicy variantPolicy);

    ValidationWorldSceneSnapshot CaptureSnapshot(
        ValidationCaptureTileRequest request,
        int framesObserved,
        int settledFrames);

    float ResolveGroundHeight(int tileX, int tileY);

    void RenderFrame(ValidationCaptureCameraFrame cameraFrame);

    byte[] ReadFramebufferRgba();
}