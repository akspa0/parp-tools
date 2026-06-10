using WowViewer.Core.Runtime.World.Validation;
using WowViewer.Tools.ValidationCapture;

namespace WowViewer.Core.Tests;

public sealed class HeadlessValidationCaptureRunnerTests
{
    [Fact]
    public void Run_WhenAdapterThrows_RecordsFailedVariantInsteadOfCrashingBatch()
    {
        using TemporaryDirectory temp = new();
        ValidationCaptureTileRequest request = new(
            "Azeroth_30_48",
            30,
            48,
            ValidationCaptureVariant.ObjectsOnly,
            Path.Combine(temp.RootPath, "objectsonly", "Azeroth_30_48_viewer_validation.png"));
        ValidationCaptureBatchPlan batchPlan = new(
            temp.RootPath,
            "Azeroth",
            Path.Combine(temp.RootPath, "primary"),
            Path.Combine(temp.RootPath, "noliquids"),
            Path.Combine(temp.RootPath, "noobjects"),
            Path.Combine(temp.RootPath, "objectsonly"),
            requestedResolution: 512,
            buildLabel: "3.3.5.12340",
            [request]);
        ValidationCaptureScenePolicy scenePolicy = new(
            requestedResolution: 512,
            requiredSettledFrames: 1,
            maxFramesBeforeCapture: 4,
            detailedTileCountOverride: 1,
            fogStartFactor: 0.75f,
            fogEndDistance: 20000f,
            objectStreamingRangeMultiplierFloor: 1.0f,
            maxVisibleMdxBoundsHeight: 24f,
            disableObjectFog: true,
            disableObjectPathFilters: true,
            hideWorldLiquids: true,
            ignoreTerrainHolesGlobally: true,
            hideUiChrome: true,
            enableRuntimeWmoGroupLiquids: true,
            enableRuntimeWmoGroupVisibility: false,
            artifactPolicy: new ValidationCaptureArtifactPolicy(
                ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette,
                ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff,
                ObjectsOnlyIntensityThreshold: 4,
                DiffMaskThreshold: 8,
                ObjectVisibilityMaskFileSuffix: "_object_visibility_mask.png",
                NoObjectMinimapFileSuffix: "_no_objects.png"));
        IReadOnlyDictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> variantPolicies =
            Enum.GetValues<ValidationCaptureVariant>().ToDictionary(
                static variant => variant,
                static _ => new ValidationCaptureVariantPolicy(
                    ShowTerrain: true,
                    ShowTerrainLiquids: true,
                    ShowObjects: true,
                    ShowWmos: true,
                    ShowDoodads: true,
                    ShowSky: true,
                    ShowWdl: true,
                    ShowWorldLiquids: false));
        HeadlessValidationCaptureSession session = new(
            clientRoot: temp.RootPath,
            mapInput: "World\\Maps\\Azeroth\\Azeroth.wdt",
            buildLabel: "3.3.5.12340",
            looseOverlayRoot: null,
            batchPlan,
            scenePolicy,
            variantPolicies);

        ValidationCaptureBatchResult result = HeadlessValidationCaptureRunner.Run(session, new ThrowingSceneAdapter());

        ValidationCaptureVariantResult variant = Assert.Single(result.VariantResults);
        Assert.False(variant.Succeeded);
        Assert.False(variant.TimedOut);
        Assert.Contains("InvalidDataException", variant.FailureReason, StringComparison.Ordinal);
        Assert.Equal(1, result.FailedVariantCount);
    }

    private sealed class ThrowingSceneAdapter : IValidationWorldSceneAdapter
    {
        public void Initialize(HeadlessValidationCaptureSession session)
        {
        }

        public void ApplyScenePolicy(ValidationCaptureScenePolicy scenePolicy)
        {
        }

        public void ApplyVariantPolicy(ValidationCaptureVariantPolicy variantPolicy)
        {
        }

        public ValidationWorldSceneSnapshot CaptureSnapshot(ValidationCaptureTileRequest request, int framesObserved, int settledFrames)
        {
            throw new InvalidDataException("synthetic reader failure");
        }

        public float ResolveGroundHeight(int tileX, int tileY)
        {
            return 0f;
        }

        public void RenderFrame(ValidationCaptureCameraFrame cameraFrame)
        {
        }

        public byte[] ReadFramebufferRgba()
        {
            return [];
        }

        public void Dispose()
        {
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            RootPath = Path.Combine(Path.GetTempPath(), "WowViewer.HeadlessValidationCaptureRunnerTests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(RootPath);
        }

        public string RootPath { get; }

        public void Dispose()
        {
            if (Directory.Exists(RootPath))
                Directory.Delete(RootPath, recursive: true);
        }
    }
}
