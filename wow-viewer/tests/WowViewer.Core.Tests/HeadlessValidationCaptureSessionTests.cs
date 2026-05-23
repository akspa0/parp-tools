using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class HeadlessValidationCaptureSessionTests
{
    [Fact]
    public void Constructor_MissingVariantPolicy_Throws()
    {
        ValidationCaptureBatchPlan batchPlan = new(
            datasetRoot: "dataset",
            mapName: "Azeroth",
            primaryOutputDirectory: "primary",
            noLiquidsOutputDirectory: "noliquids",
            noObjectsOutputDirectory: "noobjects",
            objectsOnlyOutputDirectory: "objectsonly",
            requestedResolution: 512,
            buildLabel: "0.5.3.3368",
            tileRequests:
            [
                new ValidationCaptureTileRequest("Azeroth_30_48", 30, 48, ValidationCaptureVariant.Primary, "primary.png")
            ]);

        ValidationCaptureScenePolicy scenePolicy = new(
            requestedResolution: 512,
            requiredSettledFrames: 48,
            maxFramesBeforeCapture: 2400,
            detailedTileCountOverride: 25,
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
                4,
                8,
                "_object_visibility_mask.png",
                "_no_objects.png"));

        Dictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> variantPolicies = new()
        {
            [ValidationCaptureVariant.Primary] = new(true, true, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoLiquids] = new(true, false, true, true, true, true, true, false),
            [ValidationCaptureVariant.NoObjects] = new(true, true, false, false, false, true, true, false),
        };

        Assert.Throws<ArgumentException>(() => new HeadlessValidationCaptureSession(
            clientRoot: "client",
            mapInput: "World\\Maps\\Azeroth\\Azeroth.wdt",
            buildLabel: "0.5.3.3368",
            looseOverlayRoot: null,
            batchPlan,
            scenePolicy,
            variantPolicies));
    }
}