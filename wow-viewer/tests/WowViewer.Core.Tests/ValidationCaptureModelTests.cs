using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class ValidationCaptureModelTests
{
    [Fact]
    public void TileRequest_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ValidationCaptureTileRequest(
            tileName: "Azeroth_30_48",
            tileX: 30,
            tileY: 48,
            variant: (ValidationCaptureVariant)99,
            outputPath: "out.png"));
    }

    [Fact]
    public void BatchPlan_NullTileEntry_Throws()
    {
        IReadOnlyList<ValidationCaptureTileRequest> requests = new ValidationCaptureTileRequest[]
        {
            new("Azeroth_30_48", 30, 48, ValidationCaptureVariant.Primary, "primary.png"),
            null!,
        };

        Assert.Throws<ArgumentException>(() => new ValidationCaptureBatchPlan(
            datasetRoot: "dataset",
            mapName: "Azeroth",
            primaryOutputDirectory: "primary",
            noLiquidsOutputDirectory: "noliquids",
            noObjectsOutputDirectory: "noobjects",
            objectsOnlyOutputDirectory: "objectsonly",
            requestedResolution: 512,
            buildLabel: "0.5.3.3368",
            tileRequests: requests));
    }

    [Fact]
    public void VariantResult_InvalidVariant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ValidationCaptureVariantResult(
            variant: (ValidationCaptureVariant)(-1),
            tileName: "Azeroth_30_48",
            tileX: 30,
            tileY: 48,
            outputPath: "primary.png",
            readinessState: ValidationCaptureReadinessState.Ready(50, 48),
            succeeded: true,
            timedOut: false,
            framesObserved: 50,
            settledFrames: 48,
            failureReason: null));
    }

    [Fact]
    public void BatchResult_NullVariantEntry_Throws()
    {
        IReadOnlyList<ValidationCaptureVariantResult> results = new ValidationCaptureVariantResult[]
        {
            new(
                ValidationCaptureVariant.Primary,
                "Azeroth_30_48",
                30,
                48,
                "primary.png",
                ValidationCaptureReadinessState.Ready(50, 48),
                succeeded: true,
                timedOut: false,
                framesObserved: 50,
                settledFrames: 48,
                failureReason: null),
            null!,
        };

        Assert.Throws<ArgumentException>(() => new ValidationCaptureBatchResult(
            mapName: "Azeroth",
            buildLabel: "3.3.5.12340",
            requestedResolution: 512,
            variantResults: results));
    }

    [Fact]
    public void BatchResult_CountsReflectOutcomeFlags()
    {
        ValidationCaptureVariantResult[] results =
        [
            new(
                ValidationCaptureVariant.Primary,
                "Azeroth_30_48",
                30,
                48,
                "primary.png",
                ValidationCaptureReadinessState.Ready(50, 48),
                succeeded: true,
                timedOut: false,
                framesObserved: 50,
                settledFrames: 48,
                failureReason: null),
            new(
                ValidationCaptureVariant.NoObjects,
                "Azeroth_30_48",
                30,
                48,
                "noobjects.png",
                new ValidationCaptureReadinessState(
                    ValidationCaptureReadinessStatus.TimedOut,
                    IsReady: false,
                    TimedOut: true,
                    FramesObserved: 2400,
                    SettledFrames: 0,
                    Detail: "timeout"),
                succeeded: false,
                timedOut: true,
                framesObserved: 2400,
                settledFrames: 0,
                failureReason: "timeout")
        ];

        ValidationCaptureBatchResult batch = new(
            mapName: "Azeroth",
            buildLabel: "0.5.3.3368",
            requestedResolution: 512,
            variantResults: results);

        Assert.Equal(2, batch.TotalVariantCount);
        Assert.Equal(1, batch.SucceededVariantCount);
        Assert.Equal(1, batch.TimedOutVariantCount);
        Assert.Equal(1, batch.FailedVariantCount);
    }
}