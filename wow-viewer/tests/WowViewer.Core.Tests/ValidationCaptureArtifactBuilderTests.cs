using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class ValidationCaptureArtifactBuilderTests
{
    [Fact]
    public void ResolveMaskStrategy_EarlyBuild_UsesEarlyPolicy()
    {
        ValidationCaptureArtifactPolicy policy = CreatePolicy();

        ValidationObjectMaskStrategy strategy = ValidationCaptureArtifactBuilder.ResolveMaskStrategy("0.5.3.3368", policy);

        Assert.Equal(ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette, strategy);
    }

    [Fact]
    public void ResolveMaskStrategy_LaterBuild_UsesLaterPolicy()
    {
        ValidationCaptureArtifactPolicy policy = CreatePolicy();

        ValidationObjectMaskStrategy strategy = ValidationCaptureArtifactBuilder.ResolveMaskStrategy("3.3.5.12340", policy);

        Assert.Equal(ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff, strategy);
    }

    [Fact]
    public void Build_EarlyBuildWithObjectsOnly_UsesDirectSilhouetteMask()
    {
        ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
            new ValidationCaptureArtifactInputs(
                tileName: "Azeroth_30_48",
                buildLabel: "0.5.3.3368",
                width: 2,
                height: 1,
                primaryRgbaPixels: [0, 0, 0, 255, 0, 0, 0, 255],
                noObjectsRgbaPixels: [0, 0, 0, 255, 0, 0, 0, 255],
                objectsOnlyRgbaPixels: [5, 0, 0, 255, 4, 0, 0, 255]),
            CreatePolicy());

        Assert.Equal(ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette, outputs.MaskStrategy);
        Assert.Equal([255, 0], outputs.ObjectVisibilityMaskL8Pixels);
        Assert.Equal(8, outputs.NoObjectMinimapRgbaPixels.Length);
        Assert.Equal(64, outputs.ObjectVisibilityMaskHash.Length);
    }

    [Fact]
    public void Build_LaterBuild_UsesPrimaryVsNoObjectsDiffMask()
    {
        ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
            new ValidationCaptureArtifactInputs(
                tileName: "Azeroth_30_48",
                buildLabel: "3.3.5.12340",
                width: 2,
                height: 1,
                primaryRgbaPixels: [10, 0, 0, 255, 10, 0, 0, 255],
                noObjectsRgbaPixels: [8, 0, 0, 255, 5, 0, 0, 255],
                objectsOnlyRgbaPixels: null),
            CreatePolicy());

        Assert.Equal(ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff, outputs.MaskStrategy);
        Assert.Equal([0, 255], outputs.ObjectVisibilityMaskL8Pixels);
        Assert.Equal([8, 0, 0, 255, 5, 0, 0, 255], outputs.NoObjectMinimapRgbaPixels);
    }

    [Fact]
    public void Build_EarlyBuildWithoutObjectsOnly_FallsBackToDiffMask()
    {
        ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
            new ValidationCaptureArtifactInputs(
                tileName: "Azeroth_30_48",
                buildLabel: "0.5.3.3368",
                width: 1,
                height: 1,
                primaryRgbaPixels: [9, 0, 0, 255],
                noObjectsRgbaPixels: [0, 0, 0, 255],
                objectsOnlyRgbaPixels: null),
            CreatePolicy());

        Assert.Equal(ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff, outputs.MaskStrategy);
        Assert.Equal([255], outputs.ObjectVisibilityMaskL8Pixels);
    }

    [Fact]
    public void Build_LaterBuild_WithBlankObjectsOnlyAndValidDiff_PrefersDiffMask()
    {
        ValidationCaptureArtifactOutputs outputs = ValidationCaptureArtifactBuilder.Build(
            new ValidationCaptureArtifactInputs(
                tileName: "Azeroth_30_48",
                buildLabel: "3.3.5.12340",
                width: 2,
                height: 1,
                primaryRgbaPixels: [0, 0, 0, 255, 20, 0, 0, 255],
                noObjectsRgbaPixels: [0, 0, 0, 255, 0, 0, 0, 255],
                objectsOnlyRgbaPixels: [0, 0, 0, 255, 0, 0, 0, 255]),
            CreatePolicy());

        Assert.Equal(ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff, outputs.MaskStrategy);
        Assert.Equal([0, 255], outputs.ObjectVisibilityMaskL8Pixels);
    }

    [Fact]
    public void Build_SameInputs_ProducesDeterministicHashes()
    {
        ValidationCaptureArtifactInputs inputs = new(
            tileName: "Azeroth_30_48",
            buildLabel: "3.3.5.12340",
            width: 2,
            height: 1,
            primaryRgbaPixels: [10, 0, 0, 255, 10, 0, 0, 255],
            noObjectsRgbaPixels: [8, 0, 0, 255, 5, 0, 0, 255],
            objectsOnlyRgbaPixels: null);

        ValidationCaptureArtifactPolicy policy = CreatePolicy();
        ValidationCaptureArtifactOutputs first = ValidationCaptureArtifactBuilder.Build(inputs, policy);
        ValidationCaptureArtifactOutputs second = ValidationCaptureArtifactBuilder.Build(inputs, policy);

        Assert.Equal(first.ObjectVisibilityMaskHash, second.ObjectVisibilityMaskHash);
        Assert.Equal(first.NoObjectMinimapHash, second.NoObjectMinimapHash);
    }

    private static ValidationCaptureArtifactPolicy CreatePolicy()
    {
        return new ValidationCaptureArtifactPolicy(
            ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette,
            ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff,
            ObjectsOnlyIntensityThreshold: 4,
            DiffMaskThreshold: 4,
            ObjectVisibilityMaskFileSuffix: "_object_visibility_mask.png",
            NoObjectMinimapFileSuffix: "_no_objects.png");
    }
}
