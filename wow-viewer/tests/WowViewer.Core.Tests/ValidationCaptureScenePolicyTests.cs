using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Tests;

public sealed class ValidationCaptureScenePolicyTests
{
    [Fact]
    public void Constructor_ValidInputs_RetainsConfiguredValues()
    {
        ValidationCaptureArtifactPolicy artifactPolicy = new(
            ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette,
            ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff,
            ObjectsOnlyIntensityThreshold: 4,
            DiffMaskThreshold: 1,
            ObjectVisibilityMaskFileSuffix: "_object_visibility_mask.png",
            NoObjectMinimapFileSuffix: "_no_objects.png");

        ValidationCaptureScenePolicy policy = new(
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
            artifactPolicy: artifactPolicy);

        Assert.Equal(512, policy.RequestedResolution);
        Assert.Equal(48, policy.RequiredSettledFrames);
        Assert.Equal(2400, policy.MaxFramesBeforeCapture);
        Assert.Equal(25, policy.DetailedTileCountOverride);
        Assert.Equal(0.75f, policy.FogStartFactor);
        Assert.Equal(20000f, policy.FogEndDistance);
        Assert.Equal(1.0f, policy.ObjectStreamingRangeMultiplierFloor);
        Assert.Equal(24f, policy.MaxVisibleMdxBoundsHeight);
        Assert.True(policy.DisableObjectFog);
        Assert.True(policy.DisableObjectPathFilters);
        Assert.True(policy.HideWorldLiquids);
        Assert.True(policy.IgnoreTerrainHolesGlobally);
        Assert.True(policy.HideUiChrome);
        Assert.True(policy.EnableRuntimeWmoGroupLiquids);
        Assert.False(policy.EnableRuntimeWmoGroupVisibility);
        Assert.False(policy.IgnoreDistanceCulling);
        Assert.False(policy.IgnoreProjectedSizeCulling);
        Assert.False(policy.IgnoreVisionConeCulling);
        Assert.False(policy.IgnoreFrustumCulling);
        Assert.False(policy.IgnoreMaxViewDistanceCulling);
        Assert.Equal(artifactPolicy, policy.ArtifactPolicy);
    }

    [Fact]
    public void Constructor_FogStartFactorAboveOne_Throws()
    {
        ValidationCaptureArtifactPolicy artifactPolicy = new(
            ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette,
            ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff,
            ObjectsOnlyIntensityThreshold: 4,
            DiffMaskThreshold: 1,
            ObjectVisibilityMaskFileSuffix: "_object_visibility_mask.png",
            NoObjectMinimapFileSuffix: "_no_objects.png");

        Assert.Throws<ArgumentOutOfRangeException>(() => new ValidationCaptureScenePolicy(
            requestedResolution: 512,
            requiredSettledFrames: 48,
            maxFramesBeforeCapture: 2400,
            detailedTileCountOverride: 25,
            fogStartFactor: 1.25f,
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
            artifactPolicy: artifactPolicy));
    }
}
