using WowViewer.Core.Runtime.World.Validation;
using WowViewer.Tools.ValidationCapture;

namespace WowViewer.Core.Tests;

public sealed class ValidationWorldScenePolicyApplierTests
{
    [Fact]
    public void CreateState_PrimaryVariant_AppliesDeterministicBatchOverrides()
    {
        ValidationWorldScenePolicyState state = ValidationWorldScenePolicyApplier.CreateState(
            CreateScenePolicy(),
            new ValidationCaptureVariantPolicy(
                ShowTerrain: true,
                ShowTerrainLiquids: true,
                ShowObjects: true,
                ShowWmos: true,
                ShowDoodads: true,
                ShowSky: true,
                ShowWdl: true,
                ShowWorldLiquids: true));

        Assert.Equal(512, state.RequestedResolution);
        Assert.True(state.HideUiChrome);
        Assert.Equal(25, state.DetailedTileCountOverride);
        Assert.Equal(0.75f, state.FogStartFactor);
        Assert.Equal(20000f, state.FogEndDistance);
        Assert.False(state.ObjectFogEnabled);
        Assert.True(state.ShowTerrain);
        Assert.True(state.ShowTerrainLiquids);
        Assert.False(state.ShowWorldLiquids);
        Assert.True(state.ShowObjects);
        Assert.True(state.ShowWmos);
        Assert.True(state.ShowDoodads);
        Assert.True(state.ShowSky);
        Assert.True(state.ShowWdl);
        Assert.True(state.IgnoreTerrainHolesGlobally);
        Assert.False(state.ObjectPathFiltersEnabled);
        Assert.Equal(1.0f, state.ObjectStreamingRangeMultiplier);
        Assert.Equal(24f, state.MaxVisibleMdxBoundsHeight);
        Assert.False(state.EnableRuntimeWmoGroupVisibility);
        Assert.True(state.EnableRuntimeWmoGroupLiquids);
        Assert.False(state.IgnoreDistanceCulling);
        Assert.False(state.IgnoreProjectedSizeCulling);
        Assert.False(state.IgnoreVisionConeCulling);
        Assert.False(state.IgnoreFrustumCulling);
        Assert.False(state.IgnoreMaxViewDistanceCulling);
    }

    [Fact]
    public void CreateState_ObjectsOnlyVariant_HidesTerrainAndSuppressesGroupLiquids()
    {
        ValidationWorldScenePolicyState state = ValidationWorldScenePolicyApplier.CreateState(
            CreateScenePolicy(),
            new ValidationCaptureVariantPolicy(
                ShowTerrain: false,
                ShowTerrainLiquids: false,
                ShowObjects: true,
                ShowWmos: true,
                ShowDoodads: true,
                ShowSky: false,
                ShowWdl: false,
                ShowWorldLiquids: false));

        Assert.False(state.ShowTerrain);
        Assert.False(state.ShowTerrainLiquids);
        Assert.True(state.ShowObjects);
        Assert.True(state.ShowWmos);
        Assert.True(state.ShowDoodads);
        Assert.False(state.ShowSky);
        Assert.False(state.ShowWdl);
        Assert.False(state.ShowWorldLiquids);
        Assert.False(state.EnableRuntimeWmoGroupLiquids);
        Assert.False(state.EnableRuntimeWmoGroupVisibility);
        Assert.False(state.IgnoreDistanceCulling);
        Assert.False(state.IgnoreProjectedSizeCulling);
        Assert.False(state.IgnoreVisionConeCulling);
        Assert.False(state.IgnoreFrustumCulling);
        Assert.False(state.IgnoreMaxViewDistanceCulling);
    }

    [Fact]
    public void ApplyVariantPolicy_ObjectVisibilityOff_SuppressesWmosAndDoodads()
    {
        ValidationWorldScenePolicyState state = ValidationWorldScenePolicyApplier.CreateState(
            CreateScenePolicy(),
            new ValidationCaptureVariantPolicy(
                ShowTerrain: true,
                ShowTerrainLiquids: true,
                ShowObjects: true,
                ShowWmos: true,
                ShowDoodads: true,
                ShowSky: true,
                ShowWdl: true,
                ShowWorldLiquids: true));

        ValidationWorldScenePolicyApplier.ApplyVariantPolicy(
            state,
            CreateScenePolicy(),
            new ValidationCaptureVariantPolicy(
                ShowTerrain: true,
                ShowTerrainLiquids: true,
                ShowObjects: false,
                ShowWmos: true,
                ShowDoodads: true,
                ShowSky: true,
                ShowWdl: true,
                ShowWorldLiquids: true));

        Assert.False(state.ShowObjects);
        Assert.False(state.ShowWmos);
        Assert.False(state.ShowDoodads);
    }

    private static ValidationCaptureScenePolicy CreateScenePolicy()
    {
        return new ValidationCaptureScenePolicy(
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
    }
}
