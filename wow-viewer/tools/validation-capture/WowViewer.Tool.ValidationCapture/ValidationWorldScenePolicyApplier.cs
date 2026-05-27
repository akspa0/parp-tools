using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Tools.ValidationCapture;

internal sealed class ValidationWorldScenePolicyState
{
    public int RequestedResolution { get; set; }

    public bool HideUiChrome { get; set; }

    public int DetailedTileCountOverride { get; set; }

    public float FogStartFactor { get; set; }

    public float FogEndDistance { get; set; }

    public bool ObjectFogEnabled { get; set; } = true;

    public bool ShowTerrain { get; set; } = true;

    public bool ShowTerrainLiquids { get; set; } = true;

    public bool ShowWorldLiquids { get; set; } = true;

    public bool ShowObjects { get; set; } = true;

    public bool ShowWmos { get; set; } = true;

    public bool ShowDoodads { get; set; } = true;

    public bool ShowSky { get; set; } = true;

    public bool ShowWdl { get; set; }

    public bool IgnoreTerrainHolesGlobally { get; set; }

    public bool ObjectPathFiltersEnabled { get; set; } = true;

    public float ObjectStreamingRangeMultiplier { get; set; } = 0.5f;

    public float MaxVisibleMdxBoundsHeight { get; set; }

    public bool EnableRuntimeWmoGroupVisibility { get; set; } = true;

    public bool EnableRuntimeWmoGroupLiquids { get; set; } = true;

    public bool IgnoreDistanceCulling { get; set; }

    public bool IgnoreProjectedSizeCulling { get; set; }

    public bool IgnoreVisionConeCulling { get; set; }

    public bool IgnoreFrustumCulling { get; set; }

    public bool IgnoreMaxViewDistanceCulling { get; set; }
}

internal static class ValidationWorldScenePolicyApplier
{
    public static ValidationWorldScenePolicyState CreateState(
        ValidationCaptureScenePolicy scenePolicy,
        ValidationCaptureVariantPolicy variantPolicy)
    {
        ArgumentNullException.ThrowIfNull(scenePolicy);

        ValidationWorldScenePolicyState state = new();
        ApplyScenePolicy(state, scenePolicy);
        ApplyVariantPolicy(state, scenePolicy, variantPolicy);
        return state;
    }

    public static void ApplyScenePolicy(
        ValidationWorldScenePolicyState state,
        ValidationCaptureScenePolicy scenePolicy)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(scenePolicy);

        state.RequestedResolution = scenePolicy.RequestedResolution;
        state.HideUiChrome = scenePolicy.HideUiChrome;
        state.DetailedTileCountOverride = scenePolicy.DetailedTileCountOverride;
        state.FogStartFactor = scenePolicy.FogStartFactor;
        state.FogEndDistance = scenePolicy.FogEndDistance;
        state.ObjectFogEnabled = !scenePolicy.DisableObjectFog;
        state.ShowWorldLiquids = !scenePolicy.HideWorldLiquids;
        state.IgnoreTerrainHolesGlobally = scenePolicy.IgnoreTerrainHolesGlobally;
        state.ObjectPathFiltersEnabled = !scenePolicy.DisableObjectPathFilters;
        state.ObjectStreamingRangeMultiplier = Math.Max(state.ObjectStreamingRangeMultiplier, scenePolicy.ObjectStreamingRangeMultiplierFloor);
        state.MaxVisibleMdxBoundsHeight = scenePolicy.MaxVisibleMdxBoundsHeight;
        state.EnableRuntimeWmoGroupVisibility = scenePolicy.EnableRuntimeWmoGroupVisibility;
        state.EnableRuntimeWmoGroupLiquids = scenePolicy.EnableRuntimeWmoGroupLiquids;
        state.IgnoreDistanceCulling = scenePolicy.IgnoreDistanceCulling;
        state.IgnoreProjectedSizeCulling = scenePolicy.IgnoreProjectedSizeCulling;
        state.IgnoreVisionConeCulling = scenePolicy.IgnoreVisionConeCulling;
        state.IgnoreFrustumCulling = scenePolicy.IgnoreFrustumCulling;
        state.IgnoreMaxViewDistanceCulling = scenePolicy.IgnoreMaxViewDistanceCulling;
    }

    public static void ApplyVariantPolicy(
        ValidationWorldScenePolicyState state,
        ValidationCaptureScenePolicy scenePolicy,
        ValidationCaptureVariantPolicy variantPolicy)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(scenePolicy);

        state.ShowTerrain = variantPolicy.ShowTerrain;
        state.ShowTerrainLiquids = variantPolicy.ShowTerrainLiquids;
        state.ShowObjects = variantPolicy.ShowObjects;
        state.ShowWmos = variantPolicy.ShowObjects && variantPolicy.ShowWmos;
        state.ShowDoodads = variantPolicy.ShowObjects && variantPolicy.ShowDoodads;
        state.ShowSky = variantPolicy.ShowSky;
        state.ShowWdl = variantPolicy.ShowWdl;
        state.ShowWorldLiquids = !scenePolicy.HideWorldLiquids && variantPolicy.ShowWorldLiquids;
        state.EnableRuntimeWmoGroupLiquids = scenePolicy.EnableRuntimeWmoGroupLiquids && variantPolicy.ShowTerrainLiquids;
    }
}
