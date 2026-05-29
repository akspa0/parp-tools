namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureScenePolicy
{
    public ValidationCaptureScenePolicy(
        int requestedResolution,
        int requiredSettledFrames,
        int maxFramesBeforeCapture,
        int detailedTileCountOverride,
        float fogStartFactor,
        float fogEndDistance,
        float objectStreamingRangeMultiplierFloor,
        float maxVisibleMdxBoundsHeight,
        bool disableObjectFog,
        bool disableObjectPathFilters,
        bool hideWorldLiquids,
        bool ignoreTerrainHolesGlobally,
        bool hideUiChrome,
        bool enableRuntimeWmoGroupLiquids,
        bool enableRuntimeWmoGroupVisibility,
        ValidationCaptureArtifactPolicy artifactPolicy,
        bool ignoreDistanceCulling = false,
        bool ignoreProjectedSizeCulling = false,
        bool ignoreVisionConeCulling = false,
        bool ignoreFrustumCulling = false,
        bool ignoreMaxViewDistanceCulling = false,
        int batchSettledFrames = 2,
        bool fastSettleAfterBatchReady = true)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(requiredSettledFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxFramesBeforeCapture, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(detailedTileCountOverride, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSettledFrames, 1);
        if (float.IsNaN(fogStartFactor) || fogStartFactor < 0f || fogStartFactor > 1f)
            throw new ArgumentOutOfRangeException(nameof(fogStartFactor), fogStartFactor, "Fog start factor must be between 0 and 1.");
        if (float.IsNaN(fogEndDistance) || fogEndDistance <= 0f)
            throw new ArgumentOutOfRangeException(nameof(fogEndDistance), fogEndDistance, "Fog end distance must be greater than zero.");
        if (float.IsNaN(objectStreamingRangeMultiplierFloor) || objectStreamingRangeMultiplierFloor < 0f)
            throw new ArgumentOutOfRangeException(nameof(objectStreamingRangeMultiplierFloor), objectStreamingRangeMultiplierFloor, "Object streaming range multiplier floor cannot be negative.");
        if (float.IsNaN(maxVisibleMdxBoundsHeight) || maxVisibleMdxBoundsHeight < 0f)
            throw new ArgumentOutOfRangeException(nameof(maxVisibleMdxBoundsHeight), maxVisibleMdxBoundsHeight, "Max visible MDX bounds height cannot be negative.");
        if (artifactPolicy.ObjectsOnlyIntensityThreshold < 0)
            throw new ArgumentOutOfRangeException(nameof(artifactPolicy), artifactPolicy.ObjectsOnlyIntensityThreshold, "Objects-only intensity threshold cannot be negative.");
        if (artifactPolicy.DiffMaskThreshold < 0)
            throw new ArgumentOutOfRangeException(nameof(artifactPolicy), artifactPolicy.DiffMaskThreshold, "Diff mask threshold cannot be negative.");
        ArgumentException.ThrowIfNullOrWhiteSpace(artifactPolicy.ObjectVisibilityMaskFileSuffix);
        ArgumentException.ThrowIfNullOrWhiteSpace(artifactPolicy.NoObjectMinimapFileSuffix);

        RequestedResolution = requestedResolution;
        RequiredSettledFrames = requiredSettledFrames;
        MaxFramesBeforeCapture = maxFramesBeforeCapture;
        DetailedTileCountOverride = detailedTileCountOverride;
        FogStartFactor = fogStartFactor;
        FogEndDistance = fogEndDistance;
        ObjectStreamingRangeMultiplierFloor = objectStreamingRangeMultiplierFloor;
        MaxVisibleMdxBoundsHeight = maxVisibleMdxBoundsHeight;
        DisableObjectFog = disableObjectFog;
        DisableObjectPathFilters = disableObjectPathFilters;
        HideWorldLiquids = hideWorldLiquids;
        IgnoreTerrainHolesGlobally = ignoreTerrainHolesGlobally;
        HideUiChrome = hideUiChrome;
        EnableRuntimeWmoGroupLiquids = enableRuntimeWmoGroupLiquids;
        EnableRuntimeWmoGroupVisibility = enableRuntimeWmoGroupVisibility;
        IgnoreDistanceCulling = ignoreDistanceCulling;
        IgnoreProjectedSizeCulling = ignoreProjectedSizeCulling;
        IgnoreVisionConeCulling = ignoreVisionConeCulling;
        IgnoreFrustumCulling = ignoreFrustumCulling;
        IgnoreMaxViewDistanceCulling = ignoreMaxViewDistanceCulling;
        ArtifactPolicy = artifactPolicy;
        BatchSettledFrames = batchSettledFrames;
        FastSettleAfterBatchReady = fastSettleAfterBatchReady;
    }

    public int RequestedResolution { get; }

    public int RequiredSettledFrames { get; }

    public int MaxFramesBeforeCapture { get; }

    public int BatchSettledFrames { get; }

    public bool FastSettleAfterBatchReady { get; }

    public int DetailedTileCountOverride { get; }

    public float FogStartFactor { get; }

    public float FogEndDistance { get; }

    public float ObjectStreamingRangeMultiplierFloor { get; }

    public float MaxVisibleMdxBoundsHeight { get; }

    public bool DisableObjectFog { get; }

    public bool DisableObjectPathFilters { get; }

    public bool HideWorldLiquids { get; }

    public bool IgnoreTerrainHolesGlobally { get; }

    public bool HideUiChrome { get; }

    public bool EnableRuntimeWmoGroupLiquids { get; }

    public bool EnableRuntimeWmoGroupVisibility { get; }

    public bool IgnoreDistanceCulling { get; }

    public bool IgnoreProjectedSizeCulling { get; }

    public bool IgnoreVisionConeCulling { get; }

    public bool IgnoreFrustumCulling { get; }

    public bool IgnoreMaxViewDistanceCulling { get; }

    public ValidationCaptureArtifactPolicy ArtifactPolicy { get; }
}
