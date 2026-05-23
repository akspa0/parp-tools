# Public Contract Shapes: Real Validation Batch Extraction

**Spec**: [spec.md](./spec.md)

**Plan**: [plan.md](./plan.md)

## Purpose

This note freezes the proposed public type shapes for the first implementation wave before code is written.

The goal is narrow:

- remove ambiguity from the first implementation pass
- keep the first slices focused on shared validation-batch contracts plus the minimum host or adapter seams needed to consume them
- match the current `WowViewer.Core.Runtime.World` style closely enough that the new files feel native to the existing runtime surface

These shapes now cover:

- Phase 1 shared validation-contract files
- Phase 2 readiness and artifact files
- Phase 3 headless host entry seams
- Phase 4 renderer-adapter seams

## Ownership Targets

Shared runtime ownership:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Headless tool ownership:

- `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`

## Execution Order

Implement these files in this order unless a later step proves impossible without a local adjustment:

1. `ValidationCaptureVariant.cs`, `ValidationCaptureTileRequest.cs`, `ValidationCaptureBatchPlan.cs`, `ValidationCaptureVariantResult.cs`, `ValidationCaptureBatchResult.cs`
2. `ValidationCaptureScenePolicy.cs`, `ValidationCaptureVariantPolicy.cs`, `ValidationCaptureArtifactPolicy.cs`, `ValidationCaptureCameraInput.cs`, `ValidationCaptureCameraFrame.cs`, `ValidationCaptureCameraSolver.cs`
3. `ValidationCaptureReadinessSnapshot.cs`, `ValidationCaptureReadinessState.cs`, `ValidationCaptureReadinessEvaluator.cs`, `ValidationCaptureArtifactInputs.cs`, `ValidationCaptureArtifactOutputs.cs`, `ValidationCaptureArtifactBuilder.cs`
4. focused tests in `wow-viewer/tests/WowViewer.Core.Tests/`
5. `Program.cs`, `ValidationCaptureCommand.cs`, `HeadlessValidationCaptureSession.cs`
6. `HeadlessValidationCaptureRunner.cs`, `HeadlessValidationFramebufferExporter.cs`, `IValidationWorldSceneAdapter`, `ValidationWorldSceneSnapshot.cs`, `ValidationWorldScenePolicyApplier.cs`

## Non-Negotiable Invariants

- Do not route the replacement lane through `wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs`.
- Do not add viewer-shell or ImGui types to any shared-runtime contract under `WowViewer.Core.Runtime`.
- Keep contract payloads on primitive values, byte buffers, hashes, counts, and validated strings; do not introduce UI-owned image objects into shared-runtime public shapes.
- Do not widen scope from one bounded tile and four variants to map-wide batching before bounded parity exists.
- Do not claim parity from a build or test pass alone; parity requires real-data proof against staged `0_5_3_3368` and staged `3_3_5_12340` anchors.

## Proposed File: ValidationCaptureVariant.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public enum ValidationCaptureVariant
{
    Primary = 0,
    NoLiquids = 1,
    NoObjects = 2,
    ObjectsOnly = 3,
}
```

## Proposed File: ValidationCaptureTileRequest.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureTileRequest
{
    public ValidationCaptureTileRequest(
        string tileName,
        int tileX,
        int tileY,
        ValidationCaptureVariant variant,
        string outputPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        TileName = tileName;
        TileX = tileX;
        TileY = tileY;
        Variant = variant;
        OutputPath = outputPath;
    }

    public string TileName { get; }

    public int TileX { get; }

    public int TileY { get; }

    public ValidationCaptureVariant Variant { get; }

    public string OutputPath { get; }
}
```

## Proposed File: ValidationCaptureBatchPlan.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureBatchPlan
{
    public ValidationCaptureBatchPlan(
        string datasetRoot,
        string mapName,
        string primaryOutputDirectory,
        string noLiquidsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory,
        int requestedResolution,
        string? buildLabel,
        IReadOnlyList<ValidationCaptureTileRequest> tileRequests)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(datasetRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentException.ThrowIfNullOrWhiteSpace(primaryOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(noLiquidsOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(noObjectsOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(objectsOnlyOutputDirectory);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentNullException.ThrowIfNull(tileRequests);

        DatasetRoot = datasetRoot;
        MapName = mapName;
        PrimaryOutputDirectory = primaryOutputDirectory;
        NoLiquidsOutputDirectory = noLiquidsOutputDirectory;
        NoObjectsOutputDirectory = noObjectsOutputDirectory;
        ObjectsOnlyOutputDirectory = objectsOnlyOutputDirectory;
        RequestedResolution = requestedResolution;
        BuildLabel = buildLabel;
        TileRequests = tileRequests;
    }

    public string DatasetRoot { get; }

    public string MapName { get; }

    public string PrimaryOutputDirectory { get; }

    public string NoLiquidsOutputDirectory { get; }

    public string NoObjectsOutputDirectory { get; }

    public string ObjectsOnlyOutputDirectory { get; }

    public int RequestedResolution { get; }

    public string? BuildLabel { get; }

    public IReadOnlyList<ValidationCaptureTileRequest> TileRequests { get; }

    public int RequestCount => TileRequests.Count;
}
```

## Proposed File: ValidationCaptureVariantResult.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureVariantResult
{
    public ValidationCaptureVariantResult(
        ValidationCaptureVariant variant,
        string tileName,
        int tileX,
        int tileY,
        string outputPath,
        ValidationCaptureReadinessState readinessState,
        bool succeeded,
        bool timedOut,
        int framesObserved,
        int settledFrames,
        string? failureReason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentOutOfRangeException.ThrowIfNegative(framesObserved);
        ArgumentOutOfRangeException.ThrowIfNegative(settledFrames);

        Variant = variant;
        TileName = tileName;
        TileX = tileX;
        TileY = tileY;
        OutputPath = outputPath;
        ReadinessState = readinessState;
        Succeeded = succeeded;
        TimedOut = timedOut;
        FramesObserved = framesObserved;
        SettledFrames = settledFrames;
        FailureReason = failureReason;
    }

    public ValidationCaptureVariant Variant { get; }

    public string TileName { get; }

    public int TileX { get; }

    public int TileY { get; }

    public string OutputPath { get; }

    public ValidationCaptureReadinessState ReadinessState { get; }

    public bool Succeeded { get; }

    public bool TimedOut { get; }

    public int FramesObserved { get; }

    public int SettledFrames { get; }

    public string? FailureReason { get; }
}
```

## Proposed File: ValidationCaptureBatchResult.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureBatchResult
{
    public ValidationCaptureBatchResult(
        string mapName,
        string? buildLabel,
        int requestedResolution,
        IReadOnlyList<ValidationCaptureVariantResult> variantResults)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentNullException.ThrowIfNull(variantResults);

        MapName = mapName;
        BuildLabel = buildLabel;
        RequestedResolution = requestedResolution;
        VariantResults = variantResults;
    }

    public string MapName { get; }

    public string? BuildLabel { get; }

    public int RequestedResolution { get; }

    public IReadOnlyList<ValidationCaptureVariantResult> VariantResults { get; }

    public int TotalVariantCount => VariantResults.Count;

    public int SucceededVariantCount => VariantResults.Count(static result => result.Succeeded);

    public int TimedOutVariantCount => VariantResults.Count(static result => result.TimedOut);

    public int FailedVariantCount => VariantResults.Count(static result => !result.Succeeded);
}
```

## Proposed File: ValidationCaptureScenePolicy.cs

```csharp
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
        ValidationCaptureArtifactPolicy artifactPolicy)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(requiredSettledFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxFramesBeforeCapture, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(detailedTileCountOverride, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(fogStartFactor);
        ArgumentOutOfRangeException.ThrowIfNegative(fogEndDistance);
        ArgumentOutOfRangeException.ThrowIfNegative(objectStreamingRangeMultiplierFloor);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVisibleMdxBoundsHeight);

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
        ArtifactPolicy = artifactPolicy;
    }

    public int RequestedResolution { get; }

    public int RequiredSettledFrames { get; }

    public int MaxFramesBeforeCapture { get; }

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

    public ValidationCaptureArtifactPolicy ArtifactPolicy { get; }
}
```

## Proposed File: ValidationCaptureVariantPolicy.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureVariantPolicy(
    bool ShowTerrain,
    bool ShowTerrainLiquids,
    bool ShowObjects,
    bool ShowWmos,
    bool ShowDoodads,
    bool ShowSky,
    bool ShowWdl,
    bool ShowWorldLiquids);
```

## Proposed File: ValidationCaptureArtifactPolicy.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public enum ValidationObjectMaskStrategy
{
    DirectObjectsOnlySilhouette = 0,
    PrimaryVsNoObjectsDiff = 1,
}

public readonly record struct ValidationCaptureArtifactPolicy(
    ValidationObjectMaskStrategy EarlyBuildStrategy,
    ValidationObjectMaskStrategy LaterBuildStrategy,
    int ObjectsOnlyIntensityThreshold,
    int DiffMaskThreshold,
    string ObjectVisibilityMaskFileSuffix,
    string NoObjectMinimapFileSuffix);
```

## Proposed File: ValidationCaptureCameraInput.cs

```csharp
using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureCameraInput(
    int TileX,
    int TileY,
    float AspectRatio,
    float GroundHeight,
    float MapOrigin,
    float TileWorldSize,
    float DesiredSpan,
    float EyeHeightOffset,
    float NearPlane,
    float FarPlane,
    Vector3 Up);
```

## Proposed File: ValidationCaptureCameraFrame.cs

```csharp
using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureCameraFrame(
    Vector3 Eye,
    Vector3 Target,
    Vector3 Up,
    float WorldSpanX,
    float WorldSpanY,
    Matrix4x4 View,
    Matrix4x4 Projection);
```

## Proposed File: ValidationCaptureCameraSolver.cs

```csharp
using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureCameraSolver
{
    public static Vector2 ComputeTileCenter(
        int tileX,
        int tileY,
        float mapOrigin,
        float tileWorldSize);

    public static ValidationCaptureCameraFrame SolveTopDown(
        ValidationCaptureCameraInput input);
}
```

## Proposed File: ValidationCaptureReadinessSnapshot.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureReadinessSnapshot(
    bool HasSceneContent,
    bool HasFramebuffer,
    int FramebufferWidth,
    int FramebufferHeight,
    int RequestedResolution,
    bool WaitForSceneReady,
    bool HasTargetTile,
    bool TargetTileLoaded,
    bool TerrainStreaming,
    bool TrackPendingWorldObjectLoads,
    int PendingWorldObjectLoadCount,
    int FramesObserved,
    int SettledFrames,
    int RequiredSettledFrames,
    int MaxFramesBeforeCapture);
```

## Proposed File: ValidationCaptureReadinessState.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public enum ValidationCaptureReadinessStatus
{
    Ready = 0,
    WaitingForSceneContent = 1,
    WaitingForFramebuffer = 2,
    WaitingForFramebufferResolution = 3,
    WaitingForWorldObjectLoads = 4,
    WaitingForTargetTile = 5,
    WaitingForSettledFrames = 6,
    TimedOut = 7,
}

public readonly record struct ValidationCaptureReadinessState(
    ValidationCaptureReadinessStatus Status,
    bool IsReady,
    bool TimedOut,
    int FramesObserved,
    int SettledFrames,
    string? Detail)
{
    public static ValidationCaptureReadinessState Ready(int framesObserved, int settledFrames)
        => new(
            ValidationCaptureReadinessStatus.Ready,
            IsReady: true,
            TimedOut: false,
            FramesObserved: framesObserved,
            SettledFrames: settledFrames,
            Detail: null);
}
```

## Proposed File: ValidationCaptureReadinessEvaluator.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureReadinessEvaluator
{
    public static ValidationCaptureReadinessState Evaluate(
        ValidationCaptureReadinessSnapshot snapshot);
}
```

## Proposed File: ValidationCaptureArtifactInputs.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureArtifactInputs
{
    public ValidationCaptureArtifactInputs(
        string tileName,
        string? buildLabel,
        int width,
        int height,
        byte[] primaryRgbaPixels,
        byte[] noObjectsRgbaPixels,
        byte[]? objectsOnlyRgbaPixels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(primaryRgbaPixels);
        ArgumentNullException.ThrowIfNull(noObjectsRgbaPixels);

        int rgbaLength = checked(width * height * 4);
        if (primaryRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("Primary RGBA payload length must match width * height * 4.", nameof(primaryRgbaPixels));
        if (noObjectsRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("No-objects RGBA payload length must match width * height * 4.", nameof(noObjectsRgbaPixels));
        if (objectsOnlyRgbaPixels != null && objectsOnlyRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("Objects-only RGBA payload length must match width * height * 4.", nameof(objectsOnlyRgbaPixels));

        TileName = tileName;
        BuildLabel = buildLabel;
        Width = width;
        Height = height;
        PrimaryRgbaPixels = primaryRgbaPixels;
        NoObjectsRgbaPixels = noObjectsRgbaPixels;
        ObjectsOnlyRgbaPixels = objectsOnlyRgbaPixels;
    }

    public string TileName { get; }

    public string? BuildLabel { get; }

    public int Width { get; }

    public int Height { get; }

    public byte[] PrimaryRgbaPixels { get; }

    public byte[] NoObjectsRgbaPixels { get; }

    public byte[]? ObjectsOnlyRgbaPixels { get; }
}
```

## Proposed File: ValidationCaptureArtifactOutputs.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureArtifactOutputs
{
    public ValidationCaptureArtifactOutputs(
        string tileName,
        string? buildLabel,
        int width,
        int height,
        ValidationObjectMaskStrategy maskStrategy,
        byte[] objectVisibilityMaskL8Pixels,
        string objectVisibilityMaskHash,
        byte[] noObjectMinimapRgbaPixels,
        string noObjectMinimapHash)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(objectVisibilityMaskL8Pixels);
        ArgumentException.ThrowIfNullOrWhiteSpace(objectVisibilityMaskHash);
        ArgumentNullException.ThrowIfNull(noObjectMinimapRgbaPixels);
        ArgumentException.ThrowIfNullOrWhiteSpace(noObjectMinimapHash);

        int maskLength = checked(width * height);
        int rgbaLength = checked(width * height * 4);
        if (objectVisibilityMaskL8Pixels.Length != maskLength)
            throw new ArgumentException("Object-visibility mask payload length must match width * height.", nameof(objectVisibilityMaskL8Pixels));
        if (noObjectMinimapRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("No-object minimap RGBA payload length must match width * height * 4.", nameof(noObjectMinimapRgbaPixels));

        TileName = tileName;
        BuildLabel = buildLabel;
        Width = width;
        Height = height;
        MaskStrategy = maskStrategy;
        ObjectVisibilityMaskL8Pixels = objectVisibilityMaskL8Pixels;
        ObjectVisibilityMaskHash = objectVisibilityMaskHash;
        NoObjectMinimapRgbaPixels = noObjectMinimapRgbaPixels;
        NoObjectMinimapHash = noObjectMinimapHash;
    }

    public string TileName { get; }

    public string? BuildLabel { get; }

    public int Width { get; }

    public int Height { get; }

    public ValidationObjectMaskStrategy MaskStrategy { get; }

    public byte[] ObjectVisibilityMaskL8Pixels { get; }

    public string ObjectVisibilityMaskHash { get; }

    public byte[] NoObjectMinimapRgbaPixels { get; }

    public string NoObjectMinimapHash { get; }
}
```

## Proposed File: ValidationCaptureArtifactBuilder.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureArtifactBuilder
{
    public static ValidationObjectMaskStrategy ResolveMaskStrategy(
        string? buildLabel,
        ValidationCaptureArtifactPolicy policy);

    public static ValidationCaptureArtifactOutputs Build(
        ValidationCaptureArtifactInputs inputs,
        ValidationCaptureArtifactPolicy policy);
}
```

## Proposed File: HeadlessValidationCaptureSession.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public sealed class HeadlessValidationCaptureSession
{
    public HeadlessValidationCaptureSession(
        string clientRoot,
        string mapInput,
        string? buildLabel,
        string? looseOverlayRoot,
        ValidationCaptureBatchPlan batchPlan,
        ValidationCaptureScenePolicy scenePolicy,
        IReadOnlyDictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> variantPolicies)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapInput);
        ArgumentNullException.ThrowIfNull(batchPlan);
        ArgumentNullException.ThrowIfNull(scenePolicy);
        ArgumentNullException.ThrowIfNull(variantPolicies);

        ClientRoot = clientRoot;
        MapInput = mapInput;
        BuildLabel = buildLabel;
        LooseOverlayRoot = looseOverlayRoot;
        BatchPlan = batchPlan;
        ScenePolicy = scenePolicy;
        VariantPolicies = variantPolicies;
    }

    public string ClientRoot { get; }

    public string MapInput { get; }

    public string? BuildLabel { get; }

    public string? LooseOverlayRoot { get; }

    public ValidationCaptureBatchPlan BatchPlan { get; }

    public ValidationCaptureScenePolicy ScenePolicy { get; }

    public IReadOnlyDictionary<ValidationCaptureVariant, ValidationCaptureVariantPolicy> VariantPolicies { get; }
}
```

## Proposed File: HeadlessValidationFramebufferExporter.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public static class HeadlessValidationFramebufferExporter
{
    public static void WriteImage(
        string outputPath,
        int width,
        int height,
        byte[] rgbaPixels,
        bool sourceOriginBottomLeft);
}
```

## Proposed File: HeadlessValidationCaptureRunner.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public static class HeadlessValidationCaptureRunner
{
    public static ValidationCaptureBatchResult Run(
        HeadlessValidationCaptureSession session,
        IValidationWorldSceneAdapter sceneAdapter);
}
```

## Proposed File: ValidationCaptureCommand.cs

```csharp
namespace WowViewer.Tools.ValidationCapture;

internal static class ValidationCaptureCommand
{
    public static int Execute(string[] args);
}
```

## Proposed File: ValidationWorldSceneSnapshot.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationWorldSceneSnapshot(
    bool HasSceneContent,
    int FramebufferWidth,
    int FramebufferHeight,
    bool TargetTileLoaded,
    bool TerrainStreaming,
    int PendingWorldObjectLoadCount);
```

## Proposed File: ValidationWorldSceneAdapter.cs

```csharp
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
```

## Proposed File: ValidationWorldScenePolicyApplier.cs

```csharp
namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationWorldScenePolicyApplier
{
    public static void ApplyScenePolicy(
        IValidationWorldSceneAdapter sceneAdapter,
        ValidationCaptureScenePolicy scenePolicy);

    public static void ApplyVariantPolicy(
        IValidationWorldSceneAdapter sceneAdapter,
        ValidationCaptureVariantPolicy variantPolicy);
}
```

## Notes On Shape Choices

- `ValidationCaptureVariant` stays an enum because the variant family is closed and used heavily in switching and policy lookup.
- The batch plan and batch result stay as sealed classes rather than record structs because they carry reference-typed collections and should read as validated aggregate roots.
- `ValidationCaptureVariantPolicy`, `ValidationCaptureArtifactPolicy`, `ValidationCaptureCameraInput`, and `ValidationCaptureCameraFrame` stay as readonly record structs because they are compact immutable value carriers.
- `ValidationCaptureCameraSolver` stays static because Phase 1 only needs deterministic math, not service lifetime or injected dependencies.
- The readiness and artifact types stay shared-runtime and byte-buffer based so the replacement lane does not depend on the preview-only app renderer or on UI-owned image surfaces.
- `HeadlessValidationCaptureRunner` and `ValidationCaptureCommand` stay static because existing tool command surfaces in this repo are predominantly static command classes or static runner entrypoints.
- The scene adapter is the one place where an interface is preferred: it keeps the headless runner bound to the real renderer contract without coupling the shared runtime layer to one concrete app-shell implementation class.

## Phase Gates

- Phase 1 is complete only when the shared-runtime models, policy types, and camera solver exist without any tool-project dependency and have focused tests where practical.
- Phase 2 is complete only when readiness and artifact logic are proven in `WowViewer.Core.Tests` without invoking any real renderer host.
- Phase 3 may create the tool project and runner shell, but it must stop short of inventing a preview substitute or a second fake renderer contract.
- Phase 4 is the first point where a concrete real-renderer binding is allowed; keep all renderer-specific state behind `IValidationWorldSceneAdapter`.
- Phase 5 starts only after the four-variant bounded tile proof exists on the staged proof builds and the shared artifact builder can reproduce the legacy artifact family.

## Deliberate Deferrals

These shapes now cover the shared-runtime slice plus the first headless-host and renderer-adapter slice.

Still deferred on purpose:

- `ValidationCaptureArtifactCommand.cs`
- concrete CLI option names beyond the first `ValidationCaptureCommand.Execute(string[] args)` seam
- any concrete adapter implementation class that binds the interface to a specific renderer host

The shared-runtime slices and the first headless-host or adapter slice can now be implemented without guessing public signatures.
