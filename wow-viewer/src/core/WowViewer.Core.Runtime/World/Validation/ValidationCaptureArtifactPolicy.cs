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