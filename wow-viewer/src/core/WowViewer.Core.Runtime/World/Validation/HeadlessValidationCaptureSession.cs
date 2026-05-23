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
        if (variantPolicies.Count == 0)
            throw new ArgumentException("Variant policies cannot be empty.", nameof(variantPolicies));

        foreach (ValidationCaptureVariant variant in Enum.GetValues<ValidationCaptureVariant>())
        {
            if (!variantPolicies.ContainsKey(variant))
                throw new ArgumentException($"Variant policies must include {variant}.", nameof(variantPolicies));
        }

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