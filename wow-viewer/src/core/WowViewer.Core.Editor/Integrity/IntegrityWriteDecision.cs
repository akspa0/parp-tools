namespace WowViewer.Core.Editor.Integrity;

/// <summary>
/// The gate's decision for a requested write: either approved, or refused with the blocking inputs
/// named so the user is told exactly what failed and no partial file is produced.
/// </summary>
public readonly record struct IntegrityWriteDecision(bool IsApproved, IReadOnlyList<AssetValidationResult> BlockingInputs)
{
    public static IntegrityWriteDecision Approved() => new(true, []);

    public static IntegrityWriteDecision Refused(IReadOnlyList<AssetValidationResult> blockingInputs)
        => new(false, blockingInputs);
}