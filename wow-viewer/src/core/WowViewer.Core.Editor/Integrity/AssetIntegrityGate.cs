namespace WowViewer.Core.Editor.Integrity;

/// <summary>
/// The detection-and-refusal boundary (Spec 173): nothing is written whose contributing inputs did
/// not verify, and nothing is reported as saved until it re-reads and verifies. Lossy downports must
/// record their losses; an untraceable write is refused.
/// </summary>
public sealed class AssetIntegrityGate
{
    private readonly IAssetValidator _validator;

    public AssetIntegrityGate(IAssetValidator validator)
    {
        _validator = validator ?? throw new ArgumentNullException(nameof(validator));
    }

    /// <summary>FR-001: validate an asset on read.</summary>
    public AssetValidationResult ValidateOnRead(string path, ReadOnlyMemory<byte> bytes)
        => _validator.Validate(new AssetReadInput(path, bytes));

    /// <summary>FR-002: approve or refuse a write based on every contributing input's verdict.</summary>
    public IntegrityWriteDecision AuthorizeWrite(IReadOnlyList<AssetValidationResult> contributingInputs)
    {
        ArgumentNullException.ThrowIfNull(contributingInputs);

        var blocking = new List<AssetValidationResult>();
        foreach (AssetValidationResult input in contributingInputs)
        {
            if (input.Verdict != ValidationVerdict.Verified)
                blocking.Add(input);
        }

        return blocking.Count == 0 ? IntegrityWriteDecision.Approved() : IntegrityWriteDecision.Refused(blocking);
    }

    /// <summary>FR-003: re-read and verify a written file before it can be reported as saved.</summary>
    public bool VerifyWrittenFile(ReadOnlyMemory<byte> writtenBytes, string writtenPath, out AssetValidationResult result)
    {
        result = ValidateOnRead(writtenPath, writtenBytes);
        return result.Verdict == ValidationVerdict.Verified;
    }

    /// <summary>FR-006: an output without complete provenance is refused.</summary>
    public bool HasCompleteProvenance(AssetProvenance? provenance)
        => provenance is not null && provenance.IsComplete;

    /// <summary>
    /// A lossy downport that cannot state its losses is not verified. This helper returns the losses
    /// to record on the provenance record; a null result means "cannot state losses" and must refuse.
    /// </summary>
    public IReadOnlyList<string>? RecordedLosses(IEnumerable<string>? lossDescriptions)
        => lossDescriptions?.ToArray();
}