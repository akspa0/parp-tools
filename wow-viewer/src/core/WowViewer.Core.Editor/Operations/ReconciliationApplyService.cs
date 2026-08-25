using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Reconciliation;

namespace WowViewer.Core.Editor.Operations;

/// <summary>One accepted proposal as recorded in the provenance sidecar.</summary>
public sealed record ReconciliationAppliedDecision(
    string ProposalId,
    string Action,
    string Disposition,
    string? AssetPath,
    int? EntryIndex,
    int? UniqueId,
    float PositionX,
    float PositionY,
    float PositionZ);

/// <summary>
/// The machine-readable operation/provenance report written beside the output (Spec 176 FR-016).
/// It names every accepted decision, the source fingerprints the apply was validated against, and
/// every ID allocation/name-table addition the edit produced.
/// </summary>
public sealed record ReconciliationProvenanceReport(
    string BatchId,
    string CreatedUtc,
    string GuideSourcePath,
    string? GuideSourceHash,
    string PlacementSourcePath,
    string PlacementSourceHash,
    string BuildFingerprint,
    IReadOnlyList<ReconciliationAppliedDecision> Decisions,
    IReadOnlyList<string> AddedModelNames,
    IReadOnlyList<string> AddedWorldModelNames,
    IReadOnlyList<int> AllocatedIds,
    string OutputHash);

/// <summary>Refuses an apply that would be stale, unsupported, or empty.</summary>
public sealed class ReconciliationApplyException : Exception
{
    public ReconciliationApplyException(string message) : base(message) { }
}

/// <summary>
/// Turns explicitly accepted reconciliation proposals into validated placement edits through the
/// existing <see cref="AdtPlacementEditor"/> (Spec 176 Phase 2). The service is pure: it stages
/// the output bytes in memory, validates the source fingerprint, and reports every allocation and
/// name-table addition. It writes no file itself; the caller owns the atomic commit.
/// </summary>
public static class ReconciliationApplyService
{
    public static string ComputeSha256Hex(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    /// <summary>
    /// Applies the accepted proposals to the source bytes. Align becomes a move; substitute
    /// becomes delete + add of the candidate asset (the allocated ID is reported, FR-003); clone
    /// becomes an add with the candidate asset and the proposal's transform. Any proposal that is
    /// not actionable (missing candidate, missing placement identity, non-review status) refuses
    /// the whole batch rather than approximating.
    /// </summary>
    public static (AdtPlacementEditResult Result, ReconciliationProvenanceReport Report) ApplyAccepted(
        IReadOnlyList<ReconciliationProposal> accepted,
        byte[] sourceBytes,
        string sourcePath,
        string expectedSourceHash,
        string guideSourcePath,
        string? guideSourceHash,
        string buildFingerprint)
    {
        ArgumentNullException.ThrowIfNull(accepted);
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(expectedSourceHash);
        ArgumentException.ThrowIfNullOrWhiteSpace(guideSourcePath);

        if (accepted.Count == 0)
            throw new ReconciliationApplyException("No accepted proposals were provided; nothing can be applied.");

        string actualSourceHash = ComputeSha256Hex(sourceBytes);
        if (!string.Equals(actualSourceHash, expectedSourceHash, StringComparison.OrdinalIgnoreCase))
            throw new ReconciliationApplyException(
                $"The placement source changed since the preview was built (expected {expectedSourceHash[..12]}…, found {actualSourceHash[..12]}…). Re-run the preview before applying.");

        var edits = new List<AdtPlacementEdit>();
        var decisions = new List<ReconciliationAppliedDecision>(accepted.Count);

        foreach (ReconciliationProposal proposal in accepted)
        {
            if (proposal.Status != ProposalStatus.ReviewRequired)
                throw new ReconciliationApplyException(
                    $"Proposal {proposal.ProposalId} has status {proposal.Status} and cannot be applied.");

            switch (proposal.Action)
            {
                case ReconciliationAction.Align:
                {
                    if (proposal.ExistingPlacement is null)
                        throw new ReconciliationApplyException(
                            $"Align proposal {proposal.ProposalId} has no existing placement identity.");

                    edits.Add(new AdtPlacementMoveEdit(
                        MapKind(proposal.ExistingPlacement.Kind),
                        proposal.ExistingPlacement.EntryIndex,
                        proposal.ExistingPlacement.UniqueId,
                        proposal.ProposedPosition));
                    break;
                }

                case ReconciliationAction.Substitute:
                {
                    if (proposal.ExistingPlacement is null || proposal.Candidate is null)
                        throw new ReconciliationApplyException(
                            $"Substitute proposal {proposal.ProposalId} needs both an existing placement and a resolved candidate.");

                    AdtPlacementKind kind = MapKind(proposal.ExistingPlacement.Kind);
                    edits.Add(new AdtPlacementDeleteEdit(kind, proposal.ExistingPlacement.EntryIndex, proposal.ExistingPlacement.UniqueId));
                    edits.Add(new AdtPlacementAddEdit(
                        MapKind(proposal.Candidate.Kind),
                        proposal.Candidate.AssetPath,
                        proposal.ProposedPosition,
                        proposal.ExistingPlacement is not null && proposal.Current is not null ? proposal.Current.Rotation : System.Numerics.Vector3.Zero,
                        proposal.Current is not null ? proposal.Current.Scale : 1f));
                    break;
                }

                case ReconciliationAction.Clone:
                {
                    if (proposal.Candidate is null)
                        throw new ReconciliationApplyException(
                            $"Clone proposal {proposal.ProposalId} has no resolved candidate asset.");

                    edits.Add(new AdtPlacementAddEdit(
                        MapKind(proposal.Candidate.Kind),
                        proposal.Candidate.AssetPath,
                        proposal.ProposedPosition,
                        proposal.ProposedRotation,
                        proposal.ProposedScale));
                    break;
                }

                default:
                    throw new ReconciliationApplyException($"Unsupported reconciliation action {proposal.Action}.");
            }

            decisions.Add(new ReconciliationAppliedDecision(
                proposal.ProposalId,
                proposal.Action.ToString(),
                "Accepted",
                proposal.Candidate?.AssetPath ?? proposal.ExistingPlacement?.AssetPath,
                proposal.ExistingPlacement?.EntryIndex,
                proposal.ExistingPlacement?.UniqueId,
                proposal.ProposedPosition.X,
                proposal.ProposedPosition.Y,
                proposal.ProposedPosition.Z));
        }

        AdtPlacementEditResult result = AdtPlacementEditor.Apply(sourceBytes, sourcePath, edits);

        var batchIdBuilder = new StringBuilder("batch|");
        foreach (ReconciliationProposal proposal in accepted)
            batchIdBuilder.Append(proposal.ProposalId).Append('|');
        string batchId = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(batchIdBuilder.ToString())).AsSpan(0, 16)).ToLowerInvariant();

        var report = new ReconciliationProvenanceReport(
            batchId,
            DateTimeOffset.UtcNow.ToString("O"),
            guideSourcePath,
            guideSourceHash,
            sourcePath,
            actualSourceHash,
            buildFingerprint,
            decisions,
            result.AddedModelNames,
            result.AddedWorldModelNames,
            result.AllocatedIds,
            ComputeSha256Hex(result.Bytes));

        return (result, report);
    }

    private static AdtPlacementKind MapKind(ExpectedAssetKind kind)
        => kind == ExpectedAssetKind.WorldModel ? AdtPlacementKind.WorldModel : AdtPlacementKind.Model;
}
