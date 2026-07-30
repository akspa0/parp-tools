using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Mismatch;

/// <summary>
/// Checks that <see cref="TerrainTileTensorPack.AvailableSignals"/> (the harvester's own "this
/// signal is present" claim for a tile) is not lying about a signal's presence -- i.e. the flag
/// says a signal is available but the tensor pack carries no data for it. This is the has-flag
/// truthfulness check named in spec FR-005/data-model.md, bounded to the signals this library's
/// other checks already depend on (normal_xyz, alpha_256, mcly, liquid) rather than every signal
/// in the catalog, since those are the ones a lying flag would silently corrupt.
/// </summary>
public static class HasFlagTruthfulnessDetector
{
    private static readonly (string FlagName, string SignalLabel)[] CheckedFlags =
    [
        ("has_normal_xyz", "normal_xyz"),
        ("has_alpha_256", "alpha_256"),
        ("has_mcly_texture_ids", "mcly_texture_ids"),
        ("has_liquid_mask", "unified_liquid_mask"),
    ];

    public static IReadOnlyList<MismatchFinding> Detect(
        TerrainTileTensorPack pack,
        string build,
        string map,
        int tileX,
        int tileY,
        long tileId,
        string curationRunId)
    {
        ArgumentNullException.ThrowIfNull(pack);

        var findings = new List<MismatchFinding>();
        foreach ((string flagName, string signalLabel) in CheckedFlags)
        {
            if (!pack.AvailableSignals.Contains(flagName))
                continue; // Flag not claimed present -- nothing to contradict.

            bool backingDataPresent = signalLabel switch
            {
                "normal_xyz" => pack.McnrNormalXyz is not null,
                "alpha_256" => pack.McalAlphaPack256 is not null,
                "mcly_texture_ids" => pack.MclyTextureIds is not null,
                "unified_liquid_mask" => pack.UnifiedLiquidMask is not null,
                _ => true,
            };

            if (!backingDataPresent)
            {
                findings.Add(new MismatchFinding(build, map, tileX, tileY, tileId,
                    WowViewer.Core.Curation.MismatchCategory.HasFlagMismatch,
                    WowViewer.Core.Curation.MismatchSeverity.High,
                    $"{flagName}_claims_present_but_{signalLabel}_is_absent",
                    WowViewer.Core.Curation.Evaluability.Evaluated,
                    Signal: signalLabel,
                    curationRunId));
            }
        }
        return findings;
    }
}
