namespace WowViewer.Tools.Shared.Pm4Matching;

public enum Pm4MatchCommand
{
    MatchAssets,
    SynthesizePlacements,
    ExportAssetSignals,
}

public sealed record Pm4MatchRunOptions(
    Pm4MatchCommand Command,
    string InputPath,
    string? OutputPath,
    int MaxCandidates,
    string? AssetCorpusPath,
    string? PlacementsPath,
    string? ArchiveRoot,
    string? ListfilePath,
    IReadOnlyList<string>? TargetTiles,
    string? SeedPlacements)
{
    public static class Validation
    {
        public static Pm4MatchRunOptionsErrorCode Validate(Pm4MatchRunOptions options)
        {
            if (string.IsNullOrWhiteSpace(options.InputPath))
                return Pm4MatchRunOptionsErrorCode.InputPathRequired;
            if (options.MaxCandidates <= 0)
                return Pm4MatchRunOptionsErrorCode.MaxCandidatesMustBePositive;

            if (!string.IsNullOrWhiteSpace(options.AssetCorpusPath) && !string.IsNullOrWhiteSpace(options.PlacementsPath))
                return Pm4MatchRunOptionsErrorCode.AssetCorpusAndPlacementsAreMutuallyExclusive;

            if (options.Command != Pm4MatchCommand.ExportAssetSignals)
            {
                if (string.IsNullOrWhiteSpace(options.AssetCorpusPath) && string.IsNullOrWhiteSpace(options.ArchiveRoot))
                    return Pm4MatchRunOptionsErrorCode.ArchiveRootRequiredWithoutAssetCorpus;
            }

            if (options.Command == Pm4MatchCommand.SynthesizePlacements)
            {
                if (options.TargetTiles is null || options.TargetTiles.Count == 0)
                    return Pm4MatchRunOptionsErrorCode.TargetTilesRequired;
            }

            return Pm4MatchRunOptionsErrorCode.None;
        }

        public static string FormatErrorMessage(Pm4MatchRunOptionsErrorCode code)
        {
            return code switch
            {
                Pm4MatchRunOptionsErrorCode.None => string.Empty,
                Pm4MatchRunOptionsErrorCode.InputPathRequired =>
                    "Error: input PM4 file is required.",
                Pm4MatchRunOptionsErrorCode.MaxCandidatesMustBePositive =>
                    "Error: --max-candidates must be a positive integer.",
                Pm4MatchRunOptionsErrorCode.AssetCorpusAndPlacementsAreMutuallyExclusive =>
                    "Error: choose either --asset-corpus <report.json> or --placements <tile_obj0.adt>, not both.",
                Pm4MatchRunOptionsErrorCode.ArchiveRootRequiredWithoutAssetCorpus =>
                    "Error: --archive-root is required for pm4 match-assets so WMO/M2 assets can be read from the staged client.",
                Pm4MatchRunOptionsErrorCode.TargetTilesRequired =>
                    "Error: provide at least one tile in --target-tiles <x_y[,x_y...]>.",
                _ => $"Error: invalid run options ({code}).",
            };
        }
    }
}

public enum Pm4MatchRunOptionsErrorCode
{
    None,
    InputPathRequired,
    MaxCandidatesMustBePositive,
    AssetCorpusAndPlacementsAreMutuallyExclusive,
    ArchiveRootRequiredWithoutAssetCorpus,
    TargetTilesRequired,
}
