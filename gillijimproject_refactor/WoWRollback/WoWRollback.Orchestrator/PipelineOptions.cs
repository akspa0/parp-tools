using System.Collections.Generic;

namespace WoWRollback.Orchestrator;

internal sealed record PipelineOptions(
    IReadOnlyList<string> Maps,
    IReadOnlyList<string> Versions,
    string AlphaRoot,
    string OutputRoot,
    string? DbdDirectory,
    string? LkDbcDirectory,
    string? CommunityListfile,
    string? LkListfile,
    string? NoggitClientPath,
    bool RunVerifier,
    string? AreaOverrideDirectory,
    bool Serve,
    int Port,
    bool Verbose,
    string? MpqPath,
    string? MinimapRoot,
    string? MpqLocales,
    string? ViewerLabel,
    string? ViewerAssetsPath)
{
    public bool HasLkDbcDirectory => !string.IsNullOrWhiteSpace(LkDbcDirectory);
    public bool HasMpqPath => !string.IsNullOrWhiteSpace(MpqPath);
    public string? MpqRoot => MpqPath; // Alias for ViewerBuildStageRunner
}
