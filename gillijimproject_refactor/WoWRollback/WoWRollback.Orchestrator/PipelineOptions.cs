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
    string? MpqPath)
{
    public bool HasLkDbcDirectory => !string.IsNullOrWhiteSpace(LkDbcDirectory);
    public bool HasMpqPath => !string.IsNullOrWhiteSpace(MpqPath);
}
