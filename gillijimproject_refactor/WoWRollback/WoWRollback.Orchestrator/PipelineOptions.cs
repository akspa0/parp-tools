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
    bool Serve,
    int Port,
    bool Verbose)
{
    public bool HasLkDbcDirectory => !string.IsNullOrWhiteSpace(LkDbcDirectory);
}
