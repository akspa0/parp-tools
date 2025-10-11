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
    string? MinimapRoot = null,
    string? AdtRoot = null,
    string? ViewerLabel = null,
    string? ViewerAssetsPath = null,
    string? MpqRoot = null,
    IReadOnlyList<string>? MpqLocales = null,
    bool MpqOnly = false,
    string? AdtOverlayRoot = null)
{
    public bool HasLkDbcDirectory => !string.IsNullOrWhiteSpace(LkDbcDirectory);
}
