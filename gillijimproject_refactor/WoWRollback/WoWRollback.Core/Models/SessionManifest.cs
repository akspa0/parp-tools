namespace WoWRollback.Core.Models;

/// <summary>
/// Manifest document describing a complete pipeline session.
/// Written as manifest.json in the session root directory.
/// </summary>
public sealed record SessionManifest
{
    public required string SessionId { get; init; }
    public required DateTime StartTime { get; init; }
    public DateTime? EndTime { get; init; }
    public int DurationSeconds { get; init; }
    public required PipelineOptionsSnapshot Options { get; init; }
    public required ManifestStages Stages { get; init; }
    public required ManifestOutputs Outputs { get; init; }
}

/// <summary>
/// Snapshot of pipeline options at execution time.
/// </summary>
public sealed record PipelineOptionsSnapshot
{
    public required string[] Maps { get; init; }
    public required string[] Versions { get; init; }
    public required string AlphaRoot { get; init; }
    public required string OutputRoot { get; init; }
    public string? DbdDirectory { get; init; }
    public string? LkDbcDirectory { get; init; }
}

/// <summary>
/// Status and metadata for each pipeline stage.
/// </summary>
public sealed record ManifestStages
{
    public required ManifestStageInfo Dbc { get; init; }
    public required ManifestStageInfo Adt { get; init; }
    public required ManifestStageInfo Viewer { get; init; }
}

/// <summary>
/// Information about a single pipeline stage execution.
/// </summary>
public sealed record ManifestStageInfo
{
    public required bool Success { get; init; }
    public required int DurationSeconds { get; init; }
    public string? ErrorMessage { get; init; }
    public Dictionary<string, object>? Metadata { get; init; }
}

/// <summary>
/// Paths to key output artifacts from the pipeline.
/// </summary>
public sealed record ManifestOutputs
{
    public required string DbcDirectory { get; init; }
    public required string CrosswalkDirectory { get; init; }
    public required string AdtDirectory { get; init; }
    public required string AnalysisDirectory { get; init; }
    public required string ViewerDirectory { get; init; }
    public required string LogsDirectory { get; init; }
}
