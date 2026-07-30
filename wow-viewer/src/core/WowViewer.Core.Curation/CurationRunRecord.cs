using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Core.Curation;

/// <summary>
/// Small JSON provenance record, schema <c>v50-curation-run-v1</c> (data-model.md "Curation Run
/// Record") -- mirrors the existing <c>v50-model-stage-run-v1</c> convention used throughout the
/// Python training tooling, but describes a classification pass, not a training run. This is
/// provenance/summary only; row-level data lives in the two Parquet tables
/// <see cref="CurationManifestWriter"/> writes alongside it.
/// </summary>
public sealed class CurationRunRecord
{
    public const string CurrentSchema = "v50-curation-run-v1";

    [JsonPropertyName("schema")]
    public string Schema { get; init; } = CurrentSchema;

    [JsonPropertyName("curation_run_id")]
    public required string CurationRunId { get; init; }

    [JsonPropertyName("store_path")]
    public required string StorePath { get; init; }

    [JsonPropertyName("build_fingerprint")]
    public required string BuildFingerprint { get; init; }

    [JsonPropertyName("checks_run")]
    public required IReadOnlyList<string> ChecksRun { get; init; }

    [JsonPropertyName("tile_count")]
    public required int TileCount { get; init; }

    [JsonPropertyName("bucket_counts")]
    public required IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> BucketCounts { get; init; }

    [JsonPropertyName("finding_counts")]
    public required IReadOnlyDictionary<string, int> FindingCounts { get; init; }

    [JsonPropertyName("tool_version")]
    public required string ToolVersion { get; init; }

    [JsonPropertyName("created_at")]
    public required DateTimeOffset CreatedAt { get; init; }

    /// <summary>
    /// The SC-006 hard gate expressed as code: a curation run is not permitted to report success if
    /// the number of tiles it classified does not exactly match the source store's own row count.
    /// This is enforced here, not left to a caller to remember.
    /// </summary>
    public void Verify(int expectedTileCount)
    {
        if (TileCount != expectedTileCount)
        {
            throw new InvalidOperationException(
                $"Curation run '{CurationRunId}' classified {TileCount} tiles but the source store " +
                $"'{StorePath}' has {expectedTileCount} rows in its index -- full coverage is a hard " +
                "requirement (spec FR-008/SC-006), not a best-effort target.");
        }
    }

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
    };

    public string ToJson() => JsonSerializer.Serialize(this, SerializerOptions);

    public static CurationRunRecord FromJson(string json) =>
        JsonSerializer.Deserialize<CurationRunRecord>(json)
        ?? throw new InvalidOperationException("Failed to deserialize CurationRunRecord: null result.");

    /// <summary>
    /// Deterministic-enough run identifier: build fingerprint + store directory name + a
    /// millisecond-precision UTC timestamp, so two runs against the same store never collide
    /// on-disk (data-model.md path convention: each run gets its own directory, never overwritten).
    /// </summary>
    public static string GenerateRunId(string buildFingerprint, string storePath, DateTimeOffset now)
    {
        string storeName = Path.GetFileName(storePath.TrimEnd('\\', '/'));
        return $"{buildFingerprint}-{storeName}-{now:yyyyMMddTHHmmssfffZ}";
    }
}
