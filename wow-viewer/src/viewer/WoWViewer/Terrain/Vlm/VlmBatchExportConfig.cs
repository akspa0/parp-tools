using System.Text.Json.Serialization;

namespace WoWViewer.Terrain.Vlm;

public class VlmBatchExportConfig
{
    [JsonPropertyName("clients")]
    public List<VlmClientConfig> Clients { get; set; } = new();

    [JsonPropertyName("run_minimal_curation_after_export")]
    public bool RunMinimalCurationAfterExport { get; set; } = false;

    [JsonPropertyName("minimal_curation_output")]
    public string? MinimalCurationOutput { get; set; }

    [JsonPropertyName("minimal_curation_plan_output")]
    public string? MinimalCurationPlanOutput { get; set; }

    /// <summary>
    /// Optional root path to a WoWArchive or similar versioned client store.
    /// When set, relative ClientPath values in each VlmClientConfig are resolved
    /// against this root instead of the process working directory.
    /// </summary>
    [JsonPropertyName("archive_root")]
    public string? ArchiveRoot { get; set; }

    /// <summary>
    /// Optional mounted WoWArchive root used for resolving archive-backed client paths.
    /// </summary>
    [JsonPropertyName("mount_root")]
    public string? MountRoot { get; set; }

    /// <summary>
    /// Optional mount script used when MountRoot is not currently available.
    /// </summary>
    [JsonPropertyName("mount_script")]
    public string? MountScript { get; set; }

    /// <summary>
    /// Optional staging root where archive-backed clients should be copied locally.
    /// </summary>
    [JsonPropertyName("staging_root")]
    public string? StagingRoot { get; set; }

    [JsonPropertyName("prune_staged_clients")]
    public bool PruneStagedClients { get; set; } = false;

    /// <summary>Path where all dataset output lands when clients omit output_root.</summary>
    [JsonPropertyName("default_output_root")]
    public string? DefaultOutputRoot { get; set; }
}

public class VlmClientConfig
{
    [JsonPropertyName("label")]
    public string? Label { get; set; }

    [JsonPropertyName("client_path")]
    public string ClientPath { get; set; } = "";

    [JsonPropertyName("local_client_path")]
    public string? LocalClientPath { get; set; }

    [JsonPropertyName("archive_client_path")]
    public string? ArchiveClientPath { get; set; }

    [JsonPropertyName("minimap_root")]
    public string? MinimapRoot { get; set; }

    [JsonPropertyName("local_minimap_root")]
    public string? LocalMinimapRoot { get; set; }

    [JsonPropertyName("archive_minimap_root")]
    public string? ArchiveMinimapRoot { get; set; }

    [JsonPropertyName("version")]
    public string ClientVersion { get; set; } = "3.3.5"; // e.g. "0.5.3", "3.3.5", "4.x"

    [JsonPropertyName("maps")]
    public List<string> Maps { get; set; } = new();

    [JsonPropertyName("all_maps")]
    public bool AllMaps { get; set; } = false;

    [JsonPropertyName("output_root")]
    public string OutputRoot { get; set; } = "";
    
    [JsonPropertyName("generate_depth")]
    public bool GenerateDepth { get; set; } = false;

    [JsonPropertyName("tile_limit")]
    public int? TileLimit { get; set; }

    [JsonPropertyName("interesting_only")]
    public bool InterestingOnly { get; set; } = false;

    [JsonPropertyName("interesting_min_score")]
    public int InterestingMinScore { get; set; } = 1;

    [JsonPropertyName("skip_derived_assets")]
    public bool SkipDerivedAssets { get; set; } = false;

    [JsonPropertyName("finalize_derived_assets_after_export")]
    public bool FinalizeDerivedAssetsAfterExport { get; set; } = false;
}
