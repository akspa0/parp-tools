using System.Text.Json.Serialization;

namespace WoWMapConverter.Core.VLM;

public class VlmBatchExportConfig
{
    [JsonPropertyName("clients")]
    public List<VlmClientConfig> Clients { get; set; } = new();

    /// <summary>
    /// Optional root path to a WoWArchive or similar versioned client store.
    /// When set, relative ClientPath values in each VlmClientConfig are resolved
    /// against this root instead of the process working directory.
    /// </summary>
    [JsonPropertyName("archive_root")]
    public string? ArchiveRoot { get; set; }

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

    [JsonPropertyName("minimap_root")]
    public string? MinimapRoot { get; set; }

    [JsonPropertyName("version")]
    public string ClientVersion { get; set; } = "3.3.5"; // e.g. "0.5.3", "3.3.5", "4.x"

    [JsonPropertyName("maps")]
    public List<string> Maps { get; set; } = new();

    [JsonPropertyName("output_root")]
    public string OutputRoot { get; set; } = "";
    
    [JsonPropertyName("generate_depth")]
    public bool GenerateDepth { get; set; } = false;
}
