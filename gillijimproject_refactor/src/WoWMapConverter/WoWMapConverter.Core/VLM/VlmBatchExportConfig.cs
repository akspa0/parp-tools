using System.Text.Json.Serialization;

namespace WoWMapConverter.Core.VLM;

public class VlmBatchExportConfig
{
    [JsonPropertyName("clients")]
    public List<VlmClientConfig> Clients { get; set; } = new();
}

public class VlmClientConfig
{
    [JsonPropertyName("client_path")]
    public string ClientPath { get; set; } = "";

    [JsonPropertyName("version")]
    public string ClientVersion { get; set; } = "3.3.5"; // e.g. "0.5.3", "3.3.5", "4.x"

    [JsonPropertyName("maps")]
    public List<string> Maps { get; set; } = new();

    [JsonPropertyName("output_root")]
    public string OutputRoot { get; set; } = "";
    
    [JsonPropertyName("generate_depth")]
    public bool GenerateDepth { get; set; } = false;
}
