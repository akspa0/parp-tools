using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace WoWRollback.Core.Services.Assets
{
    public sealed class ListfileSnapshot
    {
        [JsonPropertyName("source")] public string Source { get; set; } = "unknown"; // e.g., mpq-scan, casc, community
        [JsonPropertyName("clientRoot")] public string? ClientRoot { get; set; }
        [JsonPropertyName("version")] public string? Version { get; set; }
        [JsonPropertyName("generatedAt")] public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
        [JsonPropertyName("entries")] public List<Entry> Entries { get; set; } = new();

        public sealed class Entry
        {
            [JsonPropertyName("path")] public string Path { get; set; } = string.Empty;
            [JsonPropertyName("fdid")] public uint? FileDataId { get; set; }
        }
    }
}
