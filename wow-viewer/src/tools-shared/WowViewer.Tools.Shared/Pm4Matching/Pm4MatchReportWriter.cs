using System.Text.Json;

namespace WowViewer.Tools.Shared.Pm4Matching;

public static class Pm4MatchReportWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public static string ToJson(Pm4MatchRunManifest manifest)
    {
        return JsonSerializer.Serialize(manifest, JsonOptions);
    }

    public static void WriteToFile(Pm4MatchRunManifest manifest, string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(outputPath, ToJson(manifest));
    }

    public static JsonSerializerOptions CreateJsonOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
        };
    }
}
