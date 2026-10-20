using System.Text.Json;

namespace WoWViewer;

/// <summary>
/// Human-curated PM4 object → model matches. One entry per selected PM4 object part. The store is a
/// plain JSON file under <c>output/pm4_object_matches.json</c> so it can be reviewed, diffed and
/// merged without the viewer.
/// </summary>
public sealed class Pm4ObjectMatchStore
{
    private readonly string _storePath;

    public Pm4ObjectMatchStore(string baseOutputDir)
    {
        string outputDir = Path.Combine(baseOutputDir, "..", "..", "..", "..", "output");
        _storePath = Path.GetFullPath(Path.Combine(outputDir, "pm4_object_matches.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(_storePath)!);
    }

    public Dictionary<string, Pm4ObjectMatchEntry> Load()
    {
        try
        {
            if (!File.Exists(_storePath))
                return new Dictionary<string, Pm4ObjectMatchEntry>(StringComparer.OrdinalIgnoreCase);

            string json = File.ReadAllText(_storePath);
            var entries = JsonSerializer.Deserialize<Dictionary<string, Pm4ObjectMatchEntry>>(json);
            return entries ?? new Dictionary<string, Pm4ObjectMatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
        catch
        {
            return new Dictionary<string, Pm4ObjectMatchEntry>(StringComparer.OrdinalIgnoreCase);
        }
    }

    public void Save(Dictionary<string, Pm4ObjectMatchEntry> entries)
    {
        try
        {
            string json = JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_storePath, json);
        }
        catch
        {
        }
    }

    public string StorePath => _storePath;

    public static string BuildKey(string mapName, int tileX, int tileY, uint ck24, int objectPartId)
        => $"{mapName.ToLowerInvariant()}|{tileX}|{tileY}|{ck24:X6}|{objectPartId}";
}

public sealed class Pm4ObjectMatchEntry
{
    public string MapName { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public uint Ck24 { get; set; }
    public int ObjectPartId { get; set; }
    public string ModelPath { get; set; } = "";
    public string ModelName { get; set; } = "";
    public string Source { get; set; } = "manual";
    public string Notes { get; set; } = "";
    public string SavedAtUtc { get; set; } = "";
}
