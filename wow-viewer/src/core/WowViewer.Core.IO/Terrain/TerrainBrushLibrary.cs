using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// In-memory and serializable catalog of discovered or curated terrain brush pastes.
/// Supports tag-based indexing, fast category lookups, and JSON export/import.
/// </summary>
public sealed class TerrainBrushLibrary
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly Dictionary<string, TerrainBrushPaste> _pastesById = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<TerrainBrushPaste>> _pastesByCategory = new(StringComparer.OrdinalIgnoreCase);

    public IReadOnlyList<TerrainBrushPaste> AllPastes => _pastesById.Values.ToList();
    public int Count => _pastesById.Count;

    public void Add(TerrainBrushPaste paste)
    {
        ArgumentNullException.ThrowIfNull(paste);
        if (string.IsNullOrWhiteSpace(paste.Id))
            throw new ArgumentException("Paste ID cannot be empty.", nameof(paste));

        _pastesById[paste.Id] = paste;

        string category = string.IsNullOrWhiteSpace(paste.Category) ? "General" : paste.Category;
        if (!_pastesByCategory.TryGetValue(category, out List<TerrainBrushPaste>? list))
        {
            list = [];
            _pastesByCategory[category] = list;
        }

        list.RemoveAll(p => string.Equals(p.Id, paste.Id, StringComparison.OrdinalIgnoreCase));
        list.Add(paste);
    }

    public bool TryGetPaste(string id, out TerrainBrushPaste? paste)
    {
        return _pastesById.TryGetValue(id, out paste);
    }

    public IReadOnlyList<TerrainBrushPaste> GetByCategory(string category)
    {
        if (_pastesByCategory.TryGetValue(category, out List<TerrainBrushPaste>? list))
            return list;
        return [];
    }

    public IReadOnlyList<string> GetCategories()
    {
        return _pastesByCategory.Keys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public IReadOnlyList<TerrainBrushPaste> FindByTag(string tag)
    {
        if (string.IsNullOrWhiteSpace(tag))
            return [];

        return _pastesById.Values
            .Where(p => p.Tags.Any(t => string.Equals(t, tag, StringComparison.OrdinalIgnoreCase)))
            .ToList();
    }

    public IReadOnlyList<TerrainBrushPaste> Search(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return AllPastes;

        string[] tokens = query.Split([' ', ',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        return _pastesById.Values
            .Where(p => tokens.All(tok =>
                p.Name.Contains(tok, StringComparison.OrdinalIgnoreCase) ||
                p.Category.Contains(tok, StringComparison.OrdinalIgnoreCase) ||
                p.Tags.Any(t => t.Contains(tok, StringComparison.OrdinalIgnoreCase)) ||
                p.SourceMap.Contains(tok, StringComparison.OrdinalIgnoreCase)))
            .ToList();
    }

    public void SaveToJson(string filePath)
    {
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        string json = JsonSerializer.Serialize(_pastesById.Values.ToList(), JsonOpts);
        File.WriteAllText(filePath, json);
    }

    public void SaveToJson(Stream stream)
    {
        JsonSerializer.Serialize(stream, _pastesById.Values.ToList(), JsonOpts);
    }

    public static TerrainBrushLibrary LoadFromJson(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Terrain brush library file not found.", filePath);

        using FileStream fs = File.OpenRead(filePath);
        return LoadFromJson(fs);
    }

    public static TerrainBrushLibrary LoadFromJson(Stream stream)
    {
        List<TerrainBrushPaste>? list = JsonSerializer.Deserialize<List<TerrainBrushPaste>>(stream, JsonOpts);
        var library = new TerrainBrushLibrary();
        if (list != null)
        {
            foreach (TerrainBrushPaste paste in list)
                library.Add(paste);
        }
        return library;
    }
}
