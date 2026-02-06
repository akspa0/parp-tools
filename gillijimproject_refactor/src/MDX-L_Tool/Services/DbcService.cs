using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MdxLTool.Services;

/// <summary>
/// Service for loading and querying WoW DBC files (via CSV exports).
/// </summary>
public class DbcService
{
    private readonly string _dbcDir;
    private Dictionary<string, int> _modelPathToId = new(StringComparer.OrdinalIgnoreCase);
    private Dictionary<int, List<DisplayInfo>> _modelIdToDisplays = new();
    private Dictionary<int, string> _extraIdToBake = new();

    public DbcService(string dbcDir)
    {
        _dbcDir = dbcDir;
    }

    public void Initialize()
    {
        LoadModelData();
        LoadDisplayInfo();
        LoadDisplayInfoExtra();
    }

    private void LoadModelData()
    {
        var path = Path.Combine(_dbcDir, "CreatureModelData.csv");
        if (!File.Exists(path)) return;

        var lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 3) continue;

            if (int.TryParse(parts[0], out int id))
            {
                var modelPath = parts[2].Trim();
                _modelPathToId[modelPath] = id;
                // Also map by index if the table uses row-mapping (common in 0.5.3)
                // We'll use ID as the primary key for DisplayInfo lookup
            }
        }
    }

    private void LoadDisplayInfo()
    {
        var path = Path.Combine(_dbcDir, "CreatureDisplayInfo.csv");
        if (!File.Exists(path)) return;

        var lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 7) continue;

            if (int.TryParse(parts[1], out int modelId))
            {
                var display = new DisplayInfo
                {
                    Id = int.Parse(parts[0]),
                    ModelId = modelId,
                    ExtraId = int.Parse(parts[3]),
                    Variations = parts[6].Split('|', StringSplitOptions.RemoveEmptyEntries).ToList()
                };

                if (!_modelIdToDisplays.ContainsKey(modelId))
                    _modelIdToDisplays[modelId] = new List<DisplayInfo>();
                
                _modelIdToDisplays[modelId].Add(display);
            }
        }
    }

    private void LoadDisplayInfoExtra()
    {
        var path = Path.Combine(_dbcDir, "CreatureDisplayInfoExtra.csv");
        if (!File.Exists(path)) return;

        var lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 10) continue;

            if (int.TryParse(parts[0], out int id))
            {
                _extraIdToBake[id] = parts[9].Trim();
            }
        }
    }

    public List<string> GetVariations(string modelPath)
    {
        // Normalize path
        modelPath = modelPath.Replace("\\\\", "\\").Replace("/", "\\");
        
        // Try to find the model ID
        if (!_modelPathToId.TryGetValue(modelPath, out int modelId))
        {
            // Try relative path or name search
            var name = Path.GetFileName(modelPath);
            var entry = _modelPathToId.FirstOrDefault(x => x.Key.EndsWith(name, StringComparison.OrdinalIgnoreCase));
            if (entry.Key != null) modelId = entry.Value;
            else return new List<string>();
        }

        if (_modelIdToDisplays.TryGetValue(modelId, out var displays))
        {
            // Return all unique variations for this model
            var allVars = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var d in displays)
            {
                foreach (var v in d.Variations) allVars.Add(v);
                
                // If it has a baked texture, include it
                if (d.ExtraId > 0 && _extraIdToBake.TryGetValue(d.ExtraId, out var bake))
                {
                    allVars.Add(bake);
                }
            }
            return allVars.ToList();
        }

        return new List<string>();
    }

    private class DisplayInfo
    {
        public int Id { get; set; }
        public int ModelId { get; set; }
        public int ExtraId { get; set; }
        public List<string> Variations { get; set; } = new();
    }
}
