using WoWMapConverter.Core.Dbc;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// service to decode GroundEffectTexture.dbc and GroundEffectDoodad.dbc
/// to determine which models (grass, stones) appear on a texture layer.
/// </summary>
public class GroundEffectService
{
    private readonly Dictionary<uint, List<uint>> _textureToDoodads = new();
    private readonly Dictionary<uint, string> _doodadModels = new();
    private bool _loaded = false;

    public void Load(IEnumerable<string> searchPaths, MpqArchiveService? mpqService = null)
    {
        if (_loaded) return;

        string? dbcDir = null;
        
        // 1. Try disk paths
        foreach (var basePath in searchPaths)
        {
             var candidates = new[]
             {
                 Path.Combine(basePath, "DBFilesClient"),
                 Path.Combine(basePath, "DBC"),
                 basePath
             };

             foreach (var dir in candidates)
             {
                if (Directory.Exists(dir) && 
                    (File.Exists(Path.Combine(dir, "GroundEffectTexture.dbc")) || 
                     File.Exists(Path.Combine(dir, "GroundEffectDoodad.dbc"))))
                {
                    dbcDir = dir;
                    break;
                }
             }
             if (dbcDir != null) break;
        }

        try
        {
            if (dbcDir != null)
            {
                // LOAD FROM DISK
                LoadDoodads(Path.Combine(dbcDir, "GroundEffectDoodad.dbc"));
                LoadTextures(Path.Combine(dbcDir, "GroundEffectTexture.dbc"));
                _loaded = true;
                Console.WriteLine($"Loaded GroundEffects from Disk: {_textureToDoodads.Count} textures, {_doodadModels.Count} models.");
            }
            else if (mpqService != null)
            {
                // LOAD FROM MPQ
                // Try to find them in MPQ
                // Usually DBFilesClient\GroundEffectDoodad.dbc or DBC\GroundEffectDoodad.dbc
                
                var candidates = new[] { "DBFilesClient", "DBC", "" };
                byte[]? doodadBytes = null;
                byte[]? textureBytes = null;

                foreach (var prefix in candidates)
                {
                    var p1 = Path.Combine(prefix, "GroundEffectDoodad.dbc").Replace("/", "\\");
                    if (mpqService.FileExists(p1))
                    {
                         doodadBytes = mpqService.ReadFile(p1);
                         textureBytes = mpqService.ReadFile(Path.Combine(prefix, "GroundEffectTexture.dbc").Replace("/", "\\"));
                         break;
                    }
                }

                if (doodadBytes != null && textureBytes != null)
                {
                    LoadDoodadsFromBytes(doodadBytes);
                    LoadTexturesFromBytes(textureBytes);
                    _loaded = true;
                    Console.WriteLine($"Loaded GroundEffects from Archive: {_textureToDoodads.Count} textures, {_doodadModels.Count} models.");
                }
                else
                {
                    Console.WriteLine("Could not find DBC files in archives.");
                }
            }
            else
            {
                Console.WriteLine("Could not find DBC directory containing GroundEffect files.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load GroundEffects: {ex.Message}");
        }
    }
    
    private void LoadDoodadsFromBytes(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        var dbc = DbcReader.Load(reader);
        ProcessDoodadRows(dbc);
    }

    private void LoadTexturesFromBytes(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        var dbc = DbcReader.Load(reader);
        ProcessTextureRows(dbc);
    }

    private void LoadDoodads(string path)
    {
        if (!File.Exists(path)) return;
        var dbc = DbcReader.Load(path);
        ProcessDoodadRows(dbc);
    }
    
    private void LoadTextures(string path)
    {
        if (!File.Exists(path)) return;
        var dbc = DbcReader.Load(path);
        ProcessTextureRows(dbc);
    }

    private void ProcessDoodadRows(DbcReader dbc)
    {
        // Alpha 0.5.3 schema: ID(0), DoodadIdTag(1), Doodadpath(2)
        // Later versions: ID(0), Doodadpath(1), Flags(2)...
        // We try field 2 first (Alpha), fall back to field 1 (later versions)
        foreach (var row in dbc.Rows)
        {
            int rowIndex = dbc.Rows.IndexOf(row);
            uint id = dbc.GetUInt(rowIndex, 0);
            
            // Try field 2 first (Alpha schema has path at field 2)
            string model = dbc.GetString(rowIndex, 2);
            if (string.IsNullOrEmpty(model) || !model.Contains('.'))
            {
                // Fall back to field 1 (later client schema)
                model = dbc.GetString(rowIndex, 1);
            }
            
            if (!string.IsNullOrEmpty(model) && model.Length > 4) 
                _doodadModels[id] = model;
        }
    }

    private void ProcessTextureRows(DbcReader dbc)
    {
         foreach (var row in dbc.Rows)
        {
            int rowIndex = dbc.Rows.IndexOf(row);
            uint id = dbc.GetUInt(rowIndex, 0);
            
            var doodads = new List<uint>();
            for (int f = 1; f <= 4; f++)
            {
                if (f >= dbc.Header.FieldCount) break;
                
                uint doodadId = dbc.GetUInt(rowIndex, f);
                if (doodadId > 0 && _doodadModels.ContainsKey(doodadId))
                {
                    doodads.Add(doodadId);
                }
            }

            if (doodads.Count > 0)
            {
                _textureToDoodads[id] = doodads;
            }
        }
    }

    public string[]? GetDoodadsEffect(uint effectId)
    {
        if (!_loaded) return null;
        
        if (_textureToDoodads.TryGetValue(effectId, out var doodadIds))
        {
            return doodadIds.Select(id => _doodadModels[id]).ToArray();
        }

        return null;
    }
}
