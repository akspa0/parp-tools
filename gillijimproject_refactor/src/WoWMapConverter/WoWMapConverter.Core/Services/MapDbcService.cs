using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWMapConverter.Core.Dbc;

namespace WoWMapConverter.Core.Services;

public class MapDbcService
{
    private readonly Dictionary<string, string> _mapDirectoryLookup = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, string> _mapIdToDirectory = new();
    
    public bool IsLoaded => _mapDirectoryLookup.Count > 0;

    public void Load(IEnumerable<string> searchPaths, NativeMpqService? mpqService)
    {
        if (IsLoaded) return;

        // Try to find Map.dbc
        // 1. On Disk
        string? diskPath = null;
        foreach (var basePath in searchPaths)
        {
            var candidates = new[]
            {
                Path.Combine(basePath, "DBFilesClient", "Map.dbc"),
                Path.Combine(basePath, "DBC", "Map.dbc"),
                Path.Combine(basePath, "Map.dbc")
            };
            
            foreach (var cand in candidates)
            {
                if (File.Exists(cand))
                {
                    diskPath = cand;
                    break;
                }
            }
            if (diskPath != null) break;
        }

        if (diskPath != null)
        {
            Console.WriteLine($"Loading Map.dbc from disk: {diskPath}");
            LoadFromBytes(File.ReadAllBytes(diskPath));
        }
        else if (mpqService != null)
        {
            // 2. In MPQ
            var mpqCandidates = new[]
            {
                "DBFilesClient\\Map.dbc",
                "DBC\\Map.dbc",
                "Map.dbc"
            };
            
            foreach (var key in mpqCandidates)
            {
                if (mpqService.FileExists(key))
                {
                    var data = mpqService.ReadFile(key);
                    if (data != null)
                    {
                        Console.WriteLine($"Loading Map.dbc from MPQ: {key}");
                        LoadFromBytes(data);
                        return;
                    }
                }
            }
        }
        
        if (!IsLoaded)
        {
            Console.WriteLine("[WARN] Map.dbc not found. Minimap export may fail if casing is incorrect.");
        }
    }

    private void LoadFromBytes(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        var dbc = DbcReader.Load(reader);

        // Alpha/Classic Map.dbc Schema:
        // Col 0: ID (uint)
        // Col 1: InternalName / Directory (string)
        // Col 2: MapType (uint)
        // Col 3: Flags (uint) ?
        // Col 4: Real Name (string) ?
        
        // We'll try to be robust. 
        // We need column 0 (ID) and column 1 (Directory).
        
        foreach (var row in dbc.Rows)
        {
            try 
            {
                int rowIndex = dbc.Rows.IndexOf(row);
                int id = dbc.GetInt(rowIndex, 0);
                string directory = dbc.GetString(rowIndex, 1);
                
                if (!string.IsNullOrEmpty(directory))
                {
                    _mapIdToDirectory[id] = directory;
                    // Store strict mapping: "Azeroth" -> "Azeroth"
                    if (!_mapDirectoryLookup.ContainsKey(directory))
                    {
                        _mapDirectoryLookup[directory] = directory;
                    }
                    
                    // Also store case-insensitive lookup if needed, handled by Dictionary comparer
                }
                
                // Also map the localized name if available (Col 4 might be Name)
                if (dbc.Header.FieldCount > 4)
                {
                    string name = dbc.GetString(rowIndex, 4);
                    if (!string.IsNullOrEmpty(name) && !_mapDirectoryLookup.ContainsKey(name))
                    {
                        _mapDirectoryLookup[name] = directory;
                    }
                }
            }
            catch {}
        }
        
        Console.WriteLine($"Loaded {_mapDirectoryLookup.Count} map directory mappings.");
    }

    public string? ResolveDirectory(string mapNameOrId)
    {
        // Try as ID
        if (int.TryParse(mapNameOrId, out int id))
        {
            if (_mapIdToDirectory.TryGetValue(id, out var dir)) return dir;
        }

        // Try as Name (case-insensitive due to dictionary)
        if (_mapDirectoryLookup.TryGetValue(mapNameOrId, out var directMatch))
        {
            return directMatch;
        }

        return null; // Fallback to caller's provided name
    }
}
