using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWRollback.Core.Services.Archive;
using WoWRollback.LkToAlphaModule.Readers;
using WoWRollback.LkToAlphaModule.Models;
using WoWRollback.PM4Module;

namespace WoWRollback.Gui.Services;

/// <summary>
/// Service to export a VLM-compatible dataset for a map.
/// Uses PrioritizedArchiveSource from Core for robust MPQ access,
/// and MpqAdtExtractor from PM4Module for ADT byte-level parsing.
/// </summary>
public class VlmExporter : IDisposable
{
    private readonly string _cacheRoot;
    private PrioritizedArchiveSource? _source;
    private MpqAdtExtractor? _parser;

    public VlmExporter(string cacheRoot)
    {
        _cacheRoot = cacheRoot;
    }

    public async Task ExportMap(string dataRoot, string map, string vlmOutputDir, Action<string> logger)
    {
        await Task.Run(() => ExportMapInternal(dataRoot, map, vlmOutputDir, logger));
    }

    private void ExportMapInternal(string dataRoot, string map, string vlmOutputDir, Action<string> logger)
    {
        // 1. Locate MPQs using ArchiveLocator
        var mpqPaths = ArchiveLocator.LocateMpqs(dataRoot);
        logger($"[VLM] Found {mpqPaths.Count} MPQs in root");

        if (mpqPaths.Count == 0)
        {
            logger("[VLM] ERROR: No MPQs found.");
            return;
        }

        // 2. Initialize Archive Source
        _source = new PrioritizedArchiveSource(dataRoot, mpqPaths);

        // 3. Find the map (WDT file)
        string? mapWdtPath = null;
        var mapLower = map.ToLowerInvariant();
        var permutations = new[]
        {
            $"World\\Maps\\{map}\\{map}.wdt",
            $"world\\maps\\{mapLower}\\{mapLower}.wdt",
            $"World/Maps/{map}/{map}.wdt",
            $"world/maps/{mapLower}/{mapLower}.wdt"
        };
        
        foreach (var p in permutations)
        {
            if (_source.FileExists(p))
            {
                mapWdtPath = p;
                break;
            }
        }

        if (mapWdtPath == null)
        {
            logger($"[VLM] ERROR: Could not find WDT for map '{map}' in any MPQ.");
            return;
        }

        // 4. Determine Alpha vs LK mode
        bool isAlpha = true;
        var adtCheckPaths = new[] {
            mapWdtPath.Replace(".wdt", "_32_32.adt", StringComparison.OrdinalIgnoreCase),
            $"World\\Maps\\{map}\\{map}_32_32.adt",
            $"world\\maps\\{mapLower}\\{mapLower}_32_32.adt"
        };
        foreach (var p in adtCheckPaths)
        {
            if (_source.FileExists(p))
            {
                isAlpha = false;
                break;
            }
        }

        logger($"[VLM] Found map '{map}' (Mode: {(isAlpha ? "Alpha/WDT" : "LK/ADT")})");

        var mapVlmDir = Path.Combine(vlmOutputDir, map);
        Directory.CreateDirectory(mapVlmDir);

        // Initialize parser (just needs first MPQ path for constructor, we use _source for file access)
        _parser = new MpqAdtExtractor(mpqPaths[0]);

        var dataset = new VlmDataset
        {
            Map = map,
            Tiles = new List<VlmTile>()
        };

        try
        {
            if (isAlpha)
            {
                ExportAlphaMap(_source, map, mapWdtPath, dataset, mapVlmDir, logger);
            }
            else
            {
                ExportLkMap(_source, _parser, map, dataset, mapVlmDir, logger);
            }

            // Write metadata
            var jsonOpts = new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
            var jsonPath = Path.Combine(mapVlmDir, "dataset.json");
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(dataset, jsonOpts));

            logger($"[VLM] Export complete for {map}. Tiles: {dataset.Tiles.Count}. Saved to {mapVlmDir}");
        }
        catch (Exception ex)
        {
            logger($"[VLM] CRITICAL ERROR during export: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private void ExportAlphaMap(IArchiveSource source, string map, string wdtPath, VlmDataset dataset, string outDir, Action<string> logger)
    {
        logger($"[VLM] Processing Alpha Map: {map}");
        
        // Read WDT bytes
        byte[] wdtBytes;
        using (var s = source.OpenFile(wdtPath))
        using (var ms = new MemoryStream())
        {
            s.CopyTo(ms);
            wdtBytes = ms.ToArray();
        }

        // Use AlphaWdtReader (static method, requires file path)
        var tempWdt = Path.GetTempFileName();
        File.WriteAllBytes(tempWdt, wdtBytes);

        try
        {
            // Static call
            var wdt = AlphaWdtReader.Read(tempWdt);

            if (wdt.Tiles == null || wdt.Tiles.Count == 0)
            {
                logger("[VLM] WDT has no tiles.");
                return;
            }

            int tileCount = 0;
            // Each AlphaTile has Index (0-4095), corresponding to x*64+y or similar
            foreach (var alphaTile in wdt.Tiles)
            {
                int idx = alphaTile.Index;
                int y = idx / 64;
                int x = idx % 64;
                
                // Extract terrain from this tile's FirstMcnk if available
                var tileData = ExtractAlphaTileData(wdtBytes, x, y, alphaTile);
                if (tileData != null)
                {
                    dataset.Tiles.Add(tileData);
                    tileCount++;
                }
            }
            logger($"[VLM] Processed {tileCount} Alpha tiles.");
        }
        finally
        {
            if (File.Exists(tempWdt)) File.Delete(tempWdt);
        }
    }

    private VlmTile? ExtractAlphaTileData(byte[] wdtBytes, int x, int y, AlphaTile alphaTile)
    {
        var tile = new VlmTile { X = x, Y = y };
        var terrain = new VlmTerrain { BaseHeight = 0, Chunks = new List<VlmChunk>() };

        // Alpha has FirstMcnk info, we can use the reader's output or scan bytes
        // For simplicity, use the FirstMcnk header if available
        if (alphaTile.FirstMcnk == null) return null;

        // We would need to manually extract MCVT from the offset
        // This is simplified; real implementation would require parsing subchunks
        // For now, return tile with minimal data to show tiles exist
        tile.Terrain = terrain;
        return tile;
    }

    private void ExportLkMap(IArchiveSource source, MpqAdtExtractor parser, string map, VlmDataset dataset, string outDir, Action<string> logger)
    {
        logger($"[VLM] Processing LK Map: {map}");
        int count = 0;
        var mapLower = map.ToLowerInvariant();

        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
                string? tilePath = null;
                bool isSplitTarget = false;
                
                var candidates = new[] {
                    (Path: $"World\\Maps\\{map}\\{map}_{x}_{y}.adt", Split: false),
                    (Path: $"world\\maps\\{mapLower}\\{mapLower}_{x}_{y}.adt", Split: false),
                    (Path: $"World\\Maps\\{map}\\{map}_{x}_{y}_obj0.adt", Split: true),
                    (Path: $"world\\maps\\{mapLower}\\{mapLower}_{x}_{y}_obj0.adt", Split: true),
                };

                foreach (var c in candidates)
                {
                    if (source.FileExists(c.Path))
                    {
                        tilePath = c.Path;
                        isSplitTarget = c.Split;
                        break;
                    }
                }

                if (tilePath == null) continue;

                try
                {
                    byte[] adtBytes;
                    using (var s = source.OpenFile(tilePath))
                    using (var ms = new MemoryStream())
                    {
                        s.CopyTo(ms);
                        adtBytes = ms.ToArray();
                    }

                    // Extract Terrain
                    var terrainData = parser.ExtractTerrainFromBytes(adtBytes, map, x, y);
                    
                    if (isSplitTarget)
                    {
                        var rootPath = tilePath.Replace("_obj0", "", StringComparison.OrdinalIgnoreCase);
                        if (source.FileExists(rootPath))
                        {
                            using var sRoot = source.OpenFile(rootPath);
                            using var msRoot = new MemoryStream();
                            sRoot.CopyTo(msRoot);
                            var rootBytes = msRoot.ToArray();
                            if (terrainData == null || terrainData.Chunks.Count == 0)
                                terrainData = parser.ExtractTerrainFromBytes(rootBytes, map, x, y);
                        }
                    }

                    // Extract Placements
                    var placements = parser.ExtractPlacementsFromBytes(adtBytes, map, x, y, isSplitTarget);

                    if ((terrainData != null && terrainData.Chunks.Count > 0) || 
                        (placements != null && (placements.M2Placements.Count > 0 || placements.WmoPlacements.Count > 0)))
                    {
                        var tile = new VlmTile
                        {
                            X = x, Y = y,
                            Terrain = terrainData != null ? MapTerrain(terrainData) : null,
                            Placements = placements != null ? MapPlacements(placements) : null
                        };
                        dataset.Tiles.Add(tile);
                        count++;
                    }
                }
                catch
                {
                    // Silent skip failed tiles
                }
            }
        }
        logger($"[VLM] Processed {count} LK tiles.");
    }
    
    private VlmTerrain? MapTerrain(TileTerrainData data)
    {
         if (data.Chunks == null || data.Chunks.Count == 0) return null;
         var t = new VlmTerrain { BaseHeight = 0, Chunks = new List<VlmChunk>() };
         foreach(var c in data.Chunks) {
             t.Chunks.Add(new VlmChunk { Unk = c.Idx, Heights = c.Heights });
         }
         return t;
    }

    private VlmPlacementData? MapPlacements(AdtPlacementData data)
    {
        var res = new VlmPlacementData { M2 = new List<VlmObject>(), WMO = new List<VlmObject>() };
        if (data.M2Placements != null)
            foreach(var m in data.M2Placements) res.M2.Add(new VlmObject { 
                UniqueId = m.UniqueId, Pos = new[]{m.PositionX, m.PositionY, m.PositionZ}, 
                Rot = new[]{m.RotationX, m.RotationY, m.RotationZ}, Scale = m.Scale, File = m.Path ?? ""
            });
            
        if (data.WmoPlacements != null)
            foreach(var w in data.WmoPlacements) res.WMO.Add(new VlmObject { 
                UniqueId = w.UniqueId, Pos = new[]{w.PositionX, w.PositionY, w.PositionZ}, 
                Rot = new[]{w.RotationX, w.RotationY, w.RotationZ}, Scale = w.Scale, File = w.Path ?? ""
            });
            
        return res;
    }

    public void Dispose()
    {
        _parser?.Dispose();
        _source?.Dispose();
    }
}

// Models
public class VlmDataset { public string Map {get;set;} = ""; public List<VlmTile> Tiles {get;set;} = new(); }
public class VlmTile { 
    public int X {get;set;} 
    public int Y {get;set;} 
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public VlmTerrain? Terrain {get;set;} 
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public VlmPlacementData? Placements {get;set;} 
}
public class VlmTerrain { public float BaseHeight {get;set;} public List<VlmChunk> Chunks {get;set;} = new(); }
public class VlmChunk { public int Unk {get;set;} public List<float>? Heights {get;set;} }
public class VlmPlacementData { public List<VlmObject> M2 {get;set;} = new(); public List<VlmObject> WMO {get;set;} = new(); }
public class VlmObject { 
    public uint UniqueId {get;set;} 
    public float[] Pos {get;set;} = Array.Empty<float>(); 
    public float[] Rot {get;set;} = Array.Empty<float>(); 
    public float Scale {get;set;} 
    public string File {get;set;} = ""; 
}
