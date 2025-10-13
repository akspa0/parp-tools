using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Services.Archive;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.TerrainObject.Zero;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts placements (M2/WMO/MDX) from ADT files inside MPQs using Warcraft.NET parser.
/// </summary>
public sealed class AdtMpqChunkPlacementsExtractor
{
    public PlacementsExtractionResult ExtractFromArchive(IArchiveSource source, string mapName, string outputCsvPath)
    {
        try
        {
            var tiles = EnumerateTiles(source, mapName);
            if (tiles.Count == 0)
            {
                return new PlacementsExtractionResult(false, 0, 0, 0, $"No ADT tiles found in archive for map: {mapName}");
            }

            var csv = new StringBuilder();
            csv.AppendLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set,source_file");

            int m2Count = 0;
            int mdxCount = 0;
            int wmoCount = 0;
            int tilesProcessed = 0;

            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Processing {tiles.Count} tiles from archive...");

            int tileIndex = 0;
            foreach (var (tileX, tileY, _, virtualPath) in tiles)
            {
                tileIndex++;
                if (tileIndex % 10 == 0 || tileIndex == 1)
                {
                    Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Progress: {tileIndex}/{tiles.Count} tiles ({m2Count} M2, {mdxCount} MDX, {wmoCount} WMO so far)");
                }
                
                try
                {
                    var rows = ExtractPlacementsFromAdt(source, virtualPath, mapName, tileX, tileY);
                    foreach (var p in rows)
                    {
                        csv.AppendLine(FormatPlacementCsv(p, Path.GetFileName(virtualPath)));
                        if (p.Type == "M2") m2Count++;
                        else if (p.Type == "MDX") mdxCount++;
                        else wmoCount++;
                    }
                    tilesProcessed++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Warning: tile ({tileX},{tileY}) failed: {ex.Message}");
                }
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputCsvPath)!);
            File.WriteAllText(outputCsvPath, csv.ToString());

            var totalModels = m2Count + mdxCount;
            var modelSummary = mdxCount > 0 ? $"{mdxCount} MDX" : (m2Count > 0 ? $"{m2Count} M2" : "0 models");
            return new PlacementsExtractionResult(true, tilesProcessed, totalModels, wmoCount, 
                $"Extracted {modelSummary} and {wmoCount} WMO placements from {tilesProcessed} tiles");
        }
        catch (Exception ex)
        {
            return new PlacementsExtractionResult(false, 0, 0, 0, $"Archive extraction failed: {ex.Message}");
        }
    }

    private static List<(int X, int Y, bool IsCataSplit, string VirtualPath)> EnumerateTiles(IArchiveSource src, string map)
    {
        var tiles = new List<(int, int, bool, string)>();
        var basePath = $"world/maps/{map}";

        Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Scanning for {map} ADT tiles (0-63 grid)...");
        
        var chosen = new Dictionary<(int X,int Y), (bool IsCata, string Path)>();
        int foundCount = 0;
        
        // Parallel scan for better performance
        var lockObj = new object();
        System.Threading.Tasks.Parallel.For(0, 64, x =>
        {
            for (int y = 0; y < 64; y++)
            {
                var obj0 = $"{basePath}/{map}_{x}_{y}_obj0.adt";
                if (src.FileExists(obj0))
                {
                    lock (lockObj)
                    {
                        chosen[(x, y)] = (true, obj0);
                        foundCount++;
                    }
                    continue;
                }
                
                var pre = $"{basePath}/{map}_{x}_{y}.adt";
                if (src.FileExists(pre))
                {
                    lock (lockObj)
                    {
                        chosen[(x, y)] = (false, pre);
                        foundCount++;
                    }
                }
            }
            
            // Progress every 8 rows
            if (x % 8 == 0)
            {
                lock (lockObj)
                {
                    Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Scanned row {x}/64... ({foundCount} tiles found)");
                }
            }
        });
        
        Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Scan complete: found {foundCount} tiles");
        
        foreach (var kv in chosen.OrderBy(k => k.Key.X).ThenBy(k => k.Key.Y))
        {
            tiles.Add((kv.Key.X, kv.Key.Y, kv.Value.IsCata, kv.Value.Path));
        }
        
        return tiles;
    }

    private static bool TryParseTile(string fileName, string map, out int x, out int y, out bool isCata)
    {
        x = y = 0; isCata = false;
        var stem = Path.GetFileNameWithoutExtension(fileName);
        if (!stem.StartsWith(map + "_", StringComparison.OrdinalIgnoreCase)) return false;
        var body = stem[(map.Length + 1)..];
        var parts = body.Split('_');
        if (parts.Length >= 2 && int.TryParse(parts[0], out x) && int.TryParse(parts[1], out y))
        {
            isCata = body.Contains("_obj", StringComparison.OrdinalIgnoreCase);
            return true;
        }
        return false;
    }

    // Core extractor: use Warcraft.NET parser instead of manual chunk walking
    private static List<PlacementRow> ExtractPlacementsFromAdt(IArchiveSource src, string virtualPath, string map, int x, int y)
    {
        using var stream = src.OpenFile(virtualPath);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        var data = ms.ToArray();
        
        // Determine if this is a Cata-split _obj0 file or pre-Cata terrain file
        bool isCataSplit = virtualPath.Contains("_obj0.adt", StringComparison.OrdinalIgnoreCase);
        
        var rows = new List<PlacementRow>();
        
        try
        {
            if (isCataSplit)
            {
                // Cata+ split format: _obj0.adt contains placements
                var objFile = new TerrainObjectZero(data);
                rows.AddRange(ExtractFromCataObj(objFile, map, x, y));
            }
            else
            {
                // Pre-Cata format: terrain ADT contains everything
                var terrain = new Terrain(data);
                rows.AddRange(ExtractFromPreCataTerrain(terrain, map, x, y));
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Warning: Failed to parse {virtualPath}: {ex.Message}");
        }
        
        return rows;
    }
    
    private static IEnumerable<PlacementRow> ExtractFromCataObj(TerrainObjectZero objFile, string map, int x, int y)
    {
        var m2Paths = BuildM2PathLookup(objFile.Models, objFile.ModelIndices);
        var wmoPaths = BuildWmoPathLookup(objFile.WorldModelObjects, objFile.WorldModelObjectIndices);
        
        // Extract M2 placements from MDDF
        if (objFile.ModelPlacementInfo?.MDDFEntries != null)
        {
            foreach (var m2 in objFile.ModelPlacementInfo.MDDFEntries)
            {
                if (m2Paths.TryGetValue(m2.NameId, out var path))
                {
                    var modelType = path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) ? "MDX" : "M2";
                    yield return new PlacementRow(map, x, y, modelType, path, (int)m2.UniqueID,
                        m2.Position.X, m2.Position.Y, m2.Position.Z,
                        m2.Rotation.Pitch, m2.Rotation.Yaw, m2.Rotation.Roll,
                        m2.ScalingFactor, null, null);
                }
            }
        }
        
        // Extract WMO placements from MODF
        if (objFile.WorldModelObjectPlacementInfo?.MODFEntries != null)
        {
            foreach (var wmo in objFile.WorldModelObjectPlacementInfo.MODFEntries)
            {
                if (wmoPaths.TryGetValue(wmo.NameId, out var path))
                {
                    yield return new PlacementRow(map, x, y, "WMO", path, (int)wmo.UniqueId,
                        wmo.Position.X, wmo.Position.Y, wmo.Position.Z,
                        wmo.Rotation.Pitch, wmo.Rotation.Yaw, wmo.Rotation.Roll,
                        wmo.Scale, wmo.DoodadSet, wmo.NameSet);
                }
            }
        }
    }
    
    private static IEnumerable<PlacementRow> ExtractFromPreCataTerrain(Terrain terrain, string map, int x, int y)
    {
        var m2Paths = BuildM2PathLookup(terrain.Models, terrain.ModelIndices);
        var wmoPaths = BuildWmoPathLookup(terrain.WorldModelObjects, terrain.WorldModelObjectIndices);
        
        // Extract M2 placements from MDDF
        if (terrain.ModelPlacementInfo?.MDDFEntries != null)
        {
            foreach (var m2 in terrain.ModelPlacementInfo.MDDFEntries)
            {
                if (m2Paths.TryGetValue(m2.NameId, out var path))
                {
                    var modelType = path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) ? "MDX" : "M2";
                    yield return new PlacementRow(map, x, y, modelType, path, (int)m2.UniqueID,
                        m2.Position.X, m2.Position.Y, m2.Position.Z,
                        m2.Rotation.Pitch, m2.Rotation.Yaw, m2.Rotation.Roll,
                        m2.ScalingFactor, null, null);
                }
            }
        }
        
        // Extract WMO placements from MODF
        if (terrain.WorldModelObjectPlacementInfo?.MODFEntries != null)
        {
            foreach (var wmo in terrain.WorldModelObjectPlacementInfo.MODFEntries)
            {
                if (wmoPaths.TryGetValue(wmo.NameId, out var path))
                {
                    yield return new PlacementRow(map, x, y, "WMO", path, (int)wmo.UniqueId,
                        wmo.Position.X, wmo.Position.Y, wmo.Position.Z,
                        wmo.Rotation.Pitch, wmo.Rotation.Yaw, wmo.Rotation.Roll,
                        wmo.Scale, wmo.DoodadSet, wmo.NameSet);
                }
            }
        }
    }
    
    private static Dictionary<uint, string> BuildM2PathLookup(Warcraft.NET.Files.ADT.Chunks.MMDX? mmdx, Warcraft.NET.Files.ADT.Chunks.MMID? mmid)
    {
        var lookup = new Dictionary<uint, string>();
        if (mmdx == null || mmid == null || mmdx.Filenames.Count == 0 || mmid.ModelFilenameOffsets.Count == 0)
            return lookup;
        
        for (int i = 0; i < mmdx.Filenames.Count; i++)
        {
            lookup[(uint)i] = mmdx.Filenames[i];
        }
        return lookup;
    }
    
    private static Dictionary<uint, string> BuildWmoPathLookup(Warcraft.NET.Files.ADT.Chunks.MWMO? mwmo, Warcraft.NET.Files.ADT.Chunks.MWID? mwid)
    {
        var lookup = new Dictionary<uint, string>();
        if (mwmo == null || mwid == null || mwmo.Filenames.Count == 0 || mwid.ModelFilenameOffsets.Count == 0)
            return lookup;
        
        for (int i = 0; i < mwmo.Filenames.Count; i++)
        {
            lookup[(uint)i] = mwmo.Filenames[i];
        }
        return lookup;
    }
    
    private static string FormatPlacementCsv(PlacementRow p, string sourceFile)
    {
        return $"{p.Map},{p.TileX},{p.TileY},{p.Type},{p.AssetPath},{p.UniqueId}," +
               $"{p.WorldX:F3},{p.WorldY:F3},{p.WorldZ:F3}," +
               $"{p.RotX:F6},{p.RotY:F6},{p.RotZ:F6}," +
               $"{p.Scale},{p.DoodadSet?.ToString() ?? ""},{p.NameSet?.ToString() ?? ""},{sourceFile}";
    }
}

internal record PlacementRow(
    string Map,
    int TileX,
    int TileY,
    string Type,
    string AssetPath,
    int UniqueId,
    float WorldX,
    float WorldY,
    float WorldZ,
    float RotX,
    float RotY,
    float RotZ,
    ushort Scale,
    ushort? DoodadSet,
    ushort? NameSet
);
