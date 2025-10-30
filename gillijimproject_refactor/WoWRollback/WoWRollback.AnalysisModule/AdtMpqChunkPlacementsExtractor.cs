using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Services.Archive;
using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.ADT;
using WoWFormatLib.Structs.WDT;
using WoWFormatLib.Utils;

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
            csv.AppendLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set,source_file,fdid");

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
                        int fdid = 0;
                        if (source is WoWRollback.Core.Services.Archive.CascArchiveSource casc)
                        {
                            var normPath = (p.AssetPath ?? string.Empty).Replace('\\','/').ToLowerInvariant();
                            if (casc.TryGetFileDataId(normPath, out var id)) fdid = id;
                        }
                        csv.AppendLine(FormatPlacementCsv(p, Path.GetFileName(virtualPath), fdid));
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

        // Prefer listfile enumeration for Cataclysm+ _obj0 tiles
        var objMatches = src.EnumerateFiles($"{basePath}/{map}_*_obj0.adt").ToList();
        if (objMatches.Count > 0)
        {
            foreach (var vp in objMatches)
            {
                if (TryParseTile(vp, map, out var x, out var y, out var isCata))
                {
                    tiles.Add((x, y, isCata, vp));
                }
            }
            tiles.Sort((a,b) => a.Item1 != b.Item1 ? a.Item1.CompareTo(b.Item1) : a.Item2.CompareTo(b.Item2));
            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Found {tiles.Count} _obj0 tiles via listfile");
            return tiles;
        }

        Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Falling back to per-tile existence scan (0-63 grid)...");
        var chosen = new Dictionary<(int X,int Y), (bool IsCata, string Path)>();
        int foundCount = 0;
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
        });

        foreach (var kv in chosen.OrderBy(k => k.Key.X).ThenBy(k => k.Key.Y))
        {
            tiles.Add((kv.Key.X, kv.Key.Y, kv.Value.IsCata, kv.Value.Path));
        }
        Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Scan complete: found {tiles.Count} tiles");
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

    // Core extractor: use our WoWFormatLib-based reader to preserve UniqueID semantics
    private static List<PlacementRow> ExtractPlacementsFromAdt(IArchiveSource src, string virtualPath, string map, int x, int y)
    {
        var rows = new List<PlacementRow>();

        bool isCataSplit = virtualPath.Contains("_obj0.adt", StringComparison.OrdinalIgnoreCase);

        try
        {
            var reader = new ADTReader();

            if (isCataSplit)
            {
                // Cata+ split: _obj0 carries MDDF/MODF
                using var obj = src.OpenFile(virtualPath);
                reader.ReadObjFile(obj);

                // MDDF (M2/MDX)
                if (reader.adtfile.objects.models.entries != null && reader.adtfile.objects.models.entries.Length > 0)
                {
                    foreach (var m2 in reader.adtfile.objects.models.entries)
                    {
                        var path = ResolveM2Path(reader.adtfile, m2.mmidEntry) ?? $"<MMID:{m2.mmidEntry}>";
                        var modelType = path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) ? "MDX" : "M2";
                        rows.Add(new PlacementRow(map, x, y, modelType, path, (int)m2.uniqueId,
                            m2.position.X, m2.position.Y, m2.position.Z,
                            m2.rotation.X, m2.rotation.Y, m2.rotation.Z,
                            m2.scale, null, null));
                    }
                }

                // MODF (WMO)
                if (reader.adtfile.objects.worldModels.entries != null && reader.adtfile.objects.worldModels.entries.Length > 0)
                {
                    foreach (var wmo in reader.adtfile.objects.worldModels.entries)
                    {
                        var path = ResolveWmoPath(reader.adtfile, wmo.mwidEntry) ?? $"<MWID:{wmo.mwidEntry}>";
                        rows.Add(new PlacementRow(map, x, y, "WMO", path, (int)wmo.uniqueId,
                            wmo.position.X, wmo.position.Y, wmo.position.Z,
                            wmo.rotation.X, wmo.rotation.Y, wmo.rotation.Z,
                            wmo.scale, wmo.doodadSet, wmo.nameSet));
                    }
                }
            }
            else
            {
                // Pre-Cata: MDDF/MODF live in root ADT
                // Force WotLK mode to make the reader walk MDDF/MODF in root files
                VersionManager.CurrentVersion = VersionManager.FileVersion.WotLK;
                using var root = src.OpenFile(virtualPath);
                reader.ReadRootFile(root, MPHDFlags.wdt_has_maid);

                if (reader.adtfile.objects.models.entries != null)
                {
                    foreach (var m2 in reader.adtfile.objects.models.entries)
                    {
                        var path = ResolveM2Path(reader.adtfile, m2.mmidEntry) ?? $"<MMID:{m2.mmidEntry}>";
                        var modelType = path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) ? "MDX" : "M2";
                        rows.Add(new PlacementRow(map, x, y, modelType, path, (int)m2.uniqueId,
                            m2.position.X, m2.position.Y, m2.position.Z,
                            m2.rotation.X, m2.rotation.Y, m2.rotation.Z,
                            m2.scale, null, null));
                    }
                }

                if (reader.adtfile.objects.worldModels.entries != null)
                {
                    foreach (var wmo in reader.adtfile.objects.worldModels.entries)
                    {
                        var path = ResolveWmoPath(reader.adtfile, wmo.mwidEntry) ?? $"<MWID:{wmo.mwidEntry}>";
                        rows.Add(new PlacementRow(map, x, y, "WMO", path, (int)wmo.uniqueId,
                            wmo.position.X, wmo.position.Y, wmo.position.Z,
                            wmo.rotation.X, wmo.rotation.Y, wmo.rotation.Z,
                            wmo.scale, wmo.doodadSet, wmo.nameSet));
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Warning: Failed to parse {virtualPath}: {ex.Message}");
        }

        return rows;
    }
    private static string? ResolveM2Path(ADT adt, uint mmidEntry)
    {
        try
        {
            var names = adt.objects.m2Names.filenames;
            var nameOffsets = adt.objects.m2Names.offsets;
            var mmid = adt.objects.m2NameOffsets.offsets;
            if (names == null) return null;
            if (mmid != null && mmidEntry < mmid.Length && nameOffsets != null)
            {
                var offset = mmid[mmidEntry];
                var idx = Array.IndexOf(nameOffsets, offset);
                if (idx >= 0 && idx < names.Length) return names[idx];
            }
            // Fallback: direct index
            if (mmidEntry < names.Length) return names[mmidEntry];
            return null;
        }
        catch { return null; }
    }

    private static string? ResolveWmoPath(ADT adt, uint mwidEntry)
    {
        try
        {
            var names = adt.objects.wmoNames.filenames;
            var nameOffsets = adt.objects.wmoNames.offsets;
            var mwid = adt.objects.wmoNameOffsets.offsets;
            if (names == null) return null;
            if (mwid != null && mwidEntry < mwid.Length && nameOffsets != null)
            {
                var offset = mwid[mwidEntry];
                var idx = Array.IndexOf(nameOffsets, offset);
                if (idx >= 0 && idx < names.Length) return names[idx];
            }
            if (mwidEntry < names.Length) return names[mwidEntry];
            return null;
        }
        catch { return null; }
    }
    
    private static string FormatPlacementCsv(PlacementRow p, string sourceFile, int fdid)
    {
        return $"{p.Map},{p.TileX},{p.TileY},{p.Type},{p.AssetPath},{p.UniqueId}," +
               $"{p.WorldX:F3},{p.WorldY:F3},{p.WorldZ:F3}," +
               $"{p.RotX:F6},{p.RotY:F6},{p.RotZ:F6}," +
               $"{p.Scale},{p.DoodadSet?.ToString() ?? ""},{p.NameSet?.ToString() ?? ""},{sourceFile},{(fdid > 0 ? fdid.ToString() : "")}";
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
