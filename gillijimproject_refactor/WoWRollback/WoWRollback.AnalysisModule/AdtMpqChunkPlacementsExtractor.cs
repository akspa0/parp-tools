using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts placements (M2/WMO) from ADT files inside MPQs using a tolerant chunk walker.
/// Only the minimal chunks are parsed: MMDX/MMID, MWMO/MWID, MDDF, MODF.
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
            int wmoCount = 0;
            int tilesProcessed = 0;

            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Processing {tiles.Count} tiles from archive...");

            foreach (var (tileX, tileY, _, virtualPath) in tiles)
            {
                try
                {
                    var rows = ExtractPlacementsFromAdt(source, virtualPath, mapName, tileX, tileY);
                    foreach (var p in rows)
                    {
                        csv.AppendLine(FormatPlacementCsv(p, Path.GetFileName(virtualPath)));
                        if (p.Type == "M2") m2Count++; else wmoCount++;
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

            return new PlacementsExtractionResult(true, m2Count, wmoCount, tilesProcessed, null);
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

        bool any = false;
        foreach (var path in src.EnumerateFiles($"{basePath}/*.adt"))
        {
            any = true;
            var file = Path.GetFileName(path);
            if (TryParseTile(file, map, out var x, out var y, out var isCata))
                tiles.Add((x, y, isCata, path));
        }
        if (any && tiles.Count > 0)
        {
            return tiles.Distinct().OrderBy(t => t.Item1).ThenBy(t => t.Item2).ToList();
        }

        // No listfile: brute-force 64x64, prefer obj0 over pre for each (x,y)
        var chosen = new Dictionary<(int X,int Y), (bool IsCata, string Path)>();
        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
                var obj0 = $"{basePath}/{map}_{x}_{y}_obj0.adt";
                if (src.FileExists(obj0)) { chosen[(x,y)] = (true, obj0); continue; }
                var pre = $"{basePath}/{map}_{x}_{y}.adt";
                if (src.FileExists(pre)) { chosen[(x,y)] = (false, pre); }
            }
        }
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

    // Core extractor: walk chunks; build index tables; parse MDDF/MODF
    private static List<PlacementRow> ExtractPlacementsFromAdt(IArchiveSource src, string virtualPath, string map, int x, int y)
    {
        using var s = src.OpenFile(virtualPath);
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        var data = ms.ToArray();

        var chunks = EnumerateChunks(data);
        var mmdx = GetChunk(chunks, "MMDX", data);
        var mmid = GetChunk(chunks, "MMID", data);
        var mwmo = GetChunk(chunks, "MWMO", data);
        var mwid = GetChunk(chunks, "MWID", data);

        var m2IndexToPath = BuildIndexedTable(mmdx, mmid);
        var wmoIndexToPath = BuildIndexedTable(mwmo, mwid);

        // WDT fallback: if ADT-local name tables are empty, try MDNM/MONM from <map>.wdt
        if ((m2IndexToPath.Count == 0 || wmoIndexToPath.Count == 0) && TryEnsureWdtTables(src, map, out var mdnm, out var monm))
        {
            if (m2IndexToPath.Count == 0 && mdnm.Count > 0)
            {
                m2IndexToPath = mdnm;
                Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Using WDT MDNM fallback for models.");
            }
            if (wmoIndexToPath.Count == 0 && monm.Count > 0)
            {
                wmoIndexToPath = monm;
                Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Using WDT MONM fallback for WMOs.");
            }
        }

        var rows = new List<PlacementRow>();

        // MDDF entries
        var mddf = GetChunk(chunks, "MDDF", data);
        if (!mddf.IsEmpty)
        {
            const int MDDF_SIZE = 36;
            for (int pos = 0; pos + MDDF_SIZE <= mddf.Length; pos += MDDF_SIZE)
            {
                int nameIndex = ReadInt32LE(mddf.Slice(pos + 0));
                int unique = ReadInt32LE(mddf.Slice(pos + 4));
                float worldX = ReadFloatLE(mddf.Slice(pos + 8));
                float worldZ = ReadFloatLE(mddf.Slice(pos + 12));
                float worldY = ReadFloatLE(mddf.Slice(pos + 16));
                float rotX = ReadFloatLE(mddf.Slice(pos + 20));
                float rotY = ReadFloatLE(mddf.Slice(pos + 24));
                float rotZ = ReadFloatLE(mddf.Slice(pos + 28));
                ushort scaleRaw = ReadUInt16LE(mddf.Slice(pos + 32));
                float scaleF = scaleRaw > 0 ? scaleRaw / 1024.0f : 1.0f;
                ushort scale = (ushort)Math.Round(scaleF * 1024.0f);

                if (nameIndex >= 0 && nameIndex < m2IndexToPath.Count)
                {
                    rows.Add(new PlacementRow(map, x, y, "M2", m2IndexToPath[nameIndex], unique,
                        worldX, worldY, worldZ, rotX, rotY, rotZ, scale, null, null));
                }
            }
        }

        // MODF entries
        var modf = GetChunk(chunks, "MODF", data);
        if (!modf.IsEmpty)
        {
            const int MODF_SIZE = 64;
            for (int pos = 0; pos + MODF_SIZE <= modf.Length; pos += MODF_SIZE)
            {
                int nameIndex = ReadInt32LE(modf.Slice(pos + 0));
                int unique = ReadInt32LE(modf.Slice(pos + 4));
                float worldX = ReadFloatLE(modf.Slice(pos + 8));
                float worldZ = ReadFloatLE(modf.Slice(pos + 12));
                float worldY = ReadFloatLE(modf.Slice(pos + 16));
                float rotX = ReadFloatLE(modf.Slice(pos + 20));
                float rotY = ReadFloatLE(modf.Slice(pos + 24));
                float rotZ = ReadFloatLE(modf.Slice(pos + 28));
                ushort flags = ReadUInt16LE(modf.Slice(pos + 56));
                ushort doodadSet = ReadUInt16LE(modf.Slice(pos + 58));
                ushort nameSet = ReadUInt16LE(modf.Slice(pos + 60));
                ushort scaleRaw = ReadUInt16LE(modf.Slice(pos + 62));
                float scaleF = scaleRaw > 0 ? scaleRaw / 1024.0f : 1.0f;
                ushort scale = (ushort)Math.Round(scaleF * 1024.0f);

                if (nameIndex >= 0 && nameIndex < wmoIndexToPath.Count)
                {
                    rows.Add(new PlacementRow(map, x, y, "WMO", wmoIndexToPath[nameIndex], unique,
                        worldX, worldY, worldZ, rotX, rotY, rotZ, scale, doodadSet, nameSet));
                }
            }
        }

        // Minimal debug for first few tiles if no placements found but string tables exist
        if (rows.Count == 0 && (mmdx.Length > 0 || mwmo.Length > 0))
        {
            Console.WriteLine($"[AdtMpqChunkPlacementsExtractor] Debug: ({x},{y}) has MMDX={mmdx.Length} MWMO={mwmo.Length} MDDF={mddf.Length} MODF={modf.Length}");
        }
        return rows;
    }

    // Chunk walker utilities
    private readonly record struct ChunkInfo(string Id, int Offset, int Size);

    private static List<ChunkInfo> EnumerateChunks(byte[] data)
    {
        var list = new List<ChunkInfo>();
        int pos = 0;
        int safety = 0;
        while (pos + 8 <= data.Length && safety++ < 100000)
        {
            var rawId = ReadFourCc(data, pos);
            var id = NormalizeFourCc(rawId);
            int size = (int)BitConverter.ToUInt32(data, pos + 4);
            int payload = Math.Max(0, Math.Min(size, data.Length - (pos + 8)));
            list.Add(new ChunkInfo(id, pos + 8, payload));
            long next = (long)pos + 8 + size;
            if (next <= pos || next > data.Length) break;
            pos = (int)next;
        }
        return list;
    }

    private static ReadOnlySpan<byte> GetChunk(List<ChunkInfo> chunks, string id, byte[] data)
    {
        for (int i = 0; i < chunks.Count; i++)
        {
            var c = chunks[i];
            if (string.Equals(c.Id, id, StringComparison.Ordinal))
            {
                if (c.Offset >= 0 && c.Offset + c.Size <= data.Length)
                    return new ReadOnlySpan<byte>(data, c.Offset, c.Size);
                return ReadOnlySpan<byte>.Empty;
            }
        }
        return ReadOnlySpan<byte>.Empty;
    }

    private static List<string> BuildIndexedTable(ReadOnlySpan<byte> stringBlock, ReadOnlySpan<byte> indexBlock)
    {
        var list = new List<string>();
        if (stringBlock.Length == 0 || indexBlock.Length == 0) return list;

        var offsetToString = new Dictionary<int, string>();
        int i = 0;
        while (i < stringBlock.Length)
        {
            int start = i;
            while (i < stringBlock.Length && stringBlock[i] != 0) i++;
            int len = i - start;
            if (len > 0)
            {
                var s = Encoding.ASCII.GetString(stringBlock.Slice(start, len));
                offsetToString[start] = s;
            }
            i++; // skip null
        }

        for (int pos = 0; pos + 4 <= indexBlock.Length; pos += 4)
        {
            int offs = ReadInt32LE(indexBlock.Slice(pos));
            if (offs >= 0 && offsetToString.TryGetValue(offs, out var str))
                list.Add(str);
            else
                list.Add(string.Empty);
        }
        return list;
    }

    private static string ReadFourCc(byte[] data, int offset)
    {
        if (offset + 4 > data.Length) return string.Empty;
        return Encoding.ASCII.GetString(data, offset, 4);
    }

    private static string NormalizeFourCc(string id) => id switch
    {
        "REVM" => "MVER",
        "RDHM" => "MHDR",
        "NICM" => "MCIN",
        "XDDM" => "MMDX",
        "DIMM" => "MMID",
        "OMWM" => "MWMO",
        "DIWM" => "MWID",
        "MNDM" => "MDNM", // reverse guess safety
        "MNOM" => "MONM",
        "FDDM" => "MDDF",
        "FDOM" => "MODF",
        _ => id
    };

    private static int ReadInt32LE(ReadOnlySpan<byte> s) => BitConverter.ToInt32(s.Slice(0, 4));
    private static ushort ReadUInt16LE(ReadOnlySpan<byte> s) => BitConverter.ToUInt16(s.Slice(0, 2));
    private static float ReadFloatLE(ReadOnlySpan<byte> s) => BitConverter.ToSingle(s.Slice(0, 4));

    // --- WDT fallback ---
    private static readonly Dictionary<string, (List<string> MDNM, List<string> MONM)> WdtCache = new(StringComparer.OrdinalIgnoreCase);

    private static bool TryEnsureWdtTables(IArchiveSource src, string map, out List<string> mdnm, out List<string> monm)
    {
        if (WdtCache.TryGetValue(map, out var cached))
        {
            mdnm = cached.MDNM; monm = cached.MONM; return true;
        }

        mdnm = new List<string>();
        monm = new List<string>();

        var wdtPath = $"world/maps/{map}/{map}.wdt";
        if (!src.FileExists(wdtPath)) return false;

        try
        {
            using var s = src.OpenFile(wdtPath);
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            var data = ms.ToArray();
            var chunks = EnumerateChunks(data);
            var mdnmSpan = GetChunk(chunks, "MDNM", data);
            var monmSpan = GetChunk(chunks, "MONM", data);
            if (mdnmSpan.Length > 0) mdnm = BuildStringList(mdnmSpan);
            if (monmSpan.Length > 0) monm = BuildStringList(monmSpan);
            WdtCache[map] = (mdnm, monm);
            return (mdnm.Count + monm.Count) > 0;
        }
        catch
        {
            return false;
        }
    }

    private static List<string> BuildStringList(ReadOnlySpan<byte> block)
    {
        var list = new List<string>();
        int i = 0;
        while (i < block.Length)
        {
            int start = i;
            while (i < block.Length && block[i] != 0) i++;
            int len = i - start;
            if (len > 0)
            {
                var s = Encoding.ASCII.GetString(block.Slice(start, len));
                list.Add(s);
            }
            i++;
        }
        return list;
    }

    // CSV helpers
    private readonly record struct PlacementRow(
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
        ushort? NameSet);

    private static string FormatPlacementCsv(PlacementRow p, string sourceFile)
    {
        return string.Join(",",
            Csv(p.Map),
            p.TileX.ToString(CultureInfo.InvariantCulture),
            p.TileY.ToString(CultureInfo.InvariantCulture),
            Csv(p.Type),
            Csv(p.AssetPath),
            p.UniqueId.ToString(CultureInfo.InvariantCulture),
            p.WorldX.ToString("F3", CultureInfo.InvariantCulture),
            p.WorldY.ToString("F3", CultureInfo.InvariantCulture),
            p.WorldZ.ToString("F3", CultureInfo.InvariantCulture),
            p.RotX.ToString("F6", CultureInfo.InvariantCulture),
            p.RotY.ToString("F6", CultureInfo.InvariantCulture),
            p.RotZ.ToString("F6", CultureInfo.InvariantCulture),
            p.Scale.ToString(CultureInfo.InvariantCulture),
            p.DoodadSet?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
            p.NameSet?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
            Csv(sourceFile));
    }

    private static string Csv(string value)
        => string.IsNullOrEmpty(value) ? string.Empty : (value.Contains(',') || value.Contains('"') || value.Contains('\n')) ? $"\"{value.Replace("\"", "\"\"")}\"" : value;
}
