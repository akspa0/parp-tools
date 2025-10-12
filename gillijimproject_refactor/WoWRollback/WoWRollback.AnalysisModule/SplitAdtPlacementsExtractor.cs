using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts object placements (MDDF/MODF) from Cataclysm+ split ADTs (_obj*.adt) or root ADTs
/// and writes a normalized placements.csv compatible with OverlayGenerator.GenerateObjectsFromPlacementsCsv.
/// </summary>
public sealed class SplitAdtPlacementsExtractor
{
    public string GeneratePlacementsCsv(string adtOutputDir, string mapName, string version)
    {
        var mapDir = Path.Combine(adtOutputDir, "World", "Maps", mapName);
        if (!Directory.Exists(mapDir))
            throw new DirectoryNotFoundException(mapDir);

        var analysisCsvDir = Path.Combine(adtOutputDir, "analysis", "csv");
        Directory.CreateDirectory(analysisCsvDir);
        var outPath = Path.Combine(analysisCsvDir, "placements.csv");

        var tiles = Directory.EnumerateFiles(mapDir, mapName + "_*.adt", SearchOption.TopDirectoryOnly)
            .Select(p => ParseTilePath(p, mapName))
            .Where(info => info != null)
            .Select(info => info!)
            .GroupBy(x => (x.TileX, x.TileY))
            .ToList();

        using var sw = new StreamWriter(outPath);
        sw.WriteLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set");

        int totalPlacements = 0;
        foreach (var g in tiles)
        {
            var (tileX, tileY) = g.Key;
            // choose best file for placements
            var obj0 = g.FirstOrDefault(t => t.Kind == TileFileKind.Obj0)?.FullPath;
            var obj1 = g.FirstOrDefault(t => t.Kind == TileFileKind.Obj1)?.FullPath;
            var root = g.FirstOrDefault(t => t.Kind == TileFileKind.Root)?.FullPath;
            var source = obj0 ?? obj1 ?? root;
            if (string.IsNullOrWhiteSpace(source) || !File.Exists(source)) continue;

            try
            {
                ExtractPlacementsFromAdt(source!, mapDir, mapName, tileX, tileY, sw, ref totalPlacements);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[split-adt] Failed {Path.GetFileName(source)}: {ex.Message}");
            }
        }

        Console.WriteLine($"[split-adt] placements.csv written with {totalPlacements} rows");
        return outPath;
    }

    private static void ExtractPlacementsFromAdt(
        string adtPath,
        string mapDir,
        string mapName,
        int tileX,
        int tileY,
        StreamWriter sw,
        ref int totalPlacements)
    {
        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);

        // Accumulate chunks across file
        var mmdxData = Array.Empty<byte>();
        var mwmoData = Array.Empty<byte>();
        var mmid = new List<uint>();
        var mwid = new List<uint>();
        var mddf = Array.Empty<byte>();
        var modf = Array.Empty<byte>();

        while (fs.Position + 8 <= fs.Length)
        {
            long pos = fs.Position;
            uint magic = br.ReadUInt32();
            uint size = br.ReadUInt32();
            long dataPos = fs.Position;
            if (size > int.MaxValue || dataPos + size > fs.Length)
            {
                // invalid; try to resync by moving 1 byte forward
                fs.Position = pos + 1;
                continue;
            }

            if (IsFourCC(magic, "MMDX")) mmdxData = br.ReadBytes((int)size);
            else if (IsFourCC(magic, "MWMO")) mwmoData = br.ReadBytes((int)size);
            else if (IsFourCC(magic, "MMID")) mmid = ReadUintArray(br.ReadBytes((int)size));
            else if (IsFourCC(magic, "MWID")) mwid = ReadUintArray(br.ReadBytes((int)size));
            else if (IsFourCC(magic, "MDDF")) mddf = br.ReadBytes((int)size);
            else if (IsFourCC(magic, "MODF")) modf = br.ReadBytes((int)size);
            else
            {
                // skip data
                fs.Position = dataPos + size;
            }
        }

        var m2Paths = BuildStringTable(mmdxData, mmid);
        var wmoPaths = BuildStringTable(mwmoData, mwid);

        // MDDF entries: 36 bytes each
        if (mddf.Length >= 36)
        {
            int count = mddf.Length / 36;
            for (int i = 0; i < count; i++)
            {
                int off = i * 36;
                uint nameId = BitConverter.ToUInt32(mddf, off + 0);
                uint uniqueId = BitConverter.ToUInt32(mddf, off + 4);
                float x = BitConverter.ToSingle(mddf, off + 8);
                float y = BitConverter.ToSingle(mddf, off + 12);
                float z = BitConverter.ToSingle(mddf, off + 16);
                float rx = BitConverter.ToSingle(mddf, off + 20);
                float ry = BitConverter.ToSingle(mddf, off + 24);
                float rz = BitConverter.ToSingle(mddf, off + 28);
                ushort scale = BitConverter.ToUInt16(mddf, off + 32);
                // ushort flags = BitConverter.ToUInt16(mddf, off + 34);

                string path = m2Paths.TryGetValue((int)nameId, out var p) ? p : string.Empty;
                WriteRow(sw, mapName, tileX, tileY, "M2", path, uniqueId, x, y, z, rx, ry, rz, scale / 1024.0f, 0, 0);
                totalPlacements++;
            }
        }

        // MODF entries: 64 bytes each (simplified)
        if (modf.Length >= 64)
        {
            int count = modf.Length / 64;
            for (int i = 0; i < count; i++)
            {
                int off = i * 64;
                uint nameId = BitConverter.ToUInt32(modf, off + 0);
                uint uniqueId = BitConverter.ToUInt32(modf, off + 4);
                float x = BitConverter.ToSingle(modf, off + 8);
                float y = BitConverter.ToSingle(modf, off + 12);
                float z = BitConverter.ToSingle(modf, off + 16);
                float rx = BitConverter.ToSingle(modf, off + 20);
                float ry = BitConverter.ToSingle(modf, off + 24);
                float rz = BitConverter.ToSingle(modf, off + 28);
                // Many fields follow; doodad/name sets near end on some versions. Default to 0 if not available.
                ushort doodadSet = 0;
                ushort nameSet = 0;
                string path = wmoPaths.TryGetValue((int)nameId, out var p) ? p : string.Empty;
                WriteRow(sw, mapName, tileX, tileY, "WMO", path, uniqueId, x, y, z, rx, ry, rz, 1.0f, doodadSet, nameSet);
                totalPlacements++;
            }
        }
    }

    private static void WriteRow(StreamWriter sw, string map, int tileX, int tileY, string type,
        string assetPath, uint uniqueId, float wx, float wy, float wz, float rx, float ry, float rz,
        float scale, ushort doodadSet, ushort nameSet)
    {
        string F(float v) => v.ToString("G9", CultureInfo.InvariantCulture);
        sw.WriteLine(string.Join(",",
            Csv(map), tileX, tileY, Csv(type), Csv(assetPath), uniqueId,
            F(wx), F(wy), F(wz), F(rx), F(ry), F(rz), F(scale), doodadSet, nameSet));
    }

    private static string Csv(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static bool IsFourCC(uint value, string tag)
    {
        uint a = ToFourCC(tag);
        uint b = ReverseBytes(a);
        return value == a || value == b;
    }

    private static uint ToFourCC(string s)
    {
        var b = System.Text.Encoding.ASCII.GetBytes(s);
        if (b.Length != 4) throw new ArgumentException("tag must be 4 chars");
        return (uint)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
    }

    private static uint ReverseBytes(uint v)
    {
        return (v & 0x000000FFU) << 24 | (v & 0x0000FF00U) << 8 | (v & 0x00FF0000U) >> 8 | (v & 0xFF000000U) >> 24;
    }

    private static Dictionary<int, string> BuildStringTable(byte[] tableData, List<uint> offsets)
    {
        var map = new Dictionary<int, string>();
        if (tableData.Length == 0 || offsets.Count == 0) return map;
        for (int i = 0; i < offsets.Count; i++)
        {
            int off = (int)offsets[i];
            if (off < 0 || off >= tableData.Length) continue;
            int end = off;
            while (end < tableData.Length && tableData[end] != 0) end++;
            var s = System.Text.Encoding.ASCII.GetString(tableData, off, end - off);
            map[i] = s;
        }
        return map;
    }

    private static List<uint> ReadUintArray(byte[] data)
    {
        var list = new List<uint>();
        for (int i = 0; i + 4 <= data.Length; i += 4)
        {
            list.Add(BitConverter.ToUInt32(data, i));
        }
        return list;
    }

    private static TileFileInfo? ParseTilePath(string fullPath, string mapName)
    {
        try
        {
            var stem = Path.GetFileNameWithoutExtension(fullPath);
            if (string.IsNullOrEmpty(stem) || !stem.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase)) return null;
            var parts = stem.Split('_');
            if (parts.Length < 3) return null;
            // If suffix like _obj0/_obj1/_tex0 present, last two numeric tokens are not at end; handle both cases
            int tileX, tileY;
            if (int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY))
            {
                // Plain root form <Map>_X_Y
            }
            else if (parts.Length >= 4 && int.TryParse(parts[^3], out tileX) && int.TryParse(parts[^2], out tileY))
            {
                // Suffix present; adjust
            }
            else
            {
                return null;
            }

            var kind = TileFileKind.Root;
            if (stem.EndsWith("_obj0", StringComparison.OrdinalIgnoreCase)) kind = TileFileKind.Obj0;
            else if (stem.EndsWith("_obj1", StringComparison.OrdinalIgnoreCase)) kind = TileFileKind.Obj1;
            else if (stem.EndsWith("_tex0", StringComparison.OrdinalIgnoreCase) || stem.EndsWith("_tex1", StringComparison.OrdinalIgnoreCase)) kind = TileFileKind.Tex;
            return new TileFileInfo(fullPath, tileX, tileY, kind);
        }
        catch { return null; }
    }

    private sealed record TileFileInfo(string FullPath, int TileX, int TileY, TileFileKind Kind);

    private enum TileFileKind { Root, Obj0, Obj1, Tex }
}
