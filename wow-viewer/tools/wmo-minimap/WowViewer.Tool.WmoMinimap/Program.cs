using System.Text.Json;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

namespace WowViewer.Tools.WmoMinimap;

/// <summary>
/// Probe: list all BLP files under textures/minimap/ from the MPQ
/// and check if any WMO-named BLPs exist.
/// Also read WMOAreaTable.dbc if available.
/// </summary>
static class Program
{
    static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h")
        {
            Console.WriteLine("""
                WowViewer.Tool.WmoMinimap — investigate WMO minimap BLPs in the client.

                Commands:
                  list-minimap-blps --client-root <dir>   List minimap BLP paths  
                  probe-dbc-chain --client-root <dir>     Read WMOAreaTable/AreaTable/Map DBCs
                  extract-wmo-minimaps --client-root <dir> --asset-list <json> --output-dir <dir>
                    Extract per-WMO minimap BLPs from DBC chain resolution
                """);
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();
        return command switch
        {
            "list-minimap-blps" => RunListMinimapBlps(tail),
            "probe-dbc-chain" => RunProbeDbcChain(tail),
            "extract-wmo-minimaps" => RunExtractWmoMinimaps(tail),
            _ => throw new InvalidOperationException($"Unknown command '{command}'"),
        };
    }

    private static int RunListMinimapBlps(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root required.");
            return 1;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);

        int minimapBlpCount = 0;
        int wmoBlpCount = 0;
        Console.WriteLine("=== All minimap BLPs ===");
        foreach (string path in catalog.GetAllKnownFiles())
        {
            string lower = path.ToLowerInvariant();
            if (lower.Contains("minimap") && lower.EndsWith(".blp"))
            {
                minimapBlpCount++;
                if (lower.Contains("wmo") || lower.Contains("wmo/") || lower.Contains("dungeon") || lower.Contains("interior") || lower.Contains("instance"))
                {
                    Console.WriteLine($"  WMO/instance BLP: {path}");
                    wmoBlpCount++;
                }
            }
        }
        Console.WriteLine($"\nTotal minimap BLPs: {minimapBlpCount}");
        Console.WriteLine($"WMO/instance-related: {wmoBlpCount}");
        return 0;
    }

    private static int RunProbeDbcChain(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root required.");
            return 1;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);

        // Read WMOAreaTable.dbc
        byte[]? wmoAreaData = catalog.ReadFile(@"DBFilesClient\WMOAreaTable.dbc");
        if (wmoAreaData is null)
        {
            Console.Error.WriteLine("WMOAreaTable.dbc not found in client.");
            return 1;
        }

        var wmoAreaDbc = DbcReader.Load(wmoAreaData);
        Console.WriteLine($"WMOAreaTable.dbc: {wmoAreaDbc.Header.RecordCount} records, {wmoAreaDbc.Header.FieldCount} fields, {wmoAreaDbc.Header.RecordSize} bytes/record");

        // Print first few records
        for (int i = 0; i < Math.Min(5, (int)wmoAreaDbc.Header.RecordCount); i++)
        {
            uint wmoId = wmoAreaDbc.GetUInt(i, 1); // WMOAreaTable field 1 = wmoID
            uint areaTableId = wmoAreaDbc.GetUInt(i, 2); // field 2 = AreaTableID
            uint nameOffset = wmoAreaDbc.GetUInt(i, 3);
            string name = wmoAreaDbc.GetString(nameOffset);
            Console.WriteLine($"  [{i}] wmoID={wmoId} areaTableID={areaTableId} name='{name}'");
        }

        // Read AreaTable.dbc
        byte[]? areaData = catalog.ReadFile(@"DBFilesClient\AreaTable.dbc");
        if (areaData is not null)
        {
            var areaDbc = DbcReader.Load(areaData);
            Console.WriteLine($"\nAreaTable.dbc: {areaDbc.Header.RecordCount} records, {areaDbc.Header.FieldCount} fields");
            // Print first few: id, mapId, name
            for (int i = 0; i < Math.Min(5, (int)areaDbc.Header.RecordCount); i++)
            {
                uint id = areaDbc.GetUInt(i, 0);
                uint mapId = areaDbc.GetUInt(i, 1);
                uint nameOff = areaDbc.GetUInt(i, 11); // name field varies by version
                string name = areaDbc.GetString(nameOff);
                Console.WriteLine($"  [{i}] id={id} mapId={mapId} name='{name}'");
            }
        }
        else
        {
            Console.WriteLine("AreaTable.dbc not found.");
        }

        // Read Map.dbc
        byte[]? mapData = catalog.ReadFile(@"DBFilesClient\Map.dbc");
        if (mapData is not null)
        {
            var mapDbc = DbcReader.Load(mapData);
            Console.WriteLine($"\nMap.dbc: {mapDbc.Header.RecordCount} records, {mapDbc.Header.FieldCount} fields");
            for (int i = 0; i < Math.Min(5, (int)mapDbc.Header.RecordCount); i++)
            {
                uint id = mapDbc.GetUInt(i, 0);
                string dir = mapDbc.GetString(i, 1);
                uint nameOff = mapDbc.GetUInt(i, 5);
                string name = mapDbc.GetString(nameOff);
                Console.WriteLine($"  [{i}] id={id} dir='{dir}' name='{name}'");
            }
        }
        else
        {
            Console.WriteLine("Map.dbc not found.");
        }

        return 0;
    }

    private static int RunExtractWmoMinimaps(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? assetListPath = GetOption(args, "--asset-list", "-l");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string? buildLabel = GetOption(args, "--build", "-b");

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(assetListPath) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --client-root, --asset-list, and --output-dir required.");
            return 1;
        }

        if (!Directory.Exists(clientRoot)) { Console.Error.WriteLine("client root not found"); return 1; }
        Directory.CreateDirectory(outputDir);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);

        // Load md5translate for BLP resolution
        Md5TranslateIndex? md5Index = null;
        Md5TranslateResolver.TryLoad([clientRoot], p => catalog.FileExists(p), p => catalog.ReadFile(p), out md5Index);

        // Load DBCs
        byte[]? wmoAreaData = catalog.ReadFile(@"DBFilesClient\WMOAreaTable.dbc");
        byte[]? areaData = catalog.ReadFile(@"DBFilesClient\AreaTable.dbc");
        byte[]? mapData = catalog.ReadFile(@"DBFilesClient\Map.dbc");

        if (wmoAreaData is null || areaData is null || mapData is null)
        {
            Console.Error.WriteLine("Required DBCs not found in client.");
            return 1;
        }

        var wmoAreaDbc = DbcReader.Load(wmoAreaData);
        var areaDbc = DbcReader.Load(areaData);
        var mapDbc = DbcReader.Load(mapData);

        Console.Error.WriteLine($"WMOAreaTable: {wmoAreaDbc.Header.RecordCount} records");
        Console.Error.WriteLine($"AreaTable: {areaDbc.Header.RecordCount} records");
        Console.Error.WriteLine($"Map: {mapDbc.Header.RecordCount} records");

        // Build indexes
        Dictionary<uint, uint> wmoIdToAreaId = new();
        for (int i = 0; i < wmoAreaDbc.Header.RecordCount; i++)
        {
            uint wmoId = wmoAreaDbc.GetUInt(i, 1);
            uint areaId = wmoAreaDbc.GetUInt(i, 2);
            if (wmoId > 0) wmoIdToAreaId[wmoId] = areaId;
        }

        Dictionary<uint, uint> areaIdToMapId = new();
        for (int i = 0; i < areaDbc.Header.RecordCount; i++)
        {
            uint areaId = areaDbc.GetUInt(i, 0);
            uint mapId = areaDbc.GetUInt(i, 1);
            areaIdToMapId[areaId] = mapId;
        }

        Dictionary<uint, string> mapIdToDir = new();
        for (int i = 0; i < mapDbc.Header.RecordCount; i++)
        {
            uint mapId = mapDbc.GetUInt(i, 0);
            string dir = mapDbc.GetString(i, 1);
            if (!string.IsNullOrWhiteSpace(dir)) mapIdToDir[mapId] = dir;
        }

        Console.Error.WriteLine($"Index: {wmoIdToAreaId.Count} wmoIDs, {areaIdToMapId.Count} areaIDs, {mapIdToDir.Count} maps");

        // Load asset list
        string json = File.ReadAllText(assetListPath);
        List<string> allPaths = JsonSerializer.Deserialize<List<string>>(json) ?? [];
        var wmoPaths = allPaths.Where(p => p.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)).ToList();

        int success = 0, failed = 0;
        foreach (string assetPath in wmoPaths)
        {
            string mpqPath = assetPath.Replace('/', '\\');
            byte[]? rootBytes = catalog.ReadFile(mpqPath);
            if (rootBytes is null) { failed++; continue; }

            // Read MOHD.wmoID at offset 32
            uint? wmoId = ReadWmoId(rootBytes);
            if (wmoId is null || wmoId == 0) { failed++; continue; }

            // Chain: wmoID -> AreaTable ID -> Map ID -> Map directory
            if (!wmoIdToAreaId.TryGetValue(wmoId.Value, out uint areaId)) { failed++; continue; }
            if (!areaIdToMapId.TryGetValue(areaId, out uint mapId)) { failed++; continue; }
            if (!mapIdToDir.TryGetValue(mapId, out string? mapDir)) { failed++; continue; }

            // Map directory -> minimap path pattern: textures/minimap/{mapDir}/map{tx}_{ty}.blp
            // But we don't know the tile coords from just the WMO root.
            // We need the placement's world position. That comes from placements.parquet.
            // For now, list what we found.
            Console.Error.WriteLine($"  WMO: {assetPath}");
            Console.Error.WriteLine($"    wmoID={wmoId} -> areaID={areaId} -> mapID={mapId} -> mapDir='{mapDir}'");

            string safeName = SanitizeName(assetPath);
            string outTxt = Path.Combine(outputDir, $"{safeName}.txt");
            File.WriteAllText(outTxt, $"wmoID={wmoId}\nareaID={areaId}\nmapID={mapId}\nmapDir={mapDir}\nassetPath={assetPath}\n");

            success++;
        }

        Console.WriteLine($"Done. S={success} F={failed}");
        return 0;
    }

    /// <summary>Read wmoID from MOHD chunk at byte offset 32 (uint32).</summary>
    private static uint? ReadWmoId(byte[] rootBytes)
    {
        try
        {
            using var ms = new MemoryStream(rootBytes);
            var chunks = Core.IO.Chunked.ChunkedFileReader.ReadTopLevelChunks(ms, padOddChunkSizes: false);
            var mohdTarget = FourCC.FromString("MOHD");
            var mohd = chunks.FirstOrDefault(c => c.Header.Id == mohdTarget);
            if (mohd.Header.Id == mohdTarget && mohd.Header.Size >= 36)
            {
                byte[] payload = new byte[mohd.Header.Size];
                ms.Position = mohd.DataOffset;
                ms.ReadExactly(payload);
                return BitConverter.ToUInt32(payload, 32);
            }
        }
        catch { }
        return null;
    }

    private static string SanitizeName(string path)
    {
        string cleaned = path.Replace('/', '_').Replace('\\', '_').Replace('.', '_').Replace(':', '_');
        return cleaned.Length > 200 ? cleaned[..200] : cleaned;
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == longName || args[i] == shortName) return args[i + 1];
        return null;
    }
}
