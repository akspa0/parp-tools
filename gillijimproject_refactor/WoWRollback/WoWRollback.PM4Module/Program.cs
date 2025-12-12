// ADT Merger - Merges split 3.3.5 ADTs into monolithic format
// Creates "clean" LK ADTs preserving all existing object data + WDT
// Also supports WDL to ADT generation for filling gaps

using WoWRollback.PM4Module;
using WoWRollback.Core.Services.PM4;
using System.Numerics;
using System.Text;

// Check for subcommands
if (args.Length > 0)
{
    switch (args[0])
    {
        case "wdl-to-adt":
            return WdlToAdtProgram.Run(args.Skip(1).ToArray());
        
        case "merge-split":
            return RunMergeSplit(args.Skip(1).ToArray());
        
        case "merge-minimap":
            return RunMergeWithMinimap(args.Skip(1).ToArray());
        
        case "compare":
            return RunCompare(args.Skip(1).ToArray());
        
        case "extract-mpq":
            return RunExtractMpq(args.Skip(1).ToArray());
        
        case "merge-textures":
            return RunMergeTextures(args.Skip(1).ToArray());
        
        case "test-roundtrip":
            return RunTestRoundtrip(args.Skip(1).ToArray());
        
        case "pm4-reconstruct-modf":
            return RunPm4ReconstructModf(args.Skip(1).ToArray());

        case "inject-modf":
            return RunInjectModf(args.Skip(1).ToArray());

        case "dump-modf":
            return RunDumpModf(args.Skip(1).ToArray());

        case "dump-modf-csv":
            return RunDumpModfCsv(args.Skip(1).ToArray());

        case "patch-pipeline":
            return RunPatchPipeline(args.Skip(1).ToArray());
    }
}

// merge-split command - uses Warcraft.NET for proper split→monolithic conversion
static int RunMergeSplit(string[] args)
{
    Console.WriteLine("=== Split ADT Merger (Warcraft.NET) ===\n");
    
    string? inputDir = null;
    string? outputDir = null;
    string? mapName = null;
    string? referenceDir = null;
    string? singleTile = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--in": inputDir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--reference": referenceDir = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: merge-split --in <dir> --out <dir> --map <name> [--reference <dir>] [--tile X_Y]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --in <dir>        Directory containing split ADT files (root + _obj0 + _tex0)");
                Console.WriteLine("  --out <dir>       Output directory for merged monolithic ADTs");
                Console.WriteLine("  --map <name>      Map name prefix (e.g., 'development')");
                Console.WriteLine("  --reference <dir> Optional: directory with reference ADTs for comparison");
                Console.WriteLine("  --tile X_Y        Optional: process only a single tile (e.g., '1_1')");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir))
    {
        Console.Error.WriteLine("Error: --in <dir> is required and must exist");
        return 1;
    }
    
    mapName ??= "development";
    outputDir ??= Path.Combine(inputDir, "merged");
    
    var merger = new SplitAdtMerger();
    
    if (!string.IsNullOrEmpty(singleTile))
    {
        // Single tile mode
        var parts = singleTile.Split('_');
        if (parts.Length != 2)
        {
            Console.Error.WriteLine("Error: --tile must be in format X_Y (e.g., '1_1')");
            return 1;
        }
        
        var baseName = $"{mapName}_{singleTile}";
        var rootPath = Path.Combine(inputDir, $"{baseName}.adt");
        var obj0Path = Path.Combine(inputDir, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(inputDir, $"{baseName}_tex0.adt");
        
        Directory.CreateDirectory(outputDir);
        
        var result = merger.Merge(rootPath, obj0Path, tex0Path);
        
        if (result.Success && result.Data != null)
        {
            var outputPath = Path.Combine(outputDir, $"{baseName}.adt");
            File.WriteAllBytes(outputPath, result.Data);
            Console.WriteLine($"\n[SUCCESS] Written: {outputPath} ({result.Data.Length:N0} bytes)");
            
            // Compare with reference if provided
            if (!string.IsNullOrEmpty(referenceDir))
            {
                var refPath = Path.Combine(referenceDir, $"{baseName}.adt");
                if (File.Exists(refPath))
                {
                    merger.CompareWithReference(outputPath, refPath);
                }
            }
            
            return 0;
        }
        else
        {
            Console.Error.WriteLine($"[FAIL] {result.Error}");
            return 1;
        }
    }
    else
    {
        // Batch mode
        int count = merger.BatchMerge(inputDir, outputDir, mapName);
        return count > 0 ? 0 : 1;
    }
}

// compare command - compare merged ADT with reference
static int RunCompare(string[] args)
{
    if (args.Length < 2)
    {
        Console.WriteLine("Usage: compare <merged.adt> <reference.adt>");
        return 1;
    }
    
    var merger = new SplitAdtMerger();
    merger.CompareWithReference(args[0], args[1]);
    return 0;
}

// dump-modf command - dump raw MODF placements from an ADT for inspection
static int RunDumpModf(string[] args)
{
    string? adtPath = null;
    string? adtDir = null;
    string? mapName = null;
    string? tile = null;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--file": adtPath = args[++i]; break;
            case "--in": adtDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--tile": tile = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: dump-modf --file <adt_file> | --in <adt_dir> --map <name> --tile X_Y");
                return 0;
        }
    }

    if (string.IsNullOrEmpty(adtPath))
    {
        if (string.IsNullOrEmpty(adtDir) || string.IsNullOrEmpty(mapName) || string.IsNullOrEmpty(tile))
        {
            Console.Error.WriteLine("Error: either --file or (--in, --map, --tile) must be specified.");
            return 1;
        }

        var parts = tile.Split('_');
        if (parts.Length != 2 || !int.TryParse(parts[0], out int tx) || !int.TryParse(parts[1], out int ty))
        {
            Console.Error.WriteLine("Error: --tile must be in format X_Y (e.g., '22_18').");
            return 1;
        }

        adtPath = Path.Combine(adtDir, $"{mapName}_{tx}_{ty}.adt");
    }

    if (!File.Exists(adtPath))
    {
        Console.Error.WriteLine($"Error: ADT file not found: {adtPath}");
        return 1;
    }

    Console.WriteLine($"=== Dumping MODF from {adtPath} ===\n");

    var patcher = new AdtPatcher();
    var data = File.ReadAllBytes(adtPath);
    var parsed = patcher.ParseAdt(data);
    var modfChunk = parsed.FindChunk("MODF");

    if (modfChunk == null || modfChunk.Data.Length == 0)
    {
        Console.WriteLine("No MODF chunk found or it is empty.");
        return 0;
    }

    const float TileSize = 533.33333f;
    const float MapExtent = TileSize * 32f; // 17066.666...

    using var ms = new MemoryStream(modfChunk.Data);
    using var br = new BinaryReader(ms);

    int entrySize = 64; // SMMapObjDef size for 3.3.5
    int count = modfChunk.Data.Length / entrySize;
    Console.WriteLine($"Entries: {count}\n");

    for (int i = 0; i < count; i++)
    {
        uint nameId = br.ReadUInt32();
        uint uniqueId = br.ReadUInt32();
        float px = br.ReadSingle();
        float py = br.ReadSingle();
        float pz = br.ReadSingle();
        float rx = br.ReadSingle();
        float ry = br.ReadSingle();
        float rz = br.ReadSingle();
        float bminx = br.ReadSingle();
        float bminy = br.ReadSingle();
        float bminz = br.ReadSingle();
        float bmaxx = br.ReadSingle();
        float bmaxy = br.ReadSingle();
        float bmaxz = br.ReadSingle();
        ushort flags = br.ReadUInt16();
        ushort doodadSet = br.ReadUInt16();
        ushort nameSet = br.ReadUInt16();
        ushort scale = br.ReadUInt16();

        // Convert placement-space back to world-space for comparison
        float worldX = MapExtent - pz;
        float worldY = MapExtent - px;
        float worldZ = py;

        // Tile indices follow wowdev ADT/v18: blockX from X axis, blockY from Y axis
        int tileX = (int)(32 - (worldX / TileSize));
        int tileY = (int)(32 - (worldY / TileSize));

        Console.WriteLine($"[{i}] nameId={nameId} uniqueId={uniqueId}");
        Console.WriteLine($"     placement: ({px:F3}, {py:F3}, {pz:F3})");
        Console.WriteLine($"     world:     ({worldX:F3}, {worldY:F3}, {worldZ:F3})  tile=({tileX},{tileY})");
        Console.WriteLine($"     scale={scale} flags={flags} doodadSet={doodadSet} nameSet={nameSet}\n");
    }

    return 0;
}

// dump-modf-csv command - inspect PM4 reconstruction CSV entries for a tile
static int RunDumpModfCsv(string[] args)
{
    string? csvPath = null;
    string? tile = null;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--csv": csvPath = args[++i]; break;
            case "--tile": tile = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: dump-modf-csv --csv <modf_entries.csv> --tile X_Y");
                return 0;
        }
    }

    if (string.IsNullOrEmpty(csvPath) || string.IsNullOrEmpty(tile))
    {
        Console.Error.WriteLine("Error: --csv and --tile are required.");
        return 1;
    }

    if (!File.Exists(csvPath))
    {
        Console.Error.WriteLine($"Error: CSV file not found: {csvPath}");
        return 1;
    }

    var parts = tile.Split('_');
    if (parts.Length != 2 || !int.TryParse(parts[0], out int tileX) || !int.TryParse(parts[1], out int tileY))
    {
        Console.Error.WriteLine("Error: --tile must be in format X_Y (e.g., '22_18').");
        return 1;
    }

    Console.WriteLine($"=== Dumping modf_entries for tile {tileX}_{tileY} from {csvPath} ===\n");

    const float TileSize = 533.33333f;
    const float MapExtent = TileSize * 32f; // 17066.666...

    using var reader = new StreamReader(csvPath);
    string? header = reader.ReadLine();
    if (header == null)
    {
        Console.Error.WriteLine("CSV is empty.");
        return 1;
    }

    int count = 0;
    while (!reader.EndOfStream)
    {
        var line = reader.ReadLine();
        if (string.IsNullOrWhiteSpace(line))
            continue;

        var cols = line.Split(',');
        if (cols.Length < 12)
            continue;

        // ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence
        var ck24 = cols[0];
        var wmoPath = cols[1];
        if (!uint.TryParse(cols[2], out var nameId)) continue;
        if (!uint.TryParse(cols[3], out var uniqueId)) continue;
        if (!float.TryParse(cols[4], out var posX)) continue;
        if (!float.TryParse(cols[5], out var posY)) continue;
        if (!float.TryParse(cols[6], out var posZ)) continue;
        if (!float.TryParse(cols[7], out var rotX)) continue;
        if (!float.TryParse(cols[8], out var rotY)) continue;
        if (!float.TryParse(cols[9], out var rotZ)) continue;
        if (!float.TryParse(cols[10], out var scale)) continue;

        // Tile indices follow wowdev ADT/v18: blockX from X axis, blockY from Y axis
        int csvTileX = (int)(32 - (posX / TileSize));
        int csvTileY = (int)(32 - (posY / TileSize));

        if (csvTileX != tileX || csvTileY != tileY)
            continue;

        // Convert world (posX,posY,posZ) to placement space in the same way
        // RunInjectModf now does when building AdtPatcher.ModfEntry.
        float placementX = MapExtent - posY;
        float placementY = posZ;
        float placementZ = MapExtent - posX;

        Console.WriteLine($"ck24={ck24} nameId={nameId} uniqueId={uniqueId}");
        Console.WriteLine($"  WMO: {wmoPath}");
        Console.WriteLine($"  world:     ({posX:F3}, {posY:F3}, {posZ:F3})  tile=({csvTileX},{csvTileY})");
        Console.WriteLine($"  placement: ({placementX:F3}, {placementY:F3}, {placementZ:F3})");
        Console.WriteLine($"  rot=({rotX:F2}, {rotY:F2}, {rotZ:F2}) scale={scale:F4}\n");
        count++;
    }

    Console.WriteLine($"Total entries for tile {tileX}_{tileY}: {count}");
    return 0;
}

// pm4-reconstruct-modf command - run PM4 WMO matching and export MODF CSV/MWMO table
static int RunPm4ReconstructModf(string[] args)
{
    Console.WriteLine("=== PM4 MODF Reconstruction ===\n");

    string? pm4Dir = null;
    string? wmoDir = null;
    string? outDir = null;
    string? minConfStr = null;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--pm4": pm4Dir = args[++i]; break;
            case "--wmo": wmoDir = args[++i]; break;
            case "--out": outDir = args[++i]; break;
            case "--min-confidence": minConfStr = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: pm4-reconstruct-modf --pm4 <pm4_dir> --wmo <wmo_collision_dir> --out <out_dir> [--min-confidence <0-1>]");
                Console.WriteLine();
                Console.WriteLine("  --pm4    Root directory containing PM4Faces ck_instances.csv + OBJ geometry");
                Console.WriteLine("  --wmo    Root directory containing extracted WMO collision OBJ files (per-WMO folders OK)");
                Console.WriteLine("  --out    Output directory for modf_entries.csv and mwmo_names.csv");
                Console.WriteLine("  --min-confidence  Minimum match confidence (default 0.7)");
                return 0;
        }
    }

    if (string.IsNullOrEmpty(pm4Dir) || string.IsNullOrEmpty(wmoDir) || string.IsNullOrEmpty(outDir))
    {
        Console.Error.WriteLine("Error: --pm4, --wmo and --out are required. Use --help for usage.");
        return 1;
    }

    if (!Directory.Exists(pm4Dir))
    {
        Console.Error.WriteLine($"Error: PM4 directory not found: {pm4Dir}");
        return 1;
    }

    if (!Directory.Exists(wmoDir))
    {
        Console.Error.WriteLine($"Error: WMO collision directory not found: {wmoDir}");
        return 1;
    }

    Directory.CreateDirectory(outDir);

    float minConfidence = 0.7f;
    if (!string.IsNullOrEmpty(minConfStr) && float.TryParse(minConfStr, out var parsedConf))
        minConfidence = parsedConf;

    var reconstructor = new Pm4ModfReconstructor();

    // NOTE: This command is redundant with the main pipeline but kept for standalone testing.
    // It assumes wmoDir is the GAME PATH and we need a listfile.
    // Since standalone usage is vague, we'll error if insufficient args or try best effort.
    string listfile = "listfile.csv"; // Fallback
    
    Console.WriteLine($"[INFO] Building WMO library (Note: wmoDir argument treated as GamePath)...");
    var wmoLibrary = reconstructor.BuildWmoLibrary(wmoDir, listfile, outDir);
    if (wmoLibrary.Count == 0)
    {
        Console.Error.WriteLine("[ERROR] WMO library is empty. Check your --wmo path.");
        return 1;
    }

    Console.WriteLine($"[INFO] Loading PM4 objects from {pm4Dir}...");
    var pm4Objects = reconstructor.LoadPm4Objects(pm4Dir);
    if (pm4Objects.Count == 0)
    {
        Console.Error.WriteLine("[ERROR] No PM4 objects found. Check your --pm4 path.");
        return 1;
    }

    Console.WriteLine($"[INFO] Reconstructing MODF entries with min-confidence {minConfidence:F2}...");
    var result = reconstructor.ReconstructModf(pm4Objects, wmoLibrary, minConfidence);

    // Apply PM4->ADT world coordinate transform before exporting CSVs
    Console.WriteLine("[INFO] Applying PM4->ADT coordinate transform (ServerToAdtPosition)...");
    var transformedEntries = result.ModfEntries
        .Select(e => e with { Position = PipelineCoordinateService.ServerToAdtPosition(e.Position) })
        .ToList();
    result = result with { ModfEntries = transformedEntries };

    var modfCsvPath = Path.Combine(outDir, "modf_entries.csv");
    var mwmoCsvPath = Path.Combine(outDir, "mwmo_names.csv");
    var candidatesCsvPath = Path.Combine(outDir, "match_candidates.csv");
    var verifyJsonPath = Path.Combine(outDir, "placement_verification.json");

    reconstructor.ExportToCsv(result, modfCsvPath);
    reconstructor.ExportMwmoNames(result, mwmoCsvPath);
    reconstructor.ExportCandidatesCsv(result, candidatesCsvPath);
    reconstructor.ExportVerificationJson(result, pm4Objects, wmoLibrary, verifyJsonPath);

    Console.WriteLine("\n[RESULT]");
    Console.WriteLine($"  MODF CSV: {modfCsvPath}");
    Console.WriteLine($"  MWMO CSV: {mwmoCsvPath}");
    Console.WriteLine($"  Candidates CSV: {candidatesCsvPath}");
    Console.WriteLine($"  Verification JSON: {verifyJsonPath}");

    return 0;
}

// merge-minimap command - uses AdtPatcher with minimap MCCV painting for tiles without tex0
static int RunMergeWithMinimap(string[] args)
{
    Console.WriteLine("=== ADT Merger with Minimap MCCV Painting ===\n");
    
    string? inputDir = null;
    string? outputDir = null;
    string? mapName = null;
    string? minimapDir = null;
    string? singleTile = null;
    bool onlyMissingTex0 = false;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--in": inputDir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--minimap": minimapDir = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--only-missing-tex0": onlyMissingTex0 = true; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: merge-minimap --in <dir> --out <dir> --map <name> --minimap <dir> [--tile X_Y] [--only-missing-tex0]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --in <dir>          Directory containing split ADT files (root + _obj0 + _tex0)");
                Console.WriteLine("  --out <dir>         Output directory for merged monolithic ADTs");
                Console.WriteLine("  --map <name>        Map name prefix (e.g., 'development')");
                Console.WriteLine("  --minimap <dir>     Directory with minimap PNG files (e.g., development_X_Y.png)");
                Console.WriteLine("  --tile X_Y          Optional: process only a single tile (e.g., '1_1')");
                Console.WriteLine("  --only-missing-tex0 Only apply minimap MCCV to tiles without _tex0.adt");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir))
    {
        Console.Error.WriteLine("Error: --in <dir> is required and must exist");
        return 1;
    }
    
    mapName ??= "development";
    outputDir ??= Path.Combine(inputDir, "merged_minimap");
    
    Directory.CreateDirectory(outputDir);
    
    var patcher = new AdtPatcher();
    int successCount = 0;
    int failCount = 0;
    
    // Get list of tiles to process
    var tiles = new List<(int x, int y)>();
    
    if (!string.IsNullOrEmpty(singleTile))
    {
        var parts = singleTile.Split('_');
        if (parts.Length == 2 && int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y))
        {
            tiles.Add((x, y));
        }
        else
        {
            Console.Error.WriteLine("Error: --tile must be in format X_Y (e.g., '1_1')");
            return 1;
        }
    }
    else
    {
        // Find all root ADT files
        foreach (var file in Directory.GetFiles(inputDir, $"{mapName}_*.adt"))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            if (name.Contains("_obj") || name.Contains("_tex")) continue;
            
            var parts = name.Split('_');
            if (parts.Length >= 3 && int.TryParse(parts[1], out int x) && int.TryParse(parts[2], out int y))
            {
                tiles.Add((x, y));
            }
        }
    }
    
    Console.WriteLine($"Found {tiles.Count} tiles to process");
    if (!string.IsNullOrEmpty(minimapDir))
        Console.WriteLine($"Minimap directory: {minimapDir}");
    Console.WriteLine();
    
    foreach (var (x, y) in tiles.OrderBy(t => t.x).ThenBy(t => t.y))
    {
        var baseName = $"{mapName}_{x}_{y}";
        var rootPath = Path.Combine(inputDir, $"{baseName}.adt");
        var obj0Path = Path.Combine(inputDir, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(inputDir, $"{baseName}_tex0.adt");
        var outputPath = Path.Combine(outputDir, $"{baseName}.adt");
        
        bool hasTex0 = File.Exists(tex0Path) && new FileInfo(tex0Path).Length > 0;
        
        // Check if root ADT has MTEX chunk (textures)
        bool hasTextures = HasMtexChunk(rootPath);
        
        Console.WriteLine($"=== Processing {baseName} (tex0: {(hasTex0 ? "yes" : "NO")}, MTEX: {(hasTextures ? "yes" : "NO")}) ===");
        
        try
        {
            byte[][]? mccvData = null;
            
            // Only paint MCCV if tile has NO textures at all (no MTEX chunk and no tex0)
            bool needsMccv = !hasTextures && !hasTex0;
            if (!string.IsNullOrEmpty(minimapDir) && needsMccv)
            {
                var minimapPath = Path.Combine(minimapDir, $"{baseName}.png");
                if (File.Exists(minimapPath))
                {
                    try
                    {
                        mccvData = MccvPainter.GenerateAllMccvFromImage(minimapPath);
                        Console.WriteLine($"  Loaded minimap: {minimapPath}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  [WARN] Failed to load minimap: {ex.Message}");
                    }
                }
                else
                {
                    Console.WriteLine($"  [INFO] No minimap found at: {minimapPath}");
                }
            }
            
            patcher.MergeAndWrite(
                rootPath,
                File.Exists(obj0Path) ? obj0Path : null,
                File.Exists(tex0Path) ? tex0Path : null,
                outputPath,
                mccvData
            );
            
            if (File.Exists(outputPath))
            {
                var outSize = new FileInfo(outputPath).Length;
                Console.WriteLine($"  [SUCCESS] Output: {outSize:N0} bytes\n");
                successCount++;
            }
            else
            {
                Console.WriteLine($"  [FAIL] Output file not created\n");
                failCount++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [ERROR] {ex.Message}\n");
            failCount++;
        }
    }
    
    Console.WriteLine("=== Summary ===");
    Console.WriteLine($"Success: {successCount}");
    Console.WriteLine($"Failed: {failCount}");
    Console.WriteLine($"Output directory: {outputDir}");
    
    return failCount > 0 ? 1 : 0;
}

/// <summary>
/// Check if an ADT file has an MTEX chunk (textures).
/// </summary>
static bool HasMtexChunk(string adtPath)
{
    if (!File.Exists(adtPath)) return false;
    
    try
    {
        var bytes = File.ReadAllBytes(adtPath);
        // Look for XETM (reversed MTEX) in the file
        byte[] pattern = Encoding.ASCII.GetBytes("XETM");
        
        for (int i = 0; i <= bytes.Length - 8; i++)
        {
            if (bytes[i] == pattern[0] && bytes[i+1] == pattern[1] && 
                bytes[i+2] == pattern[2] && bytes[i+3] == pattern[3])
            {
                // Found MTEX, check if it has content (size > 0)
                int size = BitConverter.ToInt32(bytes, i + 4);
                return size > 0;
            }
        }
    }
    catch { }
    
    return false;
}

// extract-mpq command - extract ADT files from MPQ archive
static int RunExtractMpq(string[] args)
{
    Console.WriteLine("=== MPQ ADT Extractor ===\n");
    
    string? mpqPath = null;
    string? outputDir = null;
    string? mapName = null;
    string? singleTile = null;
    bool listOnly = false;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--mpq": mpqPath = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--list": listOnly = true; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: extract-mpq --mpq <path> --out <dir> --map <name> [--tile X_Y] [--list]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --mpq <path>   Path to MPQ archive (e.g., expansion.MPQ)");
                Console.WriteLine("  --out <dir>    Output directory for extracted ADTs");
                Console.WriteLine("  --map <name>   Map name to extract (e.g., 'development')");
                Console.WriteLine("  --tile X_Y     Optional: extract only a single tile");
                Console.WriteLine("  --list         List ADTs without extracting");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(mpqPath))
    {
        Console.Error.WriteLine("Error: --mpq <path> is required");
        return 1;
    }
    
    try
    {
        using var extractor = new MpqAdtExtractor(mpqPath);
        
        if (listOnly)
        {
            var pattern = string.IsNullOrEmpty(mapName) ? "*.adt" : $"World\\Maps\\{mapName}\\*.adt";
            var files = extractor.ListFiles(pattern);
            Console.WriteLine($"Found {files.Count} files matching '{pattern}':");
            foreach (var f in files.Take(100))
                Console.WriteLine($"  {f}");
            if (files.Count > 100)
                Console.WriteLine($"  ... and {files.Count - 100} more");
            return 0;
        }
        
        if (string.IsNullOrEmpty(mapName))
        {
            Console.Error.WriteLine("Error: --map <name> is required for extraction");
            return 1;
        }
        
        outputDir ??= Path.Combine(Directory.GetCurrentDirectory(), $"extracted_{mapName}");
        
        if (!string.IsNullOrEmpty(singleTile))
        {
            var parts = singleTile.Split('_');
            if (parts.Length != 2 || !int.TryParse(parts[0], out int x) || !int.TryParse(parts[1], out int y))
            {
                Console.Error.WriteLine("Error: --tile must be in format X_Y");
                return 1;
            }
            
            var outputPath = Path.Combine(outputDir, $"{mapName}_{x}_{y}.adt");
            if (extractor.ExtractAdt(mapName, x, y, outputPath))
            {
                Console.WriteLine($"[SUCCESS] Extracted to: {outputPath}");
                return 0;
            }
            else
            {
                Console.Error.WriteLine($"[FAIL] Could not extract {mapName}_{x}_{y}.adt");
                return 1;
            }
        }
        else
        {
            int count = extractor.ExtractMapAdts(mapName, outputDir);
            Console.WriteLine($"\n[DONE] Extracted {count} ADTs to: {outputDir}");
            return count > 0 ? 0 : 1;
        }
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[ERROR] {ex.Message}");
        return 1;
    }
}

// merge-textures command - merge texture data from monolithic ADTs into split ADT merge
static int RunMergeTextures(string[] args)
{
    Console.WriteLine("=== Texture Data Merger ===\n");
    Console.WriteLine("Merges texture data (MTEX, MCLY, MCAL) from monolithic ADTs (e.g., 2.4.3)");
    Console.WriteLine("into split ADT merge for tiles missing _tex0.adt files.\n");
    
    string? splitDir = null;      // Directory with Cata split ADTs
    string? textureDir = null;    // Directory with monolithic ADTs containing texture data
    string? outputDir = null;
    string? mapName = null;
    string? minimapDir = null;
    string? singleTile = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--split": splitDir = args[++i]; break;
            case "--textures": textureDir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--minimap": minimapDir = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: merge-textures --split <dir> --textures <dir> --out <dir> --map <name> [--minimap <dir>] [--tile X_Y]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --split <dir>    Directory with Cata split ADTs (root + _obj0 + _tex0)");
                Console.WriteLine("  --textures <dir> Directory with monolithic ADTs containing texture data (e.g., from 2.4.3)");
                Console.WriteLine("  --out <dir>      Output directory for merged ADTs");
                Console.WriteLine("  --map <name>     Map name prefix (e.g., 'development')");
                Console.WriteLine("  --minimap <dir>  Optional: minimap PNGs for MCCV painting");
                Console.WriteLine("  --tile X_Y       Optional: process only a single tile");
                Console.WriteLine();
                Console.WriteLine("This command:");
                Console.WriteLine("  1. For tiles WITH _tex0.adt: uses existing texture data");
                Console.WriteLine("  2. For tiles WITHOUT _tex0.adt: extracts MTEX/MCLY/MCAL from monolithic ADT");
                Console.WriteLine("  3. Optionally paints MCCV from minimap for tiles without textures");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(splitDir) || !Directory.Exists(splitDir))
    {
        Console.Error.WriteLine("Error: --split <dir> is required and must exist");
        return 1;
    }
    
    if (string.IsNullOrEmpty(textureDir) || !Directory.Exists(textureDir))
    {
        Console.Error.WriteLine("Error: --textures <dir> is required and must exist");
        return 1;
    }
    
    mapName ??= "development";
    outputDir ??= Path.Combine(splitDir, "merged_with_textures");
    
    Directory.CreateDirectory(outputDir);
    
    Console.WriteLine($"Split ADTs:     {splitDir}");
    Console.WriteLine($"Texture source: {textureDir}");
    Console.WriteLine($"Output:         {outputDir}");
    Console.WriteLine($"Map:            {mapName}");
    if (!string.IsNullOrEmpty(minimapDir))
        Console.WriteLine($"Minimap:        {minimapDir}");
    Console.WriteLine();
    
    // Get list of tiles to process
    var tiles = new List<(int x, int y)>();
    
    if (!string.IsNullOrEmpty(singleTile))
    {
        var parts = singleTile.Split('_');
        if (parts.Length == 2 && int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y))
            tiles.Add((x, y));
        else
        {
            Console.Error.WriteLine("Error: --tile must be in format X_Y");
            return 1;
        }
    }
    else
    {
        foreach (var file in Directory.GetFiles(splitDir, $"{mapName}_*.adt"))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            if (name.Contains("_obj") || name.Contains("_tex")) continue;
            
            var parts = name.Split('_');
            if (parts.Length >= 3 && int.TryParse(parts[1], out int x) && int.TryParse(parts[2], out int y))
                tiles.Add((x, y));
        }
    }
    
    Console.WriteLine($"Found {tiles.Count} tiles to process\n");
    
    var patcher = new AdtPatcher();
    var textureExtractor = new TextureDataExtractor();
    int successCount = 0;
    int failCount = 0;
    int texturesFromMonolithic = 0;
    
    foreach (var (x, y) in tiles.OrderBy(t => t.x).ThenBy(t => t.y))
    {
        var baseName = $"{mapName}_{x}_{y}";
        var rootPath = Path.Combine(splitDir, $"{baseName}.adt");
        var obj0Path = Path.Combine(splitDir, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(splitDir, $"{baseName}_tex0.adt");
        var monoPath = Path.Combine(textureDir, $"{baseName}.adt");
        var outputPath = Path.Combine(outputDir, $"{baseName}.adt");
        
        bool hasTex0 = File.Exists(tex0Path) && new FileInfo(tex0Path).Length > 0;
        bool hasMono = File.Exists(monoPath) && new FileInfo(monoPath).Length > 0;
        
        Console.WriteLine($"=== {baseName} ===");
        Console.WriteLine($"  tex0: {(hasTex0 ? "yes" : "NO")}, monolithic: {(hasMono ? "yes" : "NO")}");
        
        try
        {
            byte[][]? mccvData = null;
            
            // Load minimap MCCV if available and tile lacks tex0
            if (!string.IsNullOrEmpty(minimapDir) && !hasTex0)
            {
                var minimapPath = Path.Combine(minimapDir, $"{baseName}.png");
                if (File.Exists(minimapPath))
                {
                    try
                    {
                        mccvData = MccvPainter.GenerateAllMccvFromImage(minimapPath);
                        Console.WriteLine($"  Loaded minimap MCCV");
                    }
                    catch { }
                }
            }
            
            // If no tex0 but we have monolithic, extract texture data from it
            string? effectiveTex0Path = tex0Path;
            if (!hasTex0 && hasMono)
            {
                Console.WriteLine($"  Extracting textures from monolithic ADT...");
                var monoData = File.ReadAllBytes(monoPath);
                var textureData = textureExtractor.ExtractTextureChunks(monoData);
                
                if (textureData != null)
                {
                    // Write temporary tex0-like data
                    var tempTex0 = Path.Combine(outputDir, $".temp_{baseName}_tex0.bin");
                    File.WriteAllBytes(tempTex0, textureData);
                    effectiveTex0Path = tempTex0;
                    texturesFromMonolithic++;
                    Console.WriteLine($"  Extracted texture data: {textureData.Length:N0} bytes");
                }
                else
                {
                    Console.WriteLine($"  [WARN] No texture data in monolithic ADT");
                    effectiveTex0Path = null;
                }
            }
            else if (!hasTex0)
            {
                effectiveTex0Path = null;
            }
            
            patcher.MergeAndWrite(
                rootPath,
                File.Exists(obj0Path) ? obj0Path : null,
                effectiveTex0Path != null && File.Exists(effectiveTex0Path) ? effectiveTex0Path : null,
                outputPath,
                mccvData
            );
            
            // Clean up temp file
            if (effectiveTex0Path != null && effectiveTex0Path.Contains(".temp_"))
            {
                try { File.Delete(effectiveTex0Path); } catch { }
            }
            
            if (File.Exists(outputPath))
            {
                var outSize = new FileInfo(outputPath).Length;
                Console.WriteLine($"  [SUCCESS] {outSize:N0} bytes\n");
                successCount++;
            }
            else
            {
                Console.WriteLine($"  [FAIL] Output not created\n");
                failCount++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [ERROR] {ex.Message}\n");
            failCount++;
        }
    }
    
    Console.WriteLine("=== Summary ===");
    Console.WriteLine($"Success: {successCount}");
    Console.WriteLine($"Failed: {failCount}");
    Console.WriteLine($"Textures from monolithic: {texturesFromMonolithic}");
    Console.WriteLine($"Output: {outputDir}");
    
    return failCount > 0 ? 1 : 0;
}

// Simple arg parser for adt-merge
var opts = new Dictionary<string, string>();
if (args.Length > 0 && args[0] == "adt-merge")
{
    // Skip command name
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i].StartsWith("--") && i + 1 < args.Length)
        {
            opts[args[i].Substring(2)] = args[i + 1];
            i++;
        }
    }
}
else
{
    // Legacy support or direct run
    for (int i = 0; i < args.Length; i++)
    {
        if (args[i].StartsWith("--") && i + 1 < args.Length)
        {
            opts[args[i].Substring(2)] = args[i + 1];
            i++;
        }
    }
}

Console.WriteLine("=== ADT Merger - Clean LK ADT Generation ===\n");

// Paths - Default or from Args
var sourceDir = opts.GetValueOrDefault("in", @"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\development\World\Maps\development");
var outputDir = opts.GetValueOrDefault("out", @"j:\wowDev\parp-tools\gillijimproject_refactor\PM4ADTs\clean");
var mapName = opts.GetValueOrDefault("map", "development");

// Create output directory
Directory.CreateDirectory(outputDir);
Console.WriteLine($"Source: {sourceDir}");
Console.WriteLine($"Output: {outputDir}");
Console.WriteLine($"Map:    {mapName}\n");

// Find tiles that have both ADT and PM4 data (non-zero size)
var tiles = new List<(int x, int y, long adtSize, long pm4Size, bool hasObj0, bool hasTex0)>();

foreach (var file in Directory.GetFiles(sourceDir, "*.adt"))
{
    var name = Path.GetFileNameWithoutExtension(file);
    
    // Skip split files
    if (name.Contains("_obj") || name.Contains("_tex")) continue;
    
    // Parse tile coordinates: development_X_Y.adt
    var parts = name.Split('_');
    if (parts.Length >= 3 && int.TryParse(parts[1], out int x) && int.TryParse(parts[2], out int y))
    {
        var adtSize = new FileInfo(file).Length;
        var pm4Path = Path.Combine(sourceDir, $"{mapName}_{x}_{y}.pm4");
        var pm4Size = File.Exists(pm4Path) ? new FileInfo(pm4Path).Length : 0;
        
        var obj0Path = Path.Combine(sourceDir, $"{mapName}_{x}_{y}_obj0.adt");
        var tex0Path = Path.Combine(sourceDir, $"{mapName}_{x}_{y}_tex0.adt");
        var hasObj0 = File.Exists(obj0Path) && new FileInfo(obj0Path).Length > 0;
        var hasTex0 = File.Exists(tex0Path) && new FileInfo(tex0Path).Length > 0;
        
        if (adtSize > 0) // Only include tiles with actual ADT data
        {
            tiles.Add((x, y, adtSize, pm4Size, hasObj0, hasTex0));
        }
    }
}

Console.WriteLine($"Found {tiles.Count} tiles with ADT data\n");

// Show sample of tiles
Console.WriteLine("Sample tiles:");
Console.WriteLine("| Tile | ADT Size | PM4 Size | Has _obj0 | Has _tex0 |");
Console.WriteLine("|------|----------|----------|-----------|-----------|");
foreach (var t in tiles.OrderBy(t => t.x).ThenBy(t => t.y).Take(15))
{
    Console.WriteLine($"| {t.x,2}_{t.y,2} | {t.adtSize,8:N0} | {t.pm4Size,8:N0} | {(t.hasObj0 ? "Yes" : "No"),-9} | {(t.hasTex0 ? "Yes" : "No"),-9} |");
}
Console.WriteLine();

// Process ALL tiles
var testTiles = tiles.OrderBy(t => t.x).ThenBy(t => t.y).ToList();

Console.WriteLine($"Processing {testTiles.Count} tiles:\n");

var patcher = new AdtPatcher();
int successCount = 0;
int failCount = 0;

foreach (var tile in testTiles)
{
    var baseName = $"{mapName}_{tile.x}_{tile.y}";
    var rootPath = Path.Combine(sourceDir, $"{baseName}.adt");
    var obj0Path = Path.Combine(sourceDir, $"{baseName}_obj0.adt");
    var tex0Path = Path.Combine(sourceDir, $"{baseName}_tex0.adt");
    var outputPath = Path.Combine(outputDir, $"{baseName}.adt");
    
    Console.WriteLine($"=== Processing {baseName} ===");
    
    try
    {
        // Merge split ADTs into monolithic format, preserving all existing data
        patcher.MergeAndWrite(
            rootPath,
            File.Exists(obj0Path) ? obj0Path : null,
            File.Exists(tex0Path) ? tex0Path : null,
            outputPath
        );
        
        // Verify output
        if (File.Exists(outputPath))
        {
            var outSize = new FileInfo(outputPath).Length;
            Console.WriteLine($"[SUCCESS] Output: {outSize:N0} bytes\n");
            successCount++;
        }
        else
        {
            Console.WriteLine($"[FAIL] Output file not created\n");
            failCount++;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[ERROR] {ex.Message}\n");
        failCount++;
    }
}

Console.WriteLine("=== Summary ===");
Console.WriteLine($"Success: {successCount}");
Console.WriteLine($"Failed: {failCount}");
Console.WriteLine($"Output directory: {outputDir}");

// Generate WDT file with correct flags (matching reference: 0x0E = MCCV | BigAlpha | DoodadRefsSorted)
Console.WriteLine("\n=== Generating WDT ===");
var wdtWriter = new Wdt335Writer();
var tileSet = new HashSet<(int x, int y)>(testTiles.Select(t => (t.x, t.y)));
var wdtFlags = Wdt335Writer.MphdFlags.AdtHasMccv | Wdt335Writer.MphdFlags.AdtHasBigAlpha | Wdt335Writer.MphdFlags.AdtHasDoodadRefsSortedBySizeCat;
var wdtData = wdtWriter.CreateWdt(tileSet, wdtFlags);
var wdtPath = Path.Combine(outputDir, $"{mapName}.wdt");
File.WriteAllBytes(wdtPath, wdtData);
Console.WriteLine($"[SUCCESS] WDT written: {wdtPath} ({wdtData.Length:N0} bytes)");
Console.WriteLine($"  Tiles marked: {tileSet.Count}");

return 0;

// test-roundtrip command - verify Warcraft.NET can load/save ADTs without corruption
static int RunTestRoundtrip(string[] args)
{
    Console.WriteLine("=== ADT Roundtrip Test ===\n");
    Console.WriteLine("Tests if Warcraft.NET can load and save WoWMuseum ADTs without corruption.\n");
    
    string? inputPath = null;
    string? outputPath = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--in": inputPath = args[++i]; break;
            case "--out": outputPath = args[++i]; break;
        }
    }
    
    if (string.IsNullOrEmpty(inputPath) || !File.Exists(inputPath))
    {
        Console.Error.WriteLine("Usage: test-roundtrip --in <adt_file> --out <output_file>");
        return 1;
    }
    
    outputPath ??= Path.Combine(Path.GetDirectoryName(inputPath)!, "roundtrip_" + Path.GetFileName(inputPath));
    
    var patcher = new Pm4AdtPatcher();
    return patcher.TestRoundtrip(inputPath, outputPath) ? 0 : 1;
}

// inject-modf command - inject MODF data from CSV into WoWMuseum ADTs
static int RunInjectModf(string[] args)
{
    Console.WriteLine("=== MODF Injection from PM4 Reconstruction ===\n");
    
    string? modfCsvPath = null;
    string? mwmoCsvPath = null;
    string? inputAdtDir = null;
    string? outputDir = null;
    string? mapName = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--modf": modfCsvPath = args[++i]; break;
            case "--mwmo": mwmoCsvPath = args[++i]; break;
            case "--in": inputAdtDir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--map": mapName = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: inject-modf --modf <csv> --mwmo <csv> --in <adt_dir> --out <dir> --map <name>");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --modf <csv>    Path to modf_entries.csv from PM4 reconstruction");
                Console.WriteLine("  --mwmo <csv>    Path to mwmo_names.csv from PM4 reconstruction");
                Console.WriteLine("  --in <dir>      Directory containing source ADTs (e.g., WoWMuseum)");
                Console.WriteLine("  --out <dir>     Output directory for patched ADTs");
                Console.WriteLine("  --map <name>    Map name prefix (e.g., 'development')");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(modfCsvPath) || string.IsNullOrEmpty(mwmoCsvPath) || 
        string.IsNullOrEmpty(inputAdtDir) || string.IsNullOrEmpty(outputDir) || string.IsNullOrEmpty(mapName))
    {
        Console.Error.WriteLine("Missing required arguments. Use --help for usage.");
        return 1;
    }
    
    if (!File.Exists(modfCsvPath) || !File.Exists(mwmoCsvPath))
    {
        Console.Error.WriteLine($"CSV files not found: {modfCsvPath} or {mwmoCsvPath}");
        return 1;
    }
    
    Directory.CreateDirectory(outputDir);
    
    // Read MWMO names
    var wmoNames = new List<string>();
    foreach (var line in File.ReadLines(mwmoCsvPath).Skip(1))
    {
        var parts = line.Split(',');
        if (parts.Length >= 2)
            wmoNames.Add(parts[1].Trim());
    }
    Console.WriteLine($"[INFO] Loaded {wmoNames.Count} WMO names from {Path.GetFileName(mwmoCsvPath)}");
    
    // Read MODF entries and group by tile - use AdtPatcher.ModfEntry (manual writer path)
    var modfByTile = new Dictionary<(int x, int y), List<AdtPatcher.ModfEntry>>();
    int totalEntries = 0;
    
    foreach (var line in File.ReadLines(modfCsvPath).Skip(1))
    {
        var parts = line.Split(',');
        if (parts.Length < 12) continue;
        
        // Parse: ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence
        if (!uint.TryParse(parts[2], out var nameId)) continue;
        if (!uint.TryParse(parts[3], out var uniqueId)) continue;
        if (!float.TryParse(parts[4], out var posX)) continue;
        if (!float.TryParse(parts[5], out var posY)) continue;
        if (!float.TryParse(parts[6], out var posZ)) continue;
        if (!float.TryParse(parts[7], out var rotX)) continue;
        if (!float.TryParse(parts[8], out var rotY)) continue;
        if (!float.TryParse(parts[9], out var rotZ)) continue;
        if (!float.TryParse(parts[10], out var scale)) continue;
        
        // Convert world position to tile coordinates
        // WoW coordinate system: tile (32,32) is at world origin (0,0)
        // Each tile is 533.33333 units
        const float TileSize = 533.33333f;
        const float MapExtent = TileSize * 32f; // 17066.666...

        // Tile indices follow wowdev ADT/v18: blockX from X axis, blockY from Y axis
        int tileX = (int)(32 - (posX / TileSize));
        int tileY = (int)(32 - (posY / TileSize));
        
        // Clamp to valid range
        tileX = Math.Clamp(tileX, 0, 63);
        tileY = Math.Clamp(tileY, 0, 63);
        
        var key = (tileX, tileY);
        if (!modfByTile.TryGetValue(key, out var list))
        {
            list = new List<AdtPatcher.ModfEntry>();
            modfByTile[key] = list;
        }

        // CSV pos_x/pos_y are world-space coordinates (centered around 0) used for
        // tile grouping. MODF on disk stores corner-based placement coordinates
        // as described in ADT_v18. Convert world -> placement space here:
        //   worldX = 32*TILESIZE - pos.z
        //   worldY = 32*TILESIZE - pos.x
        // inverted:
        //   pos.x = 32*TILESIZE - worldY
        //   pos.z = 32*TILESIZE - worldX
        //   pos.y = worldZ

        float placementX = MapExtent - posY; // placement.x
        float placementY = posZ;             // placement.y (height)
        float placementZ = MapExtent - posX; // placement.z

        var placementPos = new Vector3(placementX, placementY, placementZ);

        list.Add(new AdtPatcher.ModfEntry
        {
            NameId = nameId,
            UniqueId = uniqueId,
            Position = placementPos,
            Rotation = new Vector3(rotX, rotY, rotZ),
            BoundsMin = new Vector3(placementX - 50f, placementY - 50f, placementZ - 50f),
            BoundsMax = new Vector3(placementX + 50f, placementY + 50f, placementZ + 50f),
            Flags = 0,
            DoodadSet = 0,
            NameSet = 0,
            Scale = (ushort)(scale * 1024f)
        });
        totalEntries++;
    }
    
    Console.WriteLine($"[INFO] Loaded {totalEntries} MODF entries across {modfByTile.Count} tiles");
    
    // Process each ADT using MuseumAdtPatcher (manual chunk-preserving path)
    var patcher = new MuseumAdtPatcher();
    int processed = 0;
    int injected = 0;
    int copied = 0;
    
    var adtFiles = Directory.GetFiles(inputAdtDir, $"{mapName}_*.adt")
        .Where(f => !f.Contains("_obj") && !f.Contains("_tex") && !f.Contains("_lod"))
        .ToList();
    
    Console.WriteLine($"[INFO] Found {adtFiles.Count} ADT files to process\n");
    
    foreach (var adtPath in adtFiles)
    {
        var fileName = Path.GetFileNameWithoutExtension(adtPath);
        var match = System.Text.RegularExpressions.Regex.Match(fileName, @"(\d+)_(\d+)$");
        if (!match.Success) continue;
        
        int tileX = int.Parse(match.Groups[1].Value);
        int tileY = int.Parse(match.Groups[2].Value);
        var outputPath = Path.Combine(outputDir, Path.GetFileName(adtPath));
        
        if (modfByTile.TryGetValue((tileX, tileY), out var entries) && entries.Count > 0)
        {
            Console.WriteLine($"[INJECT] {fileName}: {entries.Count} WMO placements");
            try
            {
                patcher.PatchWmoPlacements(adtPath, outputPath, wmoNames, entries);
                injected++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [ERROR] MuseumAdtPatcher failed: {ex.Message}");
                Console.WriteLine($"  [FALLBACK] Copying original file unchanged");
                File.Copy(adtPath, outputPath, overwrite: true);
                copied++;
            }
        }
        else
        {
            // Copy unchanged
            File.Copy(adtPath, outputPath, overwrite: true);
            copied++;
        }
        processed++;
    }
    
    Console.WriteLine($"\n=== Summary ===");
    Console.WriteLine($"Processed: {processed}");
    Console.WriteLine($"Injected MODF: {injected}");
    Console.WriteLine($"Copied unchanged: {copied}");
    Console.WriteLine($"Output: {outputDir}");
    
    return 0;
}

static int RunPatchPipeline(string[] args)
{
    string? gamePath = null;
    string? listfilePath = null;
    string? pm4Path = null;
    string? splitAdtPath = null;
    string? museumAdtPath = null;
    string? wdlPath = null;
    string? outputRoot = "PM4_to_ADT";
    string? wmoFilter = null;        // Filter WMOs by path prefix (e.g., "Northrend")
    bool useFullMesh = false;        // Use full WMO mesh instead of walkable surfaces

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--game": gamePath = args[++i]; break;
            case "--listfile": listfilePath = args[++i]; break;
            case "--pm4": pm4Path = args[++i]; break;
            case "--split-adt": splitAdtPath = args[++i]; break;
            case "--museum-adt": museumAdtPath = args[++i]; break;
            case "--wdl": wdlPath = args[++i]; break;
            case "--out": outputRoot = args[++i]; break;
            case "--wmo-filter": wmoFilter = args[++i]; break;
            case "--use-full-mesh": useFullMesh = true; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: patch-pipeline --game <path> --listfile <path> --pm4 <dir> --split-adt <dir> --museum-adt <dir> [options]");
                Console.WriteLine();
                Console.WriteLine("Required:");
                Console.WriteLine("  --game <path>       Path to WoW 3.3.5 client (for WMO extraction from MPQs)");
                Console.WriteLine("  --listfile <path>   Listfile with WMO paths");
                Console.WriteLine("  --pm4 <dir>         Directory containing .pm4 files");
                Console.WriteLine("  --split-adt <dir>   Directory with split ADT data (for WDL file)");
                Console.WriteLine("  --museum-adt <dir>  WoWMuseum LK ADTs to patch");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --wdl <file>        Path to WDL file (optional, auto-detected from split-adt)");
                Console.WriteLine("  --out <dir>         Output directory (default: PM4_to_ADT)");
                Console.WriteLine("  --wmo-filter <path> Filter WMOs by path prefix (e.g., 'Northrend' or 'Kalimdor')");
                Console.WriteLine("  --use-full-mesh     Use full WMO mesh for matching (not just walkable surfaces)");
                return 0;
        }
    }

    if (string.IsNullOrEmpty(gamePath) || string.IsNullOrEmpty(listfilePath) || 
        string.IsNullOrEmpty(pm4Path) || string.IsNullOrEmpty(splitAdtPath) || string.IsNullOrEmpty(museumAdtPath))
    {
        Console.Error.WriteLine("Error: Missing required arguments. Use --help.");
        return 1;
    }

    try
    {
        var pipeline = new PipelineService();
        pipeline.Execute(gamePath, listfilePath, pm4Path, splitAdtPath, museumAdtPath, outputRoot, wdlPath, wmoFilter, useFullMesh);
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[FATAL] Pipeline failed: {ex.Message}");
        Console.Error.WriteLine(ex.StackTrace);
        return 1;
    }
}
