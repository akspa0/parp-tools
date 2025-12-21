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
        
        case "export-mscn":
            return RunExportMscn(args.Skip(1).ToArray());
        
        case "test-wl-convert":
            return RunTestWlConvert(args.Skip(1).ToArray());

        case "analyze-pm4":
            return RunAnalyzePm4(args.Skip(1).ToArray());

        case "convert-matches-to-modf":
            return RunConvertMatchesToModf(args.Skip(1).ToArray());

        case "dump-pm4-geometry":
            return RunDumpPm4Geometry(args.Skip(1).ToArray());

        case "convert-ck24-to-wmo":
            return RunConvertCk24ToWmo(args.Skip(1).ToArray());

        case "analyze-pm4-scene":
            return RunAnalyzePm4Scene(args.Skip(1).ToArray());

        case "analyze-m2-library":
            return RunAnalyzeM2Library(args.Skip(1).ToArray());

        case "reconstruct-mddf":
            return RunReconstructMddf(args.Skip(1).ToArray());
        
        case "pm4-pipeline-v2":
            return RunPm4PipelineV2(args.Skip(1).ToArray());
        
        case "export-pm4-obj":
            return RunExportPm4Obj(args.Skip(1).ToArray());
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
    outputDir ??= Path.Combine(inputDir!, "merged");
    
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
        var rootPath = Path.Combine(inputDir!, $"{baseName}.adt");
        var obj0Path = Path.Combine(inputDir!, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(inputDir!, $"{baseName}_tex0.adt");
        
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

        adtPath = Path.Combine(adtDir!, $"{mapName}_{tx}_{ty}.adt");
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

    var modfCsvPath = Path.Combine(outDir!, "modf_entries.csv");
    var mwmoCsvPath = Path.Combine(outDir!, "mwmo_names.csv");
    var candidatesCsvPath = Path.Combine(outDir!, "match_candidates.csv");
    var verifyJsonPath = Path.Combine(outDir!, "placement_verification.json");

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
    outputDir ??= Path.Combine(inputDir!, "merged_minimap");
    
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
        foreach (var file in Directory.GetFiles(inputDir!, $"{mapName}_*.adt"))
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
        var rootPath = Path.Combine(inputDir!, $"{baseName}.adt");
        var obj0Path = Path.Combine(inputDir!, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(inputDir!, $"{baseName}_tex0.adt");
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
            
            var outputPath = Path.Combine(outputDir!, $"{mapName}_{x}_{y}.adt");
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
            int count = extractor.ExtractMapAdts(mapName, outputDir!);
            Console.WriteLine($"\n[DONE] Extracted {count} ADTs to: {outputDir!}");
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
    outputDir ??= Path.Combine(splitDir!, "merged_with_textures");
    
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
        foreach (var file in Directory.GetFiles(splitDir!, $"{mapName}_*.adt"))
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
        var rootPath = Path.Combine(splitDir!, $"{baseName}.adt");
        var obj0Path = Path.Combine(splitDir!, $"{baseName}_obj0.adt");
        var tex0Path = Path.Combine(splitDir!, $"{baseName}_tex0.adt");
        var monoPath = Path.Combine(textureDir!, $"{baseName}.adt");
        var outputPath = Path.Combine(outputDir!, $"{baseName}.adt");
        
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
    uint nextAvailableUniqueId = 200_000_000; // Start high to avoid conflicts
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
                patcher.PatchWmoPlacements(adtPath, outputPath, wmoNames, entries, ref nextAvailableUniqueId);
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
    string? originalSplitPath = null; // Original development split ADTs for UniqueID restoration
    string? wdlPath = null;
    string? outputRoot = "PM4_to_ADT";
    string? wmoFilter = null;        // Filter WMOs by path prefix (e.g., "Northrend")
    string? m2Filter = null;         // Filter M2s by path prefix (e.g., "development")
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
            case "--original-split": originalSplitPath = args[++i]; break;
            case "--wdl": wdlPath = args[++i]; break;
            case "--out": outputRoot = args[++i]; break;
            case "--wmo-filter": wmoFilter = args[++i]; break;
            case "--m2-filter": m2Filter = args[++i]; break;
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
                Console.WriteLine("  --original-split <dir>  Directory with ORIGINAL development split ADTs (for UniqueID restoration)");
                Console.WriteLine("  --wdl <file>        Path to WDL file (optional, auto-detected from split-adt)");
                Console.WriteLine("  --out <dir>         Output directory (default: PM4_to_ADT)");
                Console.WriteLine("  --wmo-filter <path> Filter WMOs by path prefix (e.g., 'Northrend' or 'Kalimdor')");
                Console.WriteLine("  --m2-filter <path>  Filter M2s by path prefix (e.g., 'development')");
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
        pipeline.Execute(gamePath, listfilePath, pm4Path, splitAdtPath, museumAdtPath, outputRoot, wdlPath, wmoFilter, m2Filter, useFullMesh, originalSplitPath);
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[FATAL] Pipeline failed: {ex.Message}");
        Console.Error.WriteLine(ex.StackTrace);
        return 1;
    }
}

// export-mscn command - export MSCN data from PM4 files as OBJ for visualization
static int RunExportMscn(string[] args)
{
    Console.WriteLine("=== MSCN Data Export (Phase 2 Investigation) ===\n");
    
    string? pm4Path = null;
    string? outDir = null;
    bool applyTransform = true;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--pm4": pm4Path = args[++i]; break;
            case "--out": outDir = args[++i]; break;
            case "--raw": applyTransform = false; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: export-mscn --pm4 <file|dir> --out <dir> [--raw]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --pm4 <path>   Path to PM4 file or directory containing PM4 files");
                Console.WriteLine("  --out <dir>    Output directory for OBJ files");
                Console.WriteLine("  --raw          Export raw coordinates without transform (default: apply MSCN transform)");
                Console.WriteLine();
                Console.WriteLine("MSCN Transform: 180° X-axis rotation + Y-negate");
                Console.WriteLine("  correctedY = -Y");
                Console.WriteLine("  newY = correctedY * cos(π) - Z * sin(π)");
                Console.WriteLine("  newZ = correctedY * sin(π) + Z * cos(π)");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(pm4Path))
    {
        Console.Error.WriteLine("Error: --pm4 is required. Use --help for usage.");
        return 1;
    }
    
    outDir ??= "mscn_export";
    Directory.CreateDirectory(outDir);
    
    var pm4Files = new List<string>();
    if (Directory.Exists(pm4Path))
    {
        pm4Files.AddRange(Directory.GetFiles(pm4Path, "*.pm4", SearchOption.AllDirectories));
    }
    else if (File.Exists(pm4Path))
    {
        pm4Files.Add(pm4Path);
    }
    else
    {
        Console.Error.WriteLine($"Error: PM4 path not found: {pm4Path}");
        return 1;
    }
    
    Console.WriteLine($"Found {pm4Files.Count} PM4 files");
    Console.WriteLine($"Transform mode: {(applyTransform ? "MSCN transform (180° X rotation + Y-negate)" : "RAW coordinates")}\n");
    
    int totalMscnVerts = 0;
    int filesWithMscn = 0;
    
    foreach (var file in pm4Files)
    {
        try
        {
            var pm4 = PM4File.FromFile(file);
            var baseName = Path.GetFileNameWithoutExtension(file);
            
            if (pm4.ExteriorVertices.Count == 0)
            {
                Console.WriteLine($"[SKIP] {baseName}: No MSCN data");
                continue;
            }
            
            filesWithMscn++;
            totalMscnVerts += pm4.ExteriorVertices.Count;
            
            // Export as OBJ point cloud
            var objPath = Path.Combine(outDir, $"{baseName}_mscn.obj");
            using var writer = new StreamWriter(objPath);
            
            writer.WriteLine($"# MSCN data from {baseName}");
            writer.WriteLine($"# Vertices: {pm4.ExteriorVertices.Count}");
            writer.WriteLine($"# Transform: {(applyTransform ? "MSCN (180° X rot + Y-negate)" : "RAW")}");
            writer.WriteLine();
            
            foreach (var v in pm4.ExteriorVertices)
            {
                float x, y, z;
                
                if (applyTransform)
                {
                    // MSCN uses (Y,X,Z) file ordering + Y-axis mirror to match minimap
                    // Visual proof: ships' masts point wrong way without Y negation
                    x = v.Y;   // Y becomes X
                    y = -v.X;  // X becomes -Y (negate to fix mirror)
                    z = v.Z;   // Z unchanged
                }
                else
                {
                    x = v.X;
                    y = v.Y;
                    z = v.Z;
                }
                
                writer.WriteLine($"v {x:F6} {y:F6} {z:F6}");
            }
            
            Console.WriteLine($"[OK] {baseName}: {pm4.ExteriorVertices.Count} MSCN verts -> {objPath}");
            
            // Also export MSVT mesh for comparison if available
            if (pm4.MeshVertices.Count > 0 && pm4.Surfaces.Count > 0)
            {
                var meshObjPath = Path.Combine(outDir, $"{baseName}_msvt.obj");
                using var meshWriter = new StreamWriter(meshObjPath);
                
                meshWriter.WriteLine($"# MSVT mesh from {baseName}");
                meshWriter.WriteLine($"# Vertices: {pm4.MeshVertices.Count}, Surfaces: {pm4.Surfaces.Count}");
                meshWriter.WriteLine();
                
                // MSVT vertices need same Y-axis correction as MSCN for minimap alignment
                // MSVT already swapped (Y,X,Z -> X,Y,Z) in PM4File.cs, now negate Y for mirror fix
                foreach (var v in pm4.MeshVertices)
                {
                    meshWriter.WriteLine($"v {v.X:F6} {-v.Y:F6} {v.Z:F6}");
                }
                
                // Triangulate surfaces using fan method
                meshWriter.WriteLine();
                foreach (var surf in pm4.Surfaces)
                {
                    if (surf.IndexCount < 3) continue;
                    
                    for (int i = 2; i < surf.IndexCount; i++)
                    {
                        uint i0 = pm4.MeshIndices[(int)surf.MsviFirstIndex];
                        uint i1 = pm4.MeshIndices[(int)surf.MsviFirstIndex + i - 1];
                        uint i2 = pm4.MeshIndices[(int)surf.MsviFirstIndex + i];
                        
                        // OBJ uses 1-indexed
                        meshWriter.WriteLine($"f {i0+1} {i1+1} {i2+1}");
                    }
                }
                
                Console.WriteLine($"     + MSVT mesh: {pm4.MeshVertices.Count} verts, {pm4.Surfaces.Count} surfaces -> {meshObjPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {Path.GetFileName(file)}: {ex.Message}");
        }
    }
    
    Console.WriteLine();
    Console.WriteLine("=== Summary ===");
    Console.WriteLine($"Files with MSCN: {filesWithMscn}/{pm4Files.Count}");
    Console.WriteLine($"Total MSCN vertices: {totalMscnVerts}");
    Console.WriteLine($"Output directory: {outDir}");
    
    return 0;
}

// test-wl-convert command - Test WL* file parsing and MH2O conversion
static int RunTestWlConvert(string[] args)
{
    Console.WriteLine("=== WL* to MH2O Conversion Test ===\n");
    
    string? wlDir = null;
    string? outDir = null;
    bool verbose = false;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--wl": wlDir = args[++i]; break;
            case "--out": outDir = args[++i]; break;
            case "-v":
            case "--verbose": verbose = true; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: test-wl-convert --wl <dir> [--out <dir>] [--verbose]");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  --wl <dir>   Directory containing WL* files (WLW/WLM/WLQ)");
                Console.WriteLine("  --out <dir>  Output directory for MH2O data (default: <wl>/mh2o_test)");
                Console.WriteLine("  --verbose    Show detailed block/chunk info");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(wlDir) || !Directory.Exists(wlDir))
    {
        Console.Error.WriteLine("Error: --wl <dir> is required and must exist");
        return 1;
    }
    
    outDir ??= Path.Combine(wlDir, "mh2o_test");
    Directory.CreateDirectory(outDir);
    
    // Find all WL* files
    var wlFiles = Directory.GetFiles(wlDir, "*.wlw", SearchOption.AllDirectories)
        .Concat(Directory.GetFiles(wlDir, "*.wlm", SearchOption.AllDirectories))
        .Concat(Directory.GetFiles(wlDir, "*.wlq", SearchOption.AllDirectories))
        .Concat(Directory.GetFiles(wlDir, "*.wll", SearchOption.AllDirectories))
        .ToList();
    
    Console.WriteLine($"Found {wlFiles.Count} WL* files in {wlDir}");
    
    if (wlFiles.Count == 0)
    {
        Console.WriteLine("No WL* files found.");
        return 0;
    }
    
    // Summary stats
    var tileMap = new Dictionary<(int, int), List<string>>();
    int totalBlocks = 0;
    int parseErrors = 0;
    
    foreach (var wlPath in wlFiles)
    {
        try
        {
            var wl = GillijimProject.WowFiles.Wl.WlFile.Read(wlPath);
            string fileName = Path.GetFileName(wlPath);
            
            Console.WriteLine($"\n[{fileName}]");
            Console.WriteLine($"  Type: {wl.Header.FileType}, Version: {wl.Header.Version}");
            Console.WriteLine($"  Liquid: {wl.Header.LiquidType} (raw: {wl.Header.RawLiquidType})");
            Console.WriteLine($"  Blocks: {wl.Blocks.Count}");
            
            totalBlocks += wl.Blocks.Count;
            
            // Convert to MH2O and check tile mapping
            var converter = new GillijimProject.WowFiles.Wl.WlToMh2oConverter();
            var result = converter.Convert(wl, fileName);
            
            Console.WriteLine($"  Maps to {result.TileData.Count} ADT tiles:");
            foreach (var kvp in result.TileData)
            {
                var (tx, ty) = kvp.Key;
                var tileData = kvp.Value;
                int chunkCount = tileData.ChunkCount;
                Console.WriteLine($"    ({tx},{ty}): {chunkCount} chunks with water");
                
                if (!tileMap.ContainsKey((tx, ty)))
                    tileMap[(tx, ty)] = new List<string>();
                tileMap[(tx, ty)].Add(fileName);
                
                if (verbose)
                {
                    for (int cy = 0; cy < 16; cy++)
                    {
                        for (int cx = 0; cx < 16; cx++)
                        {
                            var chunk = tileData.Chunks[cx, cy];
                            if (chunk != null)
                            {
                                Console.WriteLine($"      Chunk ({cx},{cy}): heights {chunk.MinHeight:F1}-{chunk.MaxHeight:F1}, type={chunk.LiquidTypeId}");
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {Path.GetFileName(wlPath)}: {ex.Message}");
            parseErrors++;
        }
    }
    
    // Write summary
    Console.WriteLine();
    Console.WriteLine("=== Summary ===");
    Console.WriteLine($"Files processed: {wlFiles.Count - parseErrors}/{wlFiles.Count}");
    Console.WriteLine($"Total liquid blocks: {totalBlocks}");
    Console.WriteLine($"ADT tiles with water: {tileMap.Count}");
    
    // Write tile mapping to CSV
    var csvPath = Path.Combine(outDir, "wl_tile_mapping.csv");
    using (var writer = new StreamWriter(csvPath))
    {
        writer.WriteLine("tile_x,tile_y,wl_files");
        foreach (var kvp in tileMap.OrderBy(k => k.Key.Item1).ThenBy(k => k.Key.Item2))
        {
            var (tx, ty) = kvp.Key;
            writer.WriteLine($"{tx},{ty},\"{string.Join("; ", kvp.Value)}\"");
        }
    }
    Console.WriteLine($"Tile mapping written to: {csvPath}");
    
    return 0;
}

// analyze-pm4 command - run dedicated PM4 analysis tools
static int RunAnalyzePm4(string[] args)
{
    Console.WriteLine("=== PM4 Structure Analysis ===\n");
    
    string? pm4Dir = null;
    string? outputDir = null;
    string? wmoDir = null;
    string? gamePath = null;
    string? listfilePath = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--pm4": pm4Dir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--wmo": wmoDir = args[++i]; break;
            case "--game": gamePath = args[++i]; break;
            case "--listfile": listfilePath = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: analyze-pm4 --pm4 <dir> --out <dir> [--wmo <dir>] [--game <dir> --listfile <file>]");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(pm4Dir) || !Directory.Exists(pm4Dir))
    {
        Console.Error.WriteLine("Error: --pm4 <dir> is required and must exist");
        return 1;
    }
    
    outputDir ??= Path.Combine(Directory.GetCurrentDirectory(), "pm4_analysis");
    Directory.CreateDirectory(outputDir);
    
    Console.WriteLine($"PM4 Source: {pm4Dir}");
    Console.WriteLine($"Output Dir: {outputDir}");
    if (!string.IsNullOrEmpty(wmoDir)) Console.WriteLine($"WMO Library: {wmoDir}");
    if (!string.IsNullOrEmpty(gamePath)) Console.WriteLine($"Game Path: {gamePath} (Listfile: {listfilePath})");
    Console.WriteLine();
    
    var pipeline = new PipelineService();
    
    Console.WriteLine("[1/7] Comprehensive Relationship Analysis...");
    pipeline.ExportComprehensiveRelationshipAnalysis(pm4Dir, outputDir);
    
    Console.WriteLine("[2/7] MPRL Rotation & Flag Data...");
    pipeline.ExportMprlRotationData(pm4Dir, outputDir);
    
    Console.WriteLine("[3/7] CK24 Z-Layer Correlation...");
    pipeline.ExportCk24ZLayerAnalysis(pm4Dir, outputDir);
    
    Console.WriteLine("[4/7] CK24 ObjectId Grouping...");
    pipeline.ExportCk24ObjectIdAnalysis(pm4Dir, outputDir);
    
    Console.WriteLine("[5/7] RefIndex Alternative Hypothesis...");
    pipeline.ExportRefIndexAlternativeAnalysis(pm4Dir, outputDir);
    
    Console.WriteLine("[6/7] Geometric Type Correlation...");
    pipeline.ExportGeometricAnalysis(pm4Dir, outputDir);

    Console.WriteLine("[7/7] Geometric Rotation Analysis...");
    string rotationOut = Path.Combine(outputDir, "rotation_analysis.txt");
    string typeCorrelationOut = Path.Combine(outputDir, "type_flag_correlation.csv");
    pipeline.AnalyzeRotationCandidatesV2(pm4Dir, wmoDir ?? "", rotationOut, typeCorrelationOut, gamePath, listfilePath);
    
    Console.WriteLine("\n[DONE] Analysis complete. Check output directory.");
    return 0;
}

// convert-matches-to-modf command - bridge matches.csv to modf_entries.csv
static int RunConvertMatchesToModf(string[] args)
{
    Console.WriteLine("=== Matches CSV to MODF Conversion ===\n");

    string? matchesPath = null;
    string? listfilePath = null;
    string? outputDir = null;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--matches": matchesPath = args[++i]; break;
            case "--listfile": listfilePath = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: convert-matches-to-modf --matches <csv> --listfile <csv> --out <dir>");
                return 0;
        }
    }

    if (string.IsNullOrEmpty(matchesPath) || string.IsNullOrEmpty(listfilePath) || string.IsNullOrEmpty(outputDir))
    {
        Console.Error.WriteLine("Error: --matches, --listfile and --out are required.");
        return 1;
    }

    Directory.CreateDirectory(outputDir);

    // 1. Load Listfile for Path Resolution (Basename -> FullPath)
    // We want to be careful with duplicates.
    var wmoPathLookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    if (File.Exists(listfilePath))
    {
        Console.WriteLine($"Loading listfile: {listfilePath}");
        foreach (var line in File.ReadLines(listfilePath))
        {
            // Assuming simplified listfile (one path per line) or CSV ID;PATH
            var path = line.Trim();
            if (path.Contains(',')) path = path.Split(',')[1].Trim(); // Handle CSV format if needed
            
            if (path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            {
                var name = Path.GetFileName(path);
                if (!wmoPathLookup.ContainsKey(name))
                    wmoPathLookup[name] = path;
            }
        }
        Console.WriteLine($"Loaded {wmoPathLookup.Count} WMO paths.");
    }

    // 2. Read Matches CSV
    Console.WriteLine($"Reading matches: {matchesPath}");
    var modfEntries = new List<string>();
    var wmoSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    
    // Header for output MODF CSV
    // ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence
    modfEntries.Add("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence");

    using (var reader = new StreamReader(matchesPath))
    {
        // Skip header: PM4_ID,WMO_Name,PosX,PosY,PosZ,RotBox_X,RotBox_Y,RotBox_Z
        var header = reader.ReadLine(); 
        
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;
            var cols = line.Split(',');
            if (cols.Length < 8) continue;
            
            var pm4IdStr = cols[0];
            var wmoName = cols[1];
            
            // Resolve Full Path
            string fullPath = wmoName;
            if (wmoPathLookup.TryGetValue(wmoName, out var p)) fullPath = p;
            else 
            {
                // Fallback: try to construct if known prefix? No, just use name and hope patcher handles it or warn.
                // Or maybe the user provided full path in listfile that matches basename.
                // Let's stick to normalized paths.
                fullPath = "World\\wmo\\" + wmoName; // Generic Fallback
            }
            // Normalize separators
            fullPath = fullPath.Replace('/', '\\');
            
            wmoSet.Add(fullPath);
            
            // Parse coords (ADT Placement Space from PM4)
            float placX = float.Parse(cols[2]);
            float placY = float.Parse(cols[3]); // Height
            float placZ = float.Parse(cols[4]);

            // Convert to Game World Coordinates for inject-modf
            // World X = 17066.666 - Plac Z
            // World Y = 17066.666 - Plac X
            // World Z = Plac Y
            const float MapExtent = 17066.66656f;
            float worldX = MapExtent - placZ;
            float worldY = MapExtent - placX;
            float worldZ = placY;
            
            // Parse Rotations (Pitch, Yaw, Roll)
            float pitch = float.Parse(cols[5]);
            float yaw   = float.Parse(cols[6]);
            float roll  = float.Parse(cols[7]);

            // Fix -0
            if (pitch == -0.0f) pitch = 0.0f;
            if (yaw == -0.0f) yaw = 0.0f;
            if (roll == -0.0f) roll = 0.0f;
            
            // MODF CSV Format:
            // ck24 = 0 (placeholder)
            // wmo_path = fullPath
            // name_id = 0 (will be resolved by inject-modf via mwmo map)
            // unique_id = pm4Id
            // pos/rot = World Coords
            // scale = 1.0
            
            modfEntries.Add($"0,{fullPath},0,{pm4IdStr},{worldX},{worldY},{worldZ},{pitch},{yaw},{roll},1.0,1.0");
        }
    }
    
    // 3. Write mwmo_names.csv
    // ID,Path
    var mwmoList = wmoSet.OrderBy(x => x).ToList();
    var mwmoPath = Path.Combine(outputDir, "mwmo_names.csv");
    using (var sw = new StreamWriter(mwmoPath))
    {
        sw.WriteLine("ID,Path");
        for (int i = 0; i < mwmoList.Count; i++)
        {
            sw.WriteLine($"{i},{mwmoList[i]}");
        }
    }
    
    // 4. Write modf_entries.csv
    // But wait, inject-modf expects name_id to match the index in mwmo_names.csv!
    // We need to re-map name_id in the entries.
    
    var finalModfEntries = new List<string>();
    finalModfEntries.Add(modfEntries[0]); // Header
    
    var wmoToIndex = mwmoList.Select((val, idx) => (val, idx)).ToDictionary(x => x.val, x => x.idx);
    
    for (int i = 1; i < modfEntries.Count; i++) // Skip header in loop
    {
        var raw = modfEntries[i];
        var parts = raw.Split(',');
        var path = parts[1];
        if (wmoToIndex.TryGetValue(path, out int idx))
        {
            // Reconstruct line with correct name_id
            // parts[2] is name_id
            parts[2] = idx.ToString();
            finalModfEntries.Add(string.Join(",", parts));
        }
    }
    
    var modfPath = Path.Combine(outputDir, "modf_entries.csv");
    File.WriteAllLines(modfPath, finalModfEntries);
    
    Console.WriteLine($"Generated {finalModfEntries.Count-1} MODF entries.");
    Console.WriteLine($"referenced {mwmoList.Count} unique WMOs.");
    Console.WriteLine($"Output: {outputDir}");
    
    return 0;
}

// dump-pm4-geometry command
static int RunDumpPm4Geometry(string[] args)
{
    if (args.Length < 2)
    {
        Console.WriteLine("Usage: dump-pm4-geometry <pm4_file_or_dir> <output_dir>");
        return 1;
    }

    string inputPath = args[0];
    string outputDir = args[1];
    var dumper = new WoWRollback.PM4Module.Analysis.Pm4GeometryDumper();

    if (File.Exists(inputPath))
    {
        dumper.Dump(inputPath, outputDir);
    }
    else if (Directory.Exists(inputPath))
    {
        foreach (var file in Directory.GetFiles(inputPath, "*.pm4"))
        {
            dumper.Dump(file, outputDir);
        }
    }
    else
    {
        Console.Error.WriteLine($"Input not found: {inputPath}");
        return 1;
    }
    return 0;
}

// convert-ck24-to-wmo command
static int RunConvertCk24ToWmo(string[] args)
{
    if (args.Length < 2)
    {
        Console.WriteLine("Usage: convert-ck24-to-wmo <pm4_file> <output_dir>");
        return 1;
    }

    string pm4Path = args[0];
    string outputRootDir = args[1];

    if (!File.Exists(pm4Path))
    {
        Console.Error.WriteLine($"File not found: {pm4Path}");
        return 1;
    }

    Console.WriteLine($"Loading PM4: {pm4Path}");
    var pm4 = PM4File.FromFile(pm4Path);
    var writer = new WoWRollback.PM4Module.Analysis.Pm4WmoWriter();
    
    // Create correct folder structure: World/wmo/pm4/
    string wmoSubDir = Path.Combine("World", "wmo", "pm4");
    string wmoOutputDir = Path.Combine(outputRootDir, wmoSubDir);
    Directory.CreateDirectory(wmoOutputDir);

    // Prepare CSV for injection
    var csvEntries = new List<string>();
    // Header format compatible with inject-modf (similar to modf_entries.csv)
    // ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale
    csvEntries.Add("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale");

    var groups = pm4.Surfaces
        .GroupBy(s => s.CK24)
        .Where(g => g.Key != 0) 
        .ToList();

    Console.WriteLine($"Found {groups.Count} CK24 objects. Generating WMOs in {wmoOutputDir}...");

    int count = 0;
    uint uniqueIdCounter = 7000000; // Safe range

    foreach (var group in groups)
    {
        uint ck24 = group.Key;
        var surfaces = group.ToList();

        // Collect geometry
        var vertices = new List<Vector3>();
        var indices = new List<int>();
        int vertexOffset = 0;

        foreach (var surf in surfaces)
        {
            // Filter out non-walkable/M2 props to match dumper logic
            if (surf.GroupKey == 0) continue;

            uint startIdx = surf.MsviFirstIndex;
            uint indexCount = surf.IndexCount;

            if (startIdx + indexCount > pm4.MeshIndices.Count) continue;

            for (int i = 0; i < indexCount; i++)
            {
                uint meshIdx = pm4.MeshIndices[(int)(startIdx + i)];
                if (meshIdx >= pm4.MeshVertices.Count) continue;
                
                var v = pm4.MeshVertices[(int)meshIdx];
                vertices.Add(v);
                indices.Add(vertexOffset++);
            }
        }

        if (vertices.Count > 0)
        {
            string wmoName = $"ck24_{ck24:X6}";
            // Write WMO and get centroid (World Position)
            Vector3 centroid = writer.WriteWmo(wmoOutputDir, wmoName, vertices, indices);
            
            // Generate CSV entry
            // Path relative to game root? Usually "World\wmo\pm4\..."
            string wmoGamePath = Path.Combine(wmoSubDir, $"{wmoName}.wmo").Replace('/', '\\');
            
            // NameID 0 (will be resolved by injector if mwmo_names.csv is used, or we just generate unique IDs)
            // Injector requires name_id.
            // We'll treat name_id as index 0 for now? No, unique per WMO.
            // Actually, we'll need to generate mwmo_names.csv too if we want robust injection.
            // But let's just output raw data first.
            
            // CSV: ck24, wmo_path, name_id (0), unique_id, x, y, z, rot...
            // Centroid is standard WoW coords (X, Y, Z).
            string line = $"{ck24:X6},{wmoGamePath},0,{uniqueIdCounter++},{centroid.X:F4},{centroid.Y:F4},{centroid.Z:F4},0,0,0,1";
            csvEntries.Add(line);

            count++;
            if (count % 100 == 0) Console.Write(".");
        }
    }

    string csvPath = Path.Combine(outputRootDir, "generated_wmo_placements.csv");
    File.WriteAllLines(csvPath, csvEntries);

    Console.WriteLine($"\nGenerated {count} WMOs.");
    Console.WriteLine($"Placements CSV written to: {csvPath}");
    return 0;
}

static int RunAnalyzePm4Scene(string[] args)
{
    if (args.Length < 3)
    {
        Console.WriteLine("Usage: analyze-pm4-scene <pm4_file> <placements_csv> <output_dir>");
        return 1;
    }

    string pm4Path = args[0];
    string csvPath = args[1];
    string outputDir = args[2];

    if (!File.Exists(pm4Path))
    {
        Console.Error.WriteLine($"File not found: {pm4Path}");
        return 1;
    }
    if (!File.Exists(csvPath))
    {
        Console.Error.WriteLine($"File not found: {csvPath}");
        return 1;
    }

    var analyzer = new WoWRollback.PM4Module.Analysis.Pm4SceneAnalyzer();
    analyzer.Analyze(pm4Path, csvPath, outputDir);
    return 0;
}

// analyze-m2-library command
static int RunAnalyzeM2Library(string[] args)
{
    Console.WriteLine("=== M2 Library Builder ===\n");
    
    string? m2Dir = null;
    string? mpqArchive = null;
    string? outDir = null;
    string? listfilePath = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--m2": m2Dir = args[++i]; break;
            case "--mpq": mpqArchive = args[++i]; break;
            case "--out": outDir = args[++i]; break;
            case "--listfile": listfilePath = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: analyze-m2-library [--m2 <dir> | --mpq <archive>] --out <output_dir> [--listfile <listfile.csv>]");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(outDir))
    {
        Console.Error.WriteLine("Error: --out is required.");
        return 1;
    }

    if (string.IsNullOrEmpty(m2Dir) && string.IsNullOrEmpty(mpqArchive))
    {
         Console.Error.WriteLine("Error: Either --m2 <dir> or --mpq <archive> must be specified.");
         return 1;
    }
    
    if (!string.IsNullOrEmpty(m2Dir) && !Directory.Exists(m2Dir))
    {
        Console.Error.WriteLine($"Error: M2 directory not found: {m2Dir}");
        return 1;
    }

    if (!string.IsNullOrEmpty(mpqArchive) && !File.Exists(mpqArchive))
    {
         Console.Error.WriteLine($"Error: MPQ archive not found: {mpqArchive}");
         return 1;
    }

    if (string.IsNullOrEmpty(listfilePath) && File.Exists("listfile.csv"))
    {
         listfilePath = "listfile.csv";
         Console.WriteLine($"[INFO] Using default listfile: {listfilePath}");
    }
    
    Directory.CreateDirectory(outDir!);
    var cachePath = Path.Combine(outDir!, "m2_library_cache.json");
    
    var builder = new M2LibraryBuilder();
    Dictionary<string, M2Reference> library;

    if (!string.IsNullOrEmpty(mpqArchive))
    {
        library = builder.BuildLibraryFromMpq(mpqArchive, listfilePath ?? "", cachePath);
    }
    else
    {
        library = builder.BuildLibrary(m2Dir!, listfilePath ?? "", cachePath);
    }
    
    Console.WriteLine($"\n[SUCCESS] Library built with {library.Count} entries.");
    Console.WriteLine($"Cache saved to: {cachePath}");
    
    return 0;
}

// reconstruct-mddf command (M2 matching)
static int RunReconstructMddf(string[] args)
{
    Console.WriteLine("=== MDDF Reconstruction (M2 Matching) ===\n");
    
    string? pm4FacesDir = null;
    string? m2LibraryPath = null;
    string? outCsv = null;
    float minConfidence = 0.7f;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--faces": pm4FacesDir = args[++i]; break;
            case "--library": m2LibraryPath = args[++i]; break;
            case "--out": outCsv = args[++i]; break;
            case "--confidence": float.TryParse(args[++i], out minConfidence); break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: reconstruct-mddf --faces <dir> --library <m2_library.json> --out <output.csv> [--confidence 0.7]");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(pm4FacesDir) || string.IsNullOrEmpty(m2LibraryPath) || string.IsNullOrEmpty(outCsv))
    {
        Console.Error.WriteLine("Error: --faces, --library, and --out are required.");
        return 1;
    }
    
    if (!Directory.Exists(pm4FacesDir))
    {
        Console.Error.WriteLine($"Error: PM4 faces directory not found: {pm4FacesDir}");
        return 1;
    }
    
    if (!File.Exists(m2LibraryPath))
    {
        Console.Error.WriteLine($"Error: M2 library not found: {m2LibraryPath}");
        return 1;
    }
    
    var reconstructor = new Pm4ModfReconstructor();
    
    // Load library
    Console.WriteLine($"Loading M2 Library from {m2LibraryPath}...");
    var library = reconstructor.LoadM2Library(m2LibraryPath);
    if (library.Count == 0)
    {
        Console.Error.WriteLine("Error: M2 library is empty or failed to load.");
        return 1;
    }
    
    // Load PM4 objects
    List<Pm4ModfReconstructor.Pm4Object> objects;
    
    // Method 1: Load from extracted CSV (legacy)
    if (File.Exists(Path.Combine(pm4FacesDir!, "ck_instances.csv")))
    {
        Console.WriteLine($"Loading PM4 objects from CSV in {pm4FacesDir}...");
        objects = reconstructor.LoadPm4Objects(pm4FacesDir!);
    }
    // Method 2: Load directly from .pm4 files (native)
    else
    {
        Console.WriteLine($"Loading PM4 files directly from {pm4FacesDir}...");
        // Ensure PipelineService is accessible. It's in the same project/namespace usually.
        objects = new PipelineService().LoadPm4ObjectsFromFiles(pm4FacesDir!);
    }

    if (objects.Count == 0)
    {
        Console.Error.WriteLine("Error: No PM4 objects found.");
        return 1;
    }
    
    // Perform reconstruction
    Console.WriteLine($"Starting matching (Min Conf: {minConfidence:P0})...");
    var result = reconstructor.ReconstructMddf(objects, library, minConfidence);
    
    // Export
    Directory.CreateDirectory(Path.GetDirectoryName(outCsv)!);
    reconstructor.ExportMddfToCsv(result, outCsv);
    
    Console.WriteLine("\nMDDF reconstruction complete.");
    return 0;
}

// pm4-pipeline-v2 command - Clean PM4 to ADT pipeline using new modular architecture
static int RunPm4PipelineV2(string[] args)
{
    Console.WriteLine("=== PM4 Pipeline V2 (Clean Architecture) ===\n");
    
    string? pm4Dir = null;
    string? outputDir = null;
    string? wmoLibraryPath = null;
    string? museumAdtDir = null;
    string? singleTile = null;
    float sizeTolerance = 0.15f;
    bool exportCsv = true;
    bool dryRun = false;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--pm4": pm4Dir = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--wmo-library": wmoLibraryPath = args[++i]; break;
            case "--museum": museumAdtDir = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--tolerance": sizeTolerance = float.Parse(args[++i]); break;
            case "--no-csv": exportCsv = false; break;
            case "--dry-run": dryRun = true; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: pm4-pipeline-v2 --pm4 <dir> --out <dir> [options]");
                Console.WriteLine();
                Console.WriteLine("Required:");
                Console.WriteLine("  --pm4             Directory containing .pm4 files");
                Console.WriteLine("  --out             Output directory for patched ADTs");
                Console.WriteLine();
                Console.WriteLine("Optional:");
                Console.WriteLine("  --wmo-library     Path to WMO library JSON cache");
                Console.WriteLine("  --museum          Directory containing museum ADTs to patch");
                Console.WriteLine("  --tile X_Y        Process only a single tile (e.g., '22_18')");
                Console.WriteLine("  --tolerance       Size matching tolerance (default: 0.15)");
                Console.WriteLine("  --no-csv          Skip CSV export");
                Console.WriteLine("  --dry-run         Analyze only, don't patch ADTs");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(pm4Dir) || string.IsNullOrEmpty(outputDir))
    {
        Console.Error.WriteLine("Error: --pm4 and --out are required. Use --help for usage.");
        return 1;
    }
    
    if (!Directory.Exists(pm4Dir))
    {
        Console.Error.WriteLine($"Error: PM4 directory not found: {pm4Dir}");
        return 1;
    }
    
    // Create configuration
    var config = new WoWRollback.PM4Module.Pipeline.PipelineConfig(
        Pm4Directory: pm4Dir,
        OutputDirectory: outputDir,
        WmoLibraryPath: wmoLibraryPath,
        MuseumAdtDirectory: museumAdtDir,
        EnableM2Matching: false,  // M2 matching disabled for now
        SizeTolerance: sizeTolerance,
        SingleTile: singleTile,
        ExportCsv: exportCsv,
        DryRun: dryRun
    );
    
    // Execute pipeline
    var orchestrator = new WoWRollback.PM4Module.Pipeline.Pm4PipelineOrchestrator();
    var result = orchestrator.Execute(config);
    
    // Summary
    if (result.Errors.Count > 0)
    {
        Console.WriteLine("\n[ERRORS]");
        foreach (var error in result.Errors)
        {
            Console.Error.WriteLine($"  - {error}");
        }
    }
    
    return result.FailedTiles > 0 ? 1 : 0;
}

// export-pm4-obj command - export PM4 candidates as OBJ files for verification
static int RunExportPm4Obj(string[] args)
{
    Console.WriteLine("=== PM4 OBJ Export (Decoder Verification) ===\n");
    
    string? pm4Path = null;
    string? outputDir = null;
    string? singleTile = null;
    
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--pm4": pm4Path = args[++i]; break;
            case "--out": outputDir = args[++i]; break;
            case "--tile": singleTile = args[++i]; break;
            case "--help":
            case "-h":
                Console.WriteLine("Usage: export-pm4-obj --pm4 <pm4_dir_or_file> --out <output_dir> [--tile X_Y]");
                Console.WriteLine();
                Console.WriteLine("  --pm4 <path>   PM4 file or directory containing PM4 files");
                Console.WriteLine("  --out <dir>    Output directory for OBJ files");
                Console.WriteLine("  --tile X_Y     Optional: process only a single tile");
                return 0;
        }
    }
    
    if (string.IsNullOrEmpty(pm4Path))
    {
        Console.Error.WriteLine("Error: --pm4 is required. Use --help for usage.");
        return 1;
    }
    
    outputDir ??= Path.Combine(Path.GetDirectoryName(pm4Path) ?? ".", "pm4_obj_export");
    Directory.CreateDirectory(outputDir);
    
    var extractor = new WoWRollback.PM4Module.Pipeline.Pm4ObjectExtractor();
    IEnumerable<WoWRollback.PM4Module.Pipeline.Pm4WmoCandidate> candidates;
    
    if (File.Exists(pm4Path))
    {
        // Single file
        Console.WriteLine($"Extracting from: {pm4Path}");
        candidates = extractor.ExtractWmoCandidates(pm4Path);
    }
    else if (Directory.Exists(pm4Path))
    {
        // Directory
        Console.WriteLine($"Extracting from directory: {pm4Path}");
        candidates = extractor.ExtractAllWmoCandidates(pm4Path);
    }
    else
    {
        Console.Error.WriteLine($"Error: Path not found: {pm4Path}");
        return 1;
    }
    
    // Filter by tile if specified
    if (!string.IsNullOrEmpty(singleTile))
    {
        var parts = singleTile.Split('_');
        if (parts.Length == 2 && int.TryParse(parts[0], out int tx) && int.TryParse(parts[1], out int ty))
        {
            candidates = candidates.Where(c => c.TileX == tx && c.TileY == ty);
            Console.WriteLine($"Filtering to tile: {singleTile}");
        }
    }
    
    WoWRollback.PM4Module.Pipeline.Pm4ObjectExtractor.ExportCandidatesToObj(candidates, outputDir);
    
    Console.WriteLine($"\nOutput directory: {outputDir}");
    Console.WriteLine("Compare these OBJ files with Pm4Reader exports to verify decoder correctness.");
    
    return 0;
}
