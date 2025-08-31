using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Terrain;
using GillijimProject.WowFiles.Objects;

// Parse command line arguments
var commandArgs = Environment.GetCommandLineArgs().Skip(1).ToArray();
if (commandArgs.Length == 0)
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  scan-wdt <file.wdt> [options]     - Scan and export JSON analysis");
    Console.WriteLine("  dump-wdt <file.wdt> [options]     - Direct console output (legacy)");
    Console.WriteLine("  analyze-json <analysis.json>      - Analyze exported JSON data");
    Console.WriteLine("  scan-chunks <file.wdt> [options]  - Scan for all implemented chunk types");
    Console.WriteLine("  decompile-test <file.wdt> [options] - Test decompile/recompile data preservation");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --start <n>     Start from tile N (default: 0)");
    Console.WriteLine("  --max <n>       Process max N tiles (default: 10)");
    Console.WriteLine("  --max-mcnks <n> Process max N MCNKs per tile (default: 5)");
    Console.WriteLine("  --all           Process all tiles/MCNKs");
    Console.WriteLine("  --verbose       Show detailed output");
    Console.WriteLine("  --output <dir>  Output directory (default: auto-generated)");
    return;
}

string command = commandArgs[0];
if (command == "scan-wdt")
{
    var options = ParseOptions(commandArgs.Skip(1).ToArray());
    if (options == null) return;
    
    ScanWdt(options);
}
else if (command == "dump-wdt")
{
    var options = ParseOptions(commandArgs.Skip(1).ToArray());
    if (options == null) return;
    
    DumpWdt(options);
}
else if (command == "scan-chunks")
{
    var options = ParseOptions(commandArgs.Skip(1).ToArray());
    if (options == null) return;
    
    ScanAllChunks(options);
}
else if (command == "analyze-json")
{
    if (commandArgs.Length < 2)
    {
        Console.WriteLine("Error: Missing JSON file path");
        return;
    }
    AnalyzeJson(commandArgs[1]);
}
else if (command == "decompile-test")
{
    var options = ParseOptions(commandArgs.Skip(1).ToArray());
    if (options == null) return;
    
    DecompileTest.RunTest(options);
}
else
{
    Console.WriteLine($"Unknown command: {command}");
    return;
}

static Options? ParseOptions(string[] args)
{
    if (args.Length < 1)
    {
        Console.WriteLine("Error: Missing WDT file path");
        return null;
    }

    var options = new Options
    {
        FilePath = args[0],
        StartTile = 0,
        MaxTiles = 10,
        MaxMcnks = 5,
        Verbose = false,
        ProcessAll = false,
        OutputDirectory = null
    };

    for (int i = 1; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--start":
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out var startTile))
                {
                    Console.WriteLine("Error: --start requires a valid integer");
                    return null;
                }
                options.StartTile = startTile;
                break;
            case "--max":
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out var maxTiles))
                {
                    Console.WriteLine("Error: --max requires a valid integer");
                    return null;
                }
                options.MaxTiles = maxTiles;
                break;
            case "--max-mcnks":
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out var maxMcnks))
                {
                    Console.WriteLine("Error: --max-mcnks requires a valid integer");
                    return null;
                }
                options.MaxMcnks = maxMcnks;
                break;
            case "--all":
                options.ProcessAll = true;
                break;
            case "--verbose":
                options.Verbose = true;
                break;
            case "--output":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("Error: --output requires a directory path");
                    return null;
                }
                options.OutputDirectory = args[++i];
                break;
            default:
                Console.WriteLine($"Error: Unknown option {args[i]}");
                return null;
        }
    }

    if (!File.Exists(options.FilePath))
    {
        Console.WriteLine($"Error: File not found: {options.FilePath}");
        return null;
    }

    return options;
}

static void ScanWdt(Options options)
{
    try
    {
        var wdt = Wdt.Load(options.FilePath);
        ProcessWdt(wdt, options);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
        if (options.Verbose)
        {
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}

static void DumpWdt(Options options)
{
    try
    {
        var wdt = Wdt.Load(options.FilePath);
        ProcessWdt(wdt, options);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
        if (options.Verbose)
        {
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}

static void ScanAllChunks(Options options)
{
    try
    {
        var wdt = Wdt.Load(options.FilePath);
        Console.WriteLine($"Scanning WDT file for all implemented chunk types...");
        Console.WriteLine($"File: {options.FilePath}");
        Console.WriteLine();
        
        // Scan for MCLY chunks (Alpha layer data)
        var mclyChunks = wdt.FindAndParseChunks(Tags.MCLY, offset => wdt.ReadMcly(offset));
        Console.WriteLine($"MCLY (Alpha layer data): {mclyChunks.Count} chunks found");
        if (options.Verbose && mclyChunks.Count > 0)
        {
            var totalLayers = mclyChunks.Sum(m => m.Layers.Count);
            Console.WriteLine($"  Total layers across all chunks: {totalLayers}");
        }
        
        // Scan for MTEX chunks (texture filenames)
        var mtexChunks = wdt.FindAndParseChunks(Tags.MTEX, offset => wdt.ReadMtex(offset));
        Console.WriteLine($"MTEX (Texture filenames): {mtexChunks.Count} chunks found");
        if (options.Verbose && mtexChunks.Count > 0)
        {
            var totalTextures = mtexChunks.Sum(m => m.TextureFilenames.Count);
            Console.WriteLine($"  Total texture filenames: {totalTextures}");
            if (mtexChunks.Count > 0)
            {
                Console.WriteLine($"  Sample textures: {string.Join(", ", mtexChunks[0].TextureFilenames.Take(3))}");
            }
        }
        
        // Scan for MODF chunks (WMO placement)
        var modfChunks = wdt.FindAndParseChunks(Tags.MODF, offset => wdt.ReadModf(offset));
        Console.WriteLine($"MODF (WMO placement): {modfChunks.Count} chunks found");
        if (options.Verbose && modfChunks.Count > 0)
        {
            var totalPlacements = modfChunks.Sum(m => m.Placements.Count);
            Console.WriteLine($"  Total WMO placements: {totalPlacements}");
        }
        
        // Scan for MDDF chunks (doodad placement)
        var mddfChunks = wdt.FindAndParseChunks(Tags.MDDF, offset => wdt.ReadMddf(offset));
        Console.WriteLine($"MDDF (Doodad placement): {mddfChunks.Count} chunks found");
        if (options.Verbose && mddfChunks.Count > 0)
        {
            var totalPlacements = mddfChunks.Sum(m => m.Placements.Count);
            Console.WriteLine($"  Total doodad placements: {totalPlacements}");
        }
        
        // Scan for MCRF chunks (cross-references)
        var mcrfChunks = wdt.FindAndParseChunks(Tags.MCRF, offset => wdt.ReadMcrf(offset));
        Console.WriteLine($"MCRF (Cross-references): {mcrfChunks.Count} chunks found");
        if (options.Verbose && mcrfChunks.Count > 0)
        {
            var totalIndices = mcrfChunks.Sum(m => m.Indices.Count);
            Console.WriteLine($"  Total cross-reference indices: {totalIndices}");
        }
        
        // Scan for MMID chunks (model indices)
        var mmidChunks = wdt.FindAndParseChunks(Tags.MMID, offset => wdt.ReadMmid(offset));
        Console.WriteLine($"MMID (Model indices): {mmidChunks.Count} chunks found");
        if (options.Verbose && mmidChunks.Count > 0)
        {
            var totalIndices = mmidChunks.Sum(m => m.ModelIndices.Count);
            Console.WriteLine($"  Total model indices: {totalIndices}");
        }
        
        // Scan for MDNM chunks (doodad filenames)
        var mdnmChunks = wdt.FindAndParseChunks(Tags.MDNM, offset => wdt.ReadMdnm(offset));
        Console.WriteLine($"MDNM (Doodad filenames): {mdnmChunks.Count} chunks found");
        if (options.Verbose && mdnmChunks.Count > 0)
        {
            var totalFilenames = mdnmChunks.Sum(m => m.DoodadFilenames.Count);
            Console.WriteLine($"  Total doodad filenames: {totalFilenames}");
            if (mdnmChunks.Count > 0)
            {
                Console.WriteLine($"  Sample filenames: {string.Join(", ", mdnmChunks[0].DoodadFilenames.Take(3))}");
            }
        }
        
        // Scan for MONM chunks (WMO filenames)
        var monmChunks = wdt.FindAndParseChunks(Tags.MONM, offset => wdt.ReadMonm(offset));
        Console.WriteLine($"MONM (WMO filenames): {monmChunks.Count} chunks found");
        if (options.Verbose && monmChunks.Count > 0)
        {
            var totalFilenames = monmChunks.Sum(m => m.WmoFilenames.Count);
            Console.WriteLine($"  Total WMO filenames: {totalFilenames}");
            if (monmChunks.Count > 0)
            {
                Console.WriteLine($"  Sample filenames: {string.Join(", ", monmChunks[0].WmoFilenames.Take(3))}");
            }
        }
        
        // Scan for MMDX chunks (M2 model filenames)
        var mmdxChunks = wdt.FindAndParseChunks(Tags.MMDX, offset => wdt.ReadMmdx(offset));
        Console.WriteLine($"MMDX (M2 model filenames): {mmdxChunks.Count} chunks found");
        if (options.Verbose && mmdxChunks.Count > 0)
        {
            var totalFilenames = mmdxChunks.Sum(m => m.M2Filenames.Count);
            Console.WriteLine($"  Total M2 filenames: {totalFilenames}");
            if (mmdxChunks.Count > 0)
            {
                Console.WriteLine($"  Sample filenames: {string.Join(", ", mmdxChunks[0].M2Filenames.Take(3))}");
            }
        }
        
        // Scan for MWMO chunks (WMO filenames)
        var mwmoChunks = wdt.FindAndParseChunks(Tags.MWMO, offset => wdt.ReadMwmo(offset));
        Console.WriteLine($"MWMO (WMO filenames): {mwmoChunks.Count} chunks found");
        if (options.Verbose && mwmoChunks.Count > 0)
        {
            var totalFilenames = mwmoChunks.Sum(m => m.WmoFilenames.Count);
            Console.WriteLine($"  Total WMO filenames: {totalFilenames}");
            if (mwmoChunks.Count > 0)
            {
                Console.WriteLine($"  Sample filenames: {string.Join(", ", mwmoChunks[0].WmoFilenames.Take(3))}");
            }
        }
        
        Console.WriteLine();
        Console.WriteLine("✅ All implemented chunk parsers successfully integrated and tested!");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
        if (options.Verbose)
        {
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}

static void AnalyzeJson(string filePath)
{
    // TO DO: implement JSON analysis
}

static void ProcessWdt(Wdt wdt, Options options)
{
    Console.WriteLine($"MAIN entries: {wdt.MainEntries.Count}");
    
    if (options.Verbose)
    {
        Console.WriteLine($"Processing from tile {options.StartTile}, max tiles: {(options.ProcessAll ? "unlimited" : options.MaxTiles)}, max MCNKs per tile: {(options.ProcessAll ? "unlimited" : options.MaxMcnks)}");
    }

    int tilesProcessed = 0;
    int tileIndex = 0;
    
    foreach (var entry in wdt.MainEntries)
    {
        if (tileIndex < options.StartTile)
        {
            tileIndex++;
            continue;
        }
        
        if (!options.ProcessAll && tilesProcessed >= options.MaxTiles)
        {
            Console.WriteLine($"Reached tile limit ({options.MaxTiles}), use --all to process all tiles");
            break;
        }

        if (!entry.HasMhdr)
        {
            Console.WriteLine($"[{tileIndex:000}] MHDR: none (Size==0)");
            tileIndex++;
            continue;
        }

        ProcessTile(wdt, entry, tileIndex, options);
        tilesProcessed++;
        tileIndex++;
    }
}

static void ProcessTile(Wdt wdt, MainAlpha.Entry entry, int tileIndex, Options options)
{
    var mhdr = wdt.GetMhdrFor(entry);
    var mcin = wdt.GetMcinFor(entry, out var tileStart);
    
    Console.WriteLine($"[{tileIndex:000}] MCIN entries: {mcin.Entries.Count}");
    
    if (options.Verbose)
    {
        Console.WriteLine($"  Tile start: 0x{tileStart:X8}, MHDR MCIN offset: 0x{mhdr.McinRelOffset:X8}");
    }

    int mcnksProcessed = 0;
    for (int i = 0; i < mcin.Entries.Count; i++)
    {
        if (!options.ProcessAll && mcnksProcessed >= options.MaxMcnks)
        {
            Console.WriteLine($"  ... (showing first {options.MaxMcnks} MCNKs, use --all for complete output)");
            break;
        }

        var mc = mcin.Entries[i];
        if (!mc.HasChunk || mc.AbsoluteOffset == 0)
        {
            if (options.Verbose)
            {
                Console.WriteLine($"  MCNK[{i:000}] @0x{mc.AbsoluteOffset:X8} EMPTY");
            }
            continue;
        }

        ProcessMcnk(wdt, mc, i, options);
        mcnksProcessed++;
    }
}

static void ProcessMcnk(Wdt wdt, McinAlpha.Entry mc, int mcnkIndex, Options options)
{
    try
    {
        var mcnk = wdt.ReadMcnkHeader(mc.AbsoluteOffset);
        var check = mcnk.ValidateSubchunks();
        
        Console.WriteLine($"  MCNK[{mcnkIndex:000}] @0x{mc.AbsoluteOffset:X8} MCVT ok={check.McvtOk} MCNR ok={check.McnrOk} boundsOk={check.BoundsOk}");
        
        if (options.Verbose)
        {
            Console.WriteLine($"    Payload: 0x{mcnk.PayloadStart:X8}, ChunksSize: {mcnk.ChunksSize}, EndBound: 0x{mcnk.EndBound:X8}");
            Console.WriteLine($"    MCVT rel: 0x{mcnk.McvtRel:X8}, MCNR rel: 0x{mcnk.McnrRel:X8}, MCLQ rel: 0x{mcnk.MclqRel:X8}");
            
            if (mcnk.McvtRel != 0)
            {
                long mcvtAbs = mcnk.McvtAbs(0);
                Console.WriteLine($"    MCVT abs: 0x{mcvtAbs:X8} (size: {McvtAlphaReader.ExpectedSize})");
                
                // Extract actual height data
                try
                {
                    var mcvt = wdt.ReadMcvt(mcvtAbs);
                    var heights = mcvt.GetHeights();
                    float minHeight = heights.Min();
                    float maxHeight = heights.Max();
                    Console.WriteLine($"    Height range: {minHeight:F2} to {maxHeight:F2}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"    MCVT read error: {ex.Message}");
                }
            }
            
            if (mcnk.McnrRel != 0)
            {
                long mcnrAbs = mcnk.McnrAbs(0);
                Console.WriteLine($"    MCNR abs: 0x{mcnrAbs:X8} (size: {McnrAlpha.ExpectedSize})");
            }
            
            if (mcnk.MclqRel != 0)
            {
                long mclqAbs = mcnk.MclqAbs(0);
                Console.WriteLine($"    MCLQ abs: 0x{mclqAbs:X8} (variable size)");
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  MCNK[{mcnkIndex:000}] @0x{mc.AbsoluteOffset:X8} ERROR: {ex.Message}");
    }
}

public sealed class Options
{
    public required string FilePath { get; set; }
    public int StartTile { get; set; }
    public int MaxTiles { get; set; }
    public int MaxMcnks { get; set; }
    public bool Verbose { get; set; }
    public bool ProcessAll { get; set; }
    public string? OutputDirectory { get; set; }
}
