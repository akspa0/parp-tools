// ADT Merger - Merges split 3.3.5 ADTs into monolithic format
// Creates "clean" LK ADTs preserving all existing object data + WDT
// Also supports WDL to ADT generation for filling gaps

using WoWRollback.PM4Module;
using WoWRollback.Core.Services.PM4;

// Check for wdl-to-adt subcommand
if (args.Length > 0 && args[0] == "wdl-to-adt")
{
    return WdlToAdtProgram.Run(args.Skip(1).ToArray());
}

Console.WriteLine("=== ADT Merger - Clean LK ADT Generation ===\n");

// Paths
var sourceDir = @"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\development\World\Maps\development";
var outputDir = @"j:\wowDev\parp-tools\gillijimproject_refactor\PM4ADTs\clean";
var mapName = "development";

// Create output directory
Directory.CreateDirectory(outputDir);
Console.WriteLine($"Source: {sourceDir}");
Console.WriteLine($"Output: {outputDir}\n");

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
