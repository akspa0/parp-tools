using System;
using System.IO;
using System.Linq;
using WoWRollback.LkToAlphaModule.Services;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.LkToAlphaModule.Readers;

var wdtPath = @"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\0.5.5\tree\World\Maps\Azeroth\Azeroth.wdt";
var testDir = @"J:\wowDev\parp-tools\gillijimproject_refactor\WoWRollback\single_tile_test";

try {
    // Read WDT
    var wdtBytes = File.ReadAllBytes(wdtPath);
    var reader = new AlphaWdtReader(wdtPath, wdtBytes);
    var alphaWdt = reader.Parse();
    
    Console.WriteLine($"WDT has {alphaWdt.Tiles.Count} tiles");
    
    // Get first non-empty tile
    var firstTile = alphaWdt.Tiles.FirstOrDefault(t => t.Size > 0);
    if (firstTile == null) {
        Console.WriteLine("No non-empty tiles found");
        return 1;
    }
    
    Console.WriteLine($"Testing tile {firstTile.Index} at offset {firstTile.Offset}, size {firstTile.Size}");
    
    // Extract tile bytes
    var tileBytes = new byte[firstTile.Size];
    Array.Copy(wdtBytes, firstTile.Offset, tileBytes, 0, firstTile.Size);
    
    var extractedPath = Path.Combine(testDir, "tile_extracted.adt");
    File.WriteAllBytes(extractedPath, tileBytes);
    Console.WriteLine($"Extracted to {extractedPath}");
    
    // Try Alpha→LK conversion
    Console.WriteLine("\\nAttempting Alpha→LK conversion...");
    var lkSource = AlphaDataExtractor.ExtractFromAlphaAdt(extractedPath);
    Console.WriteLine($"Extracted LK source: {lkSource.Mcnks.Count} MCNKs");
    
    var lkBytes = LkAdtBuilder.Build(lkSource, new WoWRollback.LkToAlphaModule.LkToAlphaOptions());
    var lkPath = Path.Combine(testDir, "tile_lk.adt");
    File.WriteAllBytes(lkPath, lkBytes);
    Console.WriteLine($"LK ADT written: {lkBytes.Length} bytes");
    
    Console.WriteLine("\\nSUCCESS!");
    return 0;
}
catch (Exception ex) {
    Console.WriteLine($"\\nERROR: {ex.GetType().Name}");
    Console.WriteLine($"Message: {ex.Message}");
    Console.WriteLine($"\\nStack trace:\\n{ex.StackTrace}");
    if (ex.InnerException != null) {
        Console.WriteLine($"\\nInner exception: {ex.InnerException.Message}");
    }
    return 1;
}
