using System;
using System.IO;
using System.Linq;

namespace PM4Rebuilder;

/// <summary>
/// Bulk direct PM4 exporter that runs the working DirectPm4Exporter on all PM4 files in a directory.
/// This provides a reliable bulk export solution using the proven direct-export pipeline.
/// Includes deduplication to prevent excessive disk usage from duplicate geometry.
/// </summary>
public static class BulkDirectExporter
{
    /// <summary>
    /// Export all PM4 files in a directory using the working direct-export approach.
    /// </summary>
    /// <param name="pm4Directory">Directory containing PM4 files</param>
    /// <param name="outputDirectory">Base output directory for all exports</param>
    /// <returns>0 on success, 1 on error</returns>
    public static int ExportAllBuildings(string pm4Directory, string outputDirectory)
    {
        try
        {
            Console.WriteLine($"[BULK EXPORTER] Starting bulk PM4 → OBJ export from: {pm4Directory}");
            Console.WriteLine($"[BULK EXPORTER] Output directory: {outputDirectory}");
            
            if (!Directory.Exists(pm4Directory))
            {
                Console.WriteLine($"[BULK EXPORTER ERROR] Directory not found: {pm4Directory}");
                return 1;
            }
            
            Directory.CreateDirectory(outputDirectory);
            
            // Find all PM4 files in the directory
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.TopDirectoryOnly);
            
            if (pm4Files.Length == 0)
            {
                Console.WriteLine($"[BULK EXPORTER WARNING] No PM4 files found in: {pm4Directory}");
                return 0;
            }
            
            Console.WriteLine($"[BULK EXPORTER] Found {pm4Files.Length} PM4 files to process");
            
            int successCount = 0;
            int failureCount = 0;
            
            foreach (var pm4File in pm4Files.OrderBy(f => f))
            {
                var fileName = Path.GetFileNameWithoutExtension(pm4File);
                var tileOutputDir = Path.Combine(outputDirectory, fileName);
                
                Console.WriteLine($"[BULK EXPORTER] Processing {fileName}...");
                
                try
                {
                    Directory.CreateDirectory(tileOutputDir);
                    int result = DirectPm4Exporter.ExportBuildings(pm4File, tileOutputDir);
                    
                    if (result == 0)
                    {
                        successCount++;
                        Console.WriteLine($"[BULK EXPORTER] ✅ {fileName} exported successfully");
                    }
                    else
                    {
                        failureCount++;
                        Console.WriteLine($"[BULK EXPORTER] ❌ {fileName} export failed");
                    }
                }
                catch (Exception ex)
                {
                    failureCount++;
                    Console.WriteLine($"[BULK EXPORTER] ❌ {fileName} export failed: {ex.Message}");
                }
            }
            
            Console.WriteLine();
            Console.WriteLine($"[BULK EXPORTER] Export complete:");
            Console.WriteLine($"  ✅ Successful: {successCount} files");
            Console.WriteLine($"  ❌ Failed: {failureCount} files");
            Console.WriteLine($"  📁 Output directory: {outputDirectory}");

            // Final pass: remove duplicate OBJ meshes (identical geometry hashes)
            Console.WriteLine("[BULK EXPORTER] Running deduplication pass on exported OBJs…");
            ObjDeduplicator.Deduplicate(outputDirectory, deleteDuplicates: true);
            
            return failureCount > 0 ? 1 : 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[BULK EXPORTER ERROR] Bulk export failed: {ex.Message}");
            return 1;
        }
    }
}
