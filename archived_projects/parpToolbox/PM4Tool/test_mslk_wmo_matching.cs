using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

namespace PM4Tool
{
    /// <summary>
    /// Demonstrates MSLK scene graph-based WMO matching
    /// </summary>
    class TestMslkWmoMatching
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🎯 MSLK Scene Graph → WMO Matching Test");
            Console.WriteLine("=====================================");
            
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: test_mslk_wmo_matching.exe <pm4_file>");
                Console.WriteLine("Example: test_mslk_wmo_matching.exe development/some_file.pm4");
                return;
            }
            
            var pm4Path = args[0];
            if (!File.Exists(pm4Path))
            {
                Console.WriteLine($"❌ PM4 file not found: {pm4Path}");
                return;
            }
            
            try
            {
                Console.WriteLine($"📂 Loading PM4: {Path.GetFileName(pm4Path)}");
                var pm4File = PM4File.FromFile(pm4Path);
                
                if (pm4File.MSLK?.Entries == null)
                {
                    Console.WriteLine("❌ No MSLK data found in this PM4 file");
                    return;
                }
                
                Console.WriteLine($"✅ Found {pm4File.MSLK.Entries.Count} MSLK entries");
                
                // Perform MSLK hierarchy analysis
                var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
                var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);
                var objectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);
                
                Console.WriteLine($"🔍 Analysis Results:");
                Console.WriteLine($"   Total Nodes: {hierarchyResult.AllNodes.Count}");
                Console.WriteLine($"   Root Nodes: {hierarchyResult.RootNodes.Count}");
                Console.WriteLine($"   Scene Objects: {objectSegments.Count}");
                Console.WriteLine($"   Max Depth: {hierarchyResult.MaxDepth}");
                Console.WriteLine();
                
                // Export individual objects for WMO matching
                var outputDir = Path.Combine("output", "mslk_objects");
                Directory.CreateDirectory(outputDir);
                
                var objectMeshExporter = new MslkObjectMeshExporter();
                var baseFileName = Path.GetFileNameWithoutExtension(pm4Path);
                
                Console.WriteLine($"🎯 Exporting {objectSegments.Count} MSLK objects for WMO matching...");
                Console.WriteLine();
                
                var exportedObjects = 0;
                foreach (var obj in objectSegments)
                {
                    try
                    {
                        var objFileName = $"{baseFileName}.obj_{obj.RootIndex:D3}.obj";
                        var objPath = Path.Combine(outputDir, objFileName);
                        
                        // Export with render-mesh-only for clean geometry
                        objectMeshExporter.ExportObjectMesh(obj, pm4File, objPath, renderMeshOnly: true);
                        
                        Console.WriteLine($"  ✅ Object {obj.RootIndex}: {obj.GeometryNodeIndices.Count} geometry nodes → {objFileName}");
                        exportedObjects++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  ❌ Failed to export object {obj.RootIndex}: {ex.Message}");
                    }
                }
                
                Console.WriteLine();
                Console.WriteLine($"📊 Export Summary:");
                Console.WriteLine($"   Objects Exported: {exportedObjects}/{objectSegments.Count}");
                Console.WriteLine($"   Output Directory: {outputDir}");
                Console.WriteLine();
                Console.WriteLine($"🔄 Next Steps for WMO Matching:");
                Console.WriteLine($"   1. Run PM4WmoMatcher.exe with --use-mslk-objects flag");
                Console.WriteLine($"   2. Compare individual MSLK objects to WMO assets");
                Console.WriteLine($"   3. Use Modified Hausdorff Distance for geometric matching");
                Console.WriteLine($"   4. Generate visualization pairs for manual inspection");
                Console.WriteLine();
                Console.WriteLine($"💡 Example Command:");
                Console.WriteLine($"   PM4WmoMatcher.exe --pm4 \"{pm4Path}\" --wmo \"path/to/wmo/files\" --output results --use-mslk-objects --visualize");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"   Inner: {ex.InnerException.Message}");
                }
            }
        }
    }
} 