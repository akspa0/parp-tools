using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

class MslkObjectExportTest
{
    static void Main(string[] args)
    {
        Console.WriteLine("🎯 MSLK OBJECT MESH EXPORT TEST");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Testing per-object mesh export functionality with scene graph segmentation");
        Console.WriteLine();

        if (args.Length > 0)
        {
            // Test specific file if provided
            TestSingleFile(args[0]);
        }
        else
        {
            // Test with development files if available
            TestDevelopmentFiles();
        }

        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }

    static void TestSingleFile(string filePath)
    {
        Console.WriteLine($"🎯 Testing single file: {filePath}");
        Console.WriteLine(new string('─', 60));

        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"❌ File not found: {filePath}");
                return;
            }

            var pm4File = PM4File.FromFile(filePath);
            var fileName = Path.GetFileName(filePath);

            if (pm4File.MSLK?.Entries == null || !pm4File.MSLK.Entries.Any())
            {
                Console.WriteLine("⚠️  No MSLK data found in file");
                return;
            }

            Console.WriteLine($"📊 MSLK Entries: {pm4File.MSLK.Entries.Count}");

            // Perform hierarchy analysis
            var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
            var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);
            var objectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);

            Console.WriteLine($"📊 Analysis Results:");
            Console.WriteLine($"   Total Nodes: {hierarchyResult.AllNodes.Count}");
            Console.WriteLine($"   Root Nodes: {hierarchyResult.RootNodes.Count}");
            Console.WriteLine($"   Objects Found: {objectSegments.Count}");
            Console.WriteLine($"   Max Depth: {hierarchyResult.MaxDepth}");

            // Export per-object meshes
            var outputDir = Path.Combine("output", "object_test");
            var objectExporter = new MslkObjectMeshExporter();
            var baseFileName = Path.GetFileNameWithoutExtension(fileName);

            Console.WriteLine($"\n🎯 Exporting {objectSegments.Count} objects...");
            objectExporter.ExportAllObjects(objectSegments, pm4File, outputDir, baseFileName);

            Console.WriteLine($"\n✅ Test complete! Check output directory: {outputDir}");
            
            // Show sample object details
            Console.WriteLine("\n📋 SAMPLE OBJECT DETAILS:");
            foreach (var obj in objectSegments.Take(3))
            {
                Console.WriteLine($"   Object {obj.RootIndex}: {obj.GeometryNodeIndices.Count} geometry, {obj.DoodadNodeIndices.Count} anchor nodes");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
        }
    }

    static void TestDevelopmentFiles()
    {
        Console.WriteLine("🔍 Looking for development test files...");

        var testPaths = new[]
        {
            "test_data/development/development_00_00.pm4",
            "test_data/development/development_22_18.pm4",
            "test_data/original_development/development_00_00.pm4"
        };

        foreach (var path in testPaths)
        {
            if (File.Exists(path))
            {
                Console.WriteLine($"✅ Found: {path}");
                TestSingleFile(path);
                return;
            }
        }

        Console.WriteLine("❌ No development test files found. Expected files:");
        foreach (var path in testPaths)
        {
            Console.WriteLine($"   - {path}");
        }
        
        Console.WriteLine("\nℹ️  You can specify a PM4 file path as a command line argument.");
        Console.WriteLine("   Example: MslkObjectExportTest.exe \"path/to/your/file.pm4\"");
    }
} 