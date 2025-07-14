using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🎯 MSLK OBJECT EXPORT RUNNER");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Exports per-object OBJ files for all PM4 files in a directory");
        Console.WriteLine();

        if (args.Length > 0)
        {
            // Process specific directory or file
            var path = args[0];
            if (Directory.Exists(path))
            {
                ProcessDirectory(path);
            }
            else if (File.Exists(path))
            {
                ProcessSingleFile(path);
            }
            else
            {
                Console.WriteLine($"❌ Path not found: {path}");
                ShowUsage();
            }
        }
        else
        {
            // Use built-in demo with test data discovery
            Console.WriteLine("🔍 Running built-in demo with automatic test data discovery...");
            Console.WriteLine();
            MslkHierarchyDemo.RunHierarchyAnalysis();
        }

        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }

    static void ProcessDirectory(string directoryPath)
    {
        Console.WriteLine($"📁 Processing directory: {directoryPath}");
        Console.WriteLine();

        var pm4Files = Directory.GetFiles(directoryPath, "*.pm4", SearchOption.AllDirectories);
        
        if (pm4Files.Length == 0)
        {
            Console.WriteLine("❌ No PM4 files found in directory");
            return;
        }

        Console.WriteLine($"🎯 Found {pm4Files.Length} PM4 files to process");
        Console.WriteLine();

        var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
        var objectMeshExporter = new MslkObjectMeshExporter();
        
        int processedCount = 0;
        int errorCount = 0;

        foreach (var filePath in pm4Files)
        {
            try
            {
                ProcessSingleFileInternal(filePath, hierarchyAnalyzer, objectMeshExporter);
                processedCount++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error processing {Path.GetFileName(filePath)}: {ex.Message}");
                errorCount++;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"📊 BATCH PROCESSING COMPLETE:");
        Console.WriteLine($"   ✅ Successfully processed: {processedCount} files");
        Console.WriteLine($"   ❌ Errors: {errorCount} files");
        Console.WriteLine($"   📁 Output directory: output/");
    }

    static void ProcessSingleFile(string filePath)
    {
        Console.WriteLine($"📄 Processing single file: {Path.GetFileName(filePath)}");
        Console.WriteLine();

        var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
        var objectMeshExporter = new MslkObjectMeshExporter();

        try
        {
            ProcessSingleFileInternal(filePath, hierarchyAnalyzer, objectMeshExporter);
            Console.WriteLine("✅ File processed successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
        }
    }

    static void ProcessSingleFileInternal(string filePath, MslkHierarchyAnalyzer hierarchyAnalyzer, MslkObjectMeshExporter objectMeshExporter)
    {
        var fileName = Path.GetFileName(filePath);
        var baseFileName = Path.GetFileNameWithoutExtension(fileName);
        
        Console.WriteLine($"📄 PROCESSING: {fileName}");
        Console.WriteLine(new string('─', 60));

        // Load PM4 file
        var pm4File = PM4File.FromFile(filePath);

        if (pm4File.MSLK?.Entries == null || !pm4File.MSLK.Entries.Any())
        {
            Console.WriteLine("   ⚠️  No MSLK chunk or entries found - skipping");
            return;
        }

        Console.WriteLine($"   📊 MSLK Entries: {pm4File.MSLK.Entries.Count}");

        // Perform hierarchy analysis
        var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);
        var objectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);

        Console.WriteLine($"   🎯 Objects Found: {objectSegments.Count}");
        Console.WriteLine($"   📈 Max Hierarchy Depth: {hierarchyResult.MaxDepth}");

        // Create output directories
        var outputDir = Path.Combine("output", baseFileName);
        Directory.CreateDirectory(outputDir);

        // Export simple TXT report only (no YAML to prevent circular references)
        var txtPath = Path.Combine(outputDir, $"{baseFileName}.mslk.txt");
        var report = hierarchyAnalyzer.GenerateHierarchyReport(hierarchyResult, fileName);
        File.WriteAllText(txtPath, report);
        Console.WriteLine($"   📄 TXT report: {Path.GetFileName(txtPath)}");

        // ✨ Export individual geometry objects (clean small components)
        var individualGeometry = hierarchyAnalyzer.SegmentByIndividualGeometry(hierarchyResult);
        var individualOutputDir = Path.Combine(outputDir, "individual_objects");
        Directory.CreateDirectory(individualOutputDir);
        
        Console.WriteLine($"🎯 Exporting {individualGeometry.Count} individual geometry objects...");
        
        var exportedCount = 0;
        foreach (var obj in individualGeometry)
        {
            try
            {
                var objFileName = $"{baseFileName}.geom_{obj.RootIndex:D3}.obj";
                var objPath = Path.Combine(individualOutputDir, objFileName);
                
                objectMeshExporter.ExportObjectMesh(obj, pm4File, objPath, renderMeshOnly: true);
                Console.WriteLine($"  ✅ Geometry {obj.RootIndex}: {objFileName}");
                exportedCount++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Failed to export geometry {obj.RootIndex}: {ex.Message}");
            }
        }
        
        Console.WriteLine($"📊 Export Summary: {exportedCount} exported, 0 skipped/failed");

        Console.WriteLine($"   📁 All outputs written to: {outputDir}");
        Console.WriteLine();
    }

    // Removed ExportAnalysisData method - now using simplified individual geometry export

    static void ShowUsage()
    {
        Console.WriteLine();
        Console.WriteLine("USAGE:");
        Console.WriteLine("  RunMslkObjectExport.exe                    # Run built-in demo with test data discovery");
        Console.WriteLine("  RunMslkObjectExport.exe <directory>        # Process all PM4 files in directory");
        Console.WriteLine("  RunMslkObjectExport.exe <file.pm4>         # Process single PM4 file");
        Console.WriteLine();
        Console.WriteLine("EXAMPLES:");
        Console.WriteLine("  RunMslkObjectExport.exe test_data/development/");
        Console.WriteLine("  RunMslkObjectExport.exe C:\\WoW\\PM4Files\\");
        Console.WriteLine("  RunMslkObjectExport.exe development_00_00.pm4");
        Console.WriteLine();
        Console.WriteLine("OUTPUT:");
        Console.WriteLine("  output/");
        Console.WriteLine("  ├── filename/");
        Console.WriteLine("  │   ├── filename.mslk.txt              # Detailed hierarchy analysis");
        Console.WriteLine("  │   ├── filename.mslk.yaml             # Structured hierarchy data");
        Console.WriteLine("  │   ├── filename.mslk.objects.yaml     # Object segmentation data");
        Console.WriteLine("  │   ├── filename.mslk.objects.txt      # Object segmentation summary");
        Console.WriteLine("  │   └── objects/                       # Per-object 3D meshes");
        Console.WriteLine("  │       ├── filename.object_000.obj");
        Console.WriteLine("  │       ├── filename.object_001.obj");
        Console.WriteLine("  │       └── filename.object_N.obj");
        Console.WriteLine("  └── [additional files...]");
    }
} 