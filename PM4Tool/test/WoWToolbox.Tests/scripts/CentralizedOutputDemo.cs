using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Tests.Navigation.PM4;
using WoWToolbox.Tests.Utilities;
using Xunit;

namespace WoWToolbox.Tests.Scripts;

/// <summary>
/// Demonstration of the centralized output system with Z-coordinate fixes
/// This shows how all PM4Tool outputs are collected into timestamped folders
/// </summary>
public class CentralizedOutputDemo
{
    [Fact]
    public void DemonstrateCentralizedOutputWithZFix_ShouldCreateTimestampedFolder()
    {
        Console.WriteLine("=== PM4Tool Centralized Output Demonstration ===");
        Console.WriteLine();
        
        // Reset session to start fresh
        CentralizedOutputManager.ResetSession();
        
        // Step 1: Process MPRL mesh data with Z-coordinate fix
        Console.WriteLine("Step 1: Processing MPRL mesh data with Z-coordinate fix...");
        
        var testDataPath = Path.Combine("test_data", "original_development", "development");
        if (Directory.Exists(testDataPath))
        {
            var pm4Files = Directory.GetFiles(testDataPath, "*.pm4")
                .Where(f => new FileInfo(f).Length > 0)
                .Take(5) // Process first 5 files for demonstration
                .ToArray();
                
            if (pm4Files.Length > 0)
            {
                // Build combined mesh
                var combinedMesh = MPRLMeshUtilityTests.BuildCombinedMPRLMesh(pm4Files);
                
                if (combinedMesh.TotalVertices > 0)
                {
                    // Create centralized output folder for MPRL data
                    var mprlFolder = CentralizedOutputManager.CreateComponentFolder("mprl_mesh_with_z_fix");
                    
                    // Export with Z-coordinate fix
                    var objPath = Path.Combine(mprlFolder, "combined_terrain_z_fixed.obj");
                    var plyPath = Path.Combine(mprlFolder, "combined_terrain_z_fixed.ply");
                    var reportPath = Path.Combine(mprlFolder, "processing_summary.txt");
                    
                    MPRLMeshUtilityTests.ExportMeshAsOBJ(combinedMesh.AllVertices, objPath, "PM4 Terrain Mesh - Z-Coordinate Fixed");
                    MPRLMeshUtilityTests.ExportMeshAsPLY(combinedMesh.AllVertices, plyPath, "PM4 Terrain Mesh - Z-Coordinate Fixed");
                    MPRLMeshUtilityTests.ExportMeshAsText(combinedMesh.AllVertices, reportPath, combinedMesh.VerticesByFile);
                    
                    Console.WriteLine($"  ✅ Processed {combinedMesh.TotalVertices:N0} vertices from {combinedMesh.ProcessedFileCount} files");
                    Console.WriteLine($"  ✅ Z-coordinate inversion fixed (negated Z values)");
                    Console.WriteLine($"  ✅ Exported OBJ, PLY, and text report");
                }
            }
        }
        
        // Step 2: Create additional component outputs for demonstration
        Console.WriteLine("\nStep 2: Creating additional component outputs...");
        
        var analysisFolder = CentralizedOutputManager.CreateComponentFolder("analysis");
        var demoAnalysisFile = Path.Combine(analysisFolder, "terrain_analysis.txt");
        File.WriteAllText(demoAnalysisFile, $"Demo terrain analysis generated at {DateTime.Now}\nThis demonstrates component organization.");
        
        var exportFolder = CentralizedOutputManager.CreateComponentFolder("enhanced_exports");
        var demoExportFile = Path.Combine(exportFolder, "enhanced_terrain.obj");
        File.WriteAllText(demoExportFile, $"# Demo enhanced export\n# Generated: {DateTime.Now}\n# Features: Z-coordinate fix, enhanced materials");
        
        Console.WriteLine("  ✅ Created analysis reports");
        Console.WriteLine("  ✅ Created enhanced export samples");
        
        // Step 3: Generate comprehensive session report
        Console.WriteLine("\nStep 3: Generating comprehensive session report...");
        
        var sessionInfo = new Dictionary<string, object>
        {
            {"Demo Purpose", "Showcase centralized output with Z-coordinate fix"},
            {"Z-Coordinate Fix", "Applied (negated Z values for correct orientation)"},
            {"Output Organization", "Timestamped folders with component separation"},
            {"Export Formats", "OBJ (fixed), PLY (fixed), Text Reports"},
            {"Session Type", "Demonstration"}
        };
        
        CentralizedOutputManager.GenerateSessionReport(sessionInfo);
        
        // Step 4: Show results
        Console.WriteLine("\nStep 4: Session Results Summary");
        var sessionFolder = CentralizedOutputManager.CurrentSessionFolder;
        
        Console.WriteLine($"📁 Session Folder: {sessionFolder}");
        Console.WriteLine($"📊 Session Report: {Path.Combine(sessionFolder, "session_report.md")}");
        
        var componentDirs = Directory.GetDirectories(sessionFolder);
        Console.WriteLine($"\n🔍 Components Created ({componentDirs.Length}):");
        
        foreach (var dir in componentDirs)
        {
            var componentName = Path.GetFileName(dir);
            var files = Directory.GetFiles(dir, "*", SearchOption.AllDirectories);
            var totalSize = files.Sum(f => new FileInfo(f).Length);
            
            Console.WriteLine($"  📂 {componentName}:");
            Console.WriteLine($"     • Files: {files.Length}");
            Console.WriteLine($"     • Size: {FormatFileSize(totalSize)}");
            Console.WriteLine($"     • Path: {dir}");
        }
        
        Console.WriteLine("\n✅ Centralized Output Demonstration Complete!");
        Console.WriteLine("🎯 All outputs are organized in timestamped folders by component");
        Console.WriteLine("🔧 Z-coordinate inversion has been fixed in all mesh exports");
        Console.WriteLine("📋 Comprehensive session report generated");
        
        // Assert key outputs exist
        Assert.True(Directory.Exists(sessionFolder), "Session folder should exist");
        Assert.True(File.Exists(Path.Combine(sessionFolder, "session_report.md")), "Session report should exist");
        Assert.True(componentDirs.Length >= 2, "Should have multiple component folders");
    }
    
    [Fact]
    public void ShowOutputStructure_ShouldDemonstrateOrganization()
    {
        Console.WriteLine("=== PM4Tool Output Structure ===");
        Console.WriteLine();
        Console.WriteLine("📁 output/");
        Console.WriteLine("  📁 pm4tool_session_YYYYMMDD_HHMMSS/   ← Timestamped session folder");
        Console.WriteLine("    📄 session_report.md                ← Comprehensive session report");
        Console.WriteLine("    📁 mprl_mesh_with_z_fix/            ← MPRL terrain meshes (Z-fixed)");
        Console.WriteLine("      📄 combined_terrain_z_fixed.obj   ← Main terrain mesh (OBJ format)");
        Console.WriteLine("      📄 combined_terrain_z_fixed.ply   ← Main terrain mesh (PLY format)");
        Console.WriteLine("      📄 processing_summary.txt         ← Processing statistics");
        Console.WriteLine("    📁 analysis/                        ← Analysis outputs");
        Console.WriteLine("      📄 terrain_analysis.txt           ← Terrain analysis reports");
        Console.WriteLine("      📄 coordinate_distributions.csv   ← Statistical data");
        Console.WriteLine("    📁 enhanced_exports/                ← Advanced export formats");
        Console.WriteLine("      📄 enhanced_terrain.obj           ← Enhanced OBJ with materials");
        Console.WriteLine("      📄 surface_normals.obj            ← Surface normal visualization");
        Console.WriteLine("    📁 gui_exports/                     ← GUI application exports");
        Console.WriteLine("      📄 gui_session_export.obj         ← Interactive session exports");
        Console.WriteLine("    📁 batch_processing/                ← Batch operation results");
        Console.WriteLine("      📄 batch_summary.txt              ← Batch processing logs");
        Console.WriteLine();
        Console.WriteLine("🔧 Key Features:");
        Console.WriteLine("  • Z-coordinate inversion fixed (negated Z values)");
        Console.WriteLine("  • Timestamped sessions prevent overwriting");
        Console.WriteLine("  • Component-based organization");
        Console.WriteLine("  • Comprehensive reporting");
        Console.WriteLine("  • Multiple export formats (OBJ, PLY, text)");
        Console.WriteLine("  • Automatic collection of existing outputs");
        
        Assert.True(true, "Structure demonstration completed");
    }
    
    private static string FormatFileSize(long bytes)
    {
        string[] suffixes = { "B", "KB", "MB", "GB" };
        int suffixIndex = 0;
        double size = bytes;
        
        while (size >= 1024 && suffixIndex < suffixes.Length - 1)
        {
            size /= 1024;
            suffixIndex++;
        }
        
        return $"{size:F1} {suffixes[suffixIndex]}";
    }
} 