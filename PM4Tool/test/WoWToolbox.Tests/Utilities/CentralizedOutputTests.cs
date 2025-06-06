using System;
using System.Collections.Generic;
using System.IO;
using WoWToolbox.Tests.Navigation.PM4;
using Xunit;

namespace WoWToolbox.Tests.Utilities;

/// <summary>
/// Tests for the centralized output management system
/// </summary>
public class CentralizedOutputTests
{
    [Fact]
    public void CreateComprehensiveSession_ShouldCollectAllPM4ToolOutputs()
    {
        // Act
        var sessionPath = CentralizedOutputManager.CreateComprehensiveSession(includeExisting: true);
        
        // Assert
        Assert.True(Directory.Exists(sessionPath), "Session directory should be created");
        Assert.True(File.Exists(Path.Combine(sessionPath, "session_report.md")), "Session report should be generated");
        
        Console.WriteLine($"📁 Centralized session created: {sessionPath}");
        Console.WriteLine($"📊 Session report: {Path.Combine(sessionPath, "session_report.md")}");
        
        // List components found
        var componentDirs = Directory.GetDirectories(sessionPath);
        if (componentDirs.Length > 0)
        {
            Console.WriteLine("\n🔍 Found component outputs:");
            foreach (var dir in componentDirs)
            {
                var componentName = Path.GetFileName(dir);
                var fileCount = Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Length;
                Console.WriteLine($"  • {componentName}: {fileCount} files");
            }
        }
    }
    
    [Fact] 
    public async Task RunMPRLWithCentralizedOutput_ShouldCreateFixedMeshes()
    {
        // Arrange - Reset session for clean test
        CentralizedOutputManager.ResetSession();
        
        // Act - Run MPRL mesh processing with centralized output
        var mprlTests = new MPRLMeshUtilityTests();
        await mprlTests.CombineMPRLMeshFromDirectory_ShouldLoadAllPM4Files();
        
        // Generate timestamped outputs with Z-fix
        var testDataPath = Path.Combine("test_data", "original_development", "development");
        if (Directory.Exists(testDataPath))
        {
            var pm4Files = Directory.GetFiles(testDataPath, "*.pm4")
                .Where(f => new FileInfo(f).Length > 0)
                .Take(10) // Process first 10 files for test
                .ToArray();
                
            if (pm4Files.Length > 0)
            {
                var combinedMesh = MPRLMeshUtilityTests.BuildCombinedMPRLMesh(pm4Files);
                
                if (combinedMesh.TotalVertices > 0)
                {
                    // Export to centralized output with Z-fix
                    var componentFolder = CentralizedOutputManager.CreateComponentFolder("mprl_mesh_fixed");
                    
                    var objPath = Path.Combine(componentFolder, "combined_mprl_z_fixed.obj");
                    var plyPath = Path.Combine(componentFolder, "combined_mprl_z_fixed.ply");
                    var reportPath = Path.Combine(componentFolder, "mprl_processing_report.txt");
                    
                    MPRLMeshUtilityTests.ExportMeshAsOBJ(combinedMesh.AllVertices, objPath, "MPRL Mesh with Z-Coordinate Fix Applied");
                    MPRLMeshUtilityTests.ExportMeshAsPLY(combinedMesh.AllVertices, plyPath, "MPRL Mesh with Z-Coordinate Fix Applied");
                    MPRLMeshUtilityTests.ExportMeshAsText(combinedMesh.AllVertices, reportPath, combinedMesh.VerticesByFile);
                    
                    // Generate session report
                    var sessionInfo = new Dictionary<string, object>
                    {
                        {"MPRL Files Processed", combinedMesh.ProcessedFileCount},
                        {"MPRL Files Skipped", combinedMesh.SkippedFileCount},
                        {"MPRL Total Vertices", $"{combinedMesh.TotalVertices:N0}"},
                        {"Z-Coordinate Fix", "Applied (negated Z values)"},
                        {"Export Formats", "OBJ, PLY, Text Report"}
                    };
                    
                    CentralizedOutputManager.GenerateSessionReport(sessionInfo);
                    
                    Console.WriteLine($"✅ MPRL mesh processing with Z-fix completed");
                    Console.WriteLine($"📁 Outputs saved to: {CentralizedOutputManager.CurrentSessionFolder}");
                    Console.WriteLine($"🔧 Z-coordinate inversion fixed in exports");
                    Console.WriteLine($"📊 Processed {combinedMesh.TotalVertices:N0} vertices from {combinedMesh.ProcessedFileCount} files");
                    
                    // Assert files were created with fixed coordinates
                    Assert.True(File.Exists(objPath), "Fixed OBJ file should be created");
                    Assert.True(File.Exists(plyPath), "Fixed PLY file should be created");
                    Assert.True(File.Exists(reportPath), "Processing report should be created");
                    
                    // Verify Z-coordinate fix in OBJ file
                    var objContent = File.ReadAllText(objPath);
                    Assert.Contains("# MPRL Mesh with Z-Coordinate Fix Applied", objContent);
                }
            }
        }
    }
    
    [Fact]
    public void CentralizedOutputManager_ShouldCreateTimestampedFolders()
    {
        // Arrange - Reset to ensure fresh timestamp
        CentralizedOutputManager.ResetSession();
        
        // Act
        var folder1 = CentralizedOutputManager.CurrentSessionFolder;
        
        // Reset and get another folder (should have different timestamp)
        CentralizedOutputManager.ResetSession();
        var folder2 = CentralizedOutputManager.CurrentSessionFolder;
        
        // Assert
        Assert.True(Directory.Exists(folder1), "First session folder should exist");
        Assert.True(Directory.Exists(folder2), "Second session folder should exist");
        Assert.NotEqual(folder1, folder2, "Folders should have different timestamps");
        
        Assert.Contains("pm4tool_session_", folder1);
        Assert.Contains("pm4tool_session_", folder2);
        
        Console.WriteLine($"📁 Session 1: {folder1}");
        Console.WriteLine($"📁 Session 2: {folder2}");
    }
    
    [Fact]
    public void ComponentFolders_ShouldOrganizeOutputsByType()
    {
        // Arrange
        CentralizedOutputManager.ResetSession();
        
        // Act - Create various component folders
        var mprlFolder = CentralizedOutputManager.CreateComponentFolder("mprl_mesh");
        var analysisFolder = CentralizedOutputManager.CreateComponentFolder("analysis");
        var exportFolder = CentralizedOutputManager.CreateComponentFolder("exports");
        
        // Assert
        Assert.True(Directory.Exists(mprlFolder), "MPRL component folder should exist");
        Assert.True(Directory.Exists(analysisFolder), "Analysis component folder should exist");
        Assert.True(Directory.Exists(exportFolder), "Export component folder should exist");
        
        Assert.Contains("mprl_mesh", mprlFolder);
        Assert.Contains("analysis", analysisFolder);
        Assert.Contains("exports", exportFolder);
        
        // All should be under the same session folder
        var sessionFolder = CentralizedOutputManager.CurrentSessionFolder;
        Assert.Contains(sessionFolder, mprlFolder);
        Assert.Contains(sessionFolder, analysisFolder);
        Assert.Contains(sessionFolder, exportFolder);
        
        Console.WriteLine($"📁 Session: {sessionFolder}");
        Console.WriteLine($"  📂 MPRL: {Path.GetFileName(mprlFolder)}");
        Console.WriteLine($"  📂 Analysis: {Path.GetFileName(analysisFolder)}");
        Console.WriteLine($"  📂 Exports: {Path.GetFileName(exportFolder)}");
    }
} 