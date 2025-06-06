using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Tests.Navigation.PM4;

namespace WoWToolbox.Tests.Scripts;

/// <summary>
/// Standalone utility to combine MPRL mesh data from multiple PM4 files.
/// This can be used to restore missing ADT terrain data by stitching together
/// PM4 terrain data across multiple files in a region.
/// 
/// Usage:
/// - Run as unit test: dotnet test --filter "CombineMPRLMeshFromDirectory"
/// - Or modify and run this as a script
/// </summary>
public class CombineMPRLMeshScript
{
    /// <summary>
    /// Example of how to use the MPRL mesh utility to process a directory of PM4 files
    /// </summary>
    [Fact]
    public void CombineMPRLMeshFromDirectory_Example()
    {
        // Configuration
        var inputDirectory = Path.Combine("test_data", "original_development", "development");
        var outputDirectory = Path.Combine("output", "combined_mprl_mesh");
        
        if (!Directory.Exists(inputDirectory))
        {
            Console.WriteLine($"Input directory not found: {inputDirectory}");
            return;
        }

        // Create output directory
        Directory.CreateDirectory(outputDirectory);

        // Find all PM4 files
        var pm4Files = Directory.GetFiles(inputDirectory, "*.pm4", SearchOption.AllDirectories);
        Console.WriteLine($"Found {pm4Files.Length} PM4 files in {inputDirectory}");

        if (pm4Files.Length == 0)
        {
            Console.WriteLine("No PM4 files found to process.");
            return;
        }

        // Build combined mesh
        Console.WriteLine("Building combined MPRL mesh...");
        var combinedMesh = MPRLMeshUtilityTests.BuildCombinedMPRLMesh(pm4Files);

        // Print summary
        Console.WriteLine(combinedMesh.GetSummary());
        Console.WriteLine($"Processed files:");
        foreach (var file in combinedMesh.ProcessedFiles)
        {
            var vertexCount = combinedMesh.VerticesByFile[file].Count;
            Console.WriteLine($"  {file}: {vertexCount:N0} vertices");
        }

        if (combinedMesh.SkippedFiles.Any())
        {
            Console.WriteLine($"Skipped files:");
            foreach (var kvp in combinedMesh.SkippedFiles)
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            }
        }

        if (combinedMesh.TotalVertices == 0)
        {
            Console.WriteLine("No vertices found in any files. Nothing to export.");
            return;
        }

        // Export in multiple formats
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        
        // Export as OBJ
        var objPath = Path.Combine(outputDirectory, $"combined_mprl_mesh_{timestamp}.obj");
        MPRLMeshUtilityTests.ExportMeshAsOBJ(combinedMesh.AllVertices, objPath, 
            $"Combined MPRL Mesh from {combinedMesh.ProcessedFileCount} PM4 files");
        Console.WriteLine($"Exported OBJ: {objPath}");

        // Export as PLY
        var plyPath = Path.Combine(outputDirectory, $"combined_mprl_mesh_{timestamp}.ply");
        MPRLMeshUtilityTests.ExportMeshAsPLY(combinedMesh.AllVertices, plyPath,
            $"Combined MPRL Mesh from {combinedMesh.ProcessedFileCount} PM4 files");
        Console.WriteLine($"Exported PLY: {plyPath}");

        // Export detailed report
        var reportPath = Path.Combine(outputDirectory, $"combined_mprl_report_{timestamp}.txt");
        MPRLMeshUtilityTests.ExportMeshAsText(combinedMesh.AllVertices, reportPath, combinedMesh.VerticesByFile);
        Console.WriteLine($"Exported report: {reportPath}");

        Console.WriteLine($"Combined MPRL mesh processing complete!");
        Console.WriteLine($"Combined {combinedMesh.TotalVertices:N0} vertices from {combinedMesh.ProcessedFileCount} files");
    }

    /// <summary>
    /// Example of combining mesh data from a specific list of files
    /// </summary>
    [Fact]
    public void CombineMPRLMeshFromSpecificFiles_Example()
    {
        // Example: Process specific development tiles that form a region
        var specificFiles = new[]
        {
            Path.Combine("test_data", "original_development", "development", "development_22_56.pm4"),
            Path.Combine("test_data", "original_development", "development", "development_23_56.pm4"),
            Path.Combine("test_data", "original_development", "development", "development_22_57.pm4"),
            Path.Combine("test_data", "original_development", "development", "development_23_57.pm4")
        };

        // Filter to only existing files
        var existingFiles = specificFiles.Where(File.Exists).ToArray();
        
        if (existingFiles.Length == 0)
        {
            Console.WriteLine("No specified files found for processing.");
            return;
        }

        Console.WriteLine($"Processing {existingFiles.Length} specific PM4 files...");

        // Build combined mesh
        var combinedMesh = MPRLMeshUtilityTests.BuildCombinedMPRLMesh(existingFiles);
        
        Console.WriteLine(combinedMesh.GetSummary());

        if (combinedMesh.TotalVertices > 0)
        {
            var outputPath = Path.Combine("output", "specific_region_mesh.obj");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            
            MPRLMeshUtilityTests.ExportMeshAsOBJ(combinedMesh.AllVertices, outputPath,
                "Combined MPRL Mesh from specific region tiles");
            
            Console.WriteLine($"Exported specific region mesh: {outputPath}");
        }
    }

    /// <summary>
    /// Example of processing files with coordinate filtering/grouping
    /// This could be useful for organizing mesh data by spatial regions
    /// </summary>
    [Fact]
    public void CombineMPRLMeshWithCoordinateAnalysis_Example()
    {
        var inputDirectory = Path.Combine("test_data", "original_development", "development");
        
        if (!Directory.Exists(inputDirectory))
        {
            Console.WriteLine($"Input directory not found: {inputDirectory}");
            return;
        }

        var pm4Files = Directory.GetFiles(inputDirectory, "*.pm4", SearchOption.AllDirectories)
            .Take(10) // Limit for testing
            .ToArray();

        if (pm4Files.Length == 0)
        {
            return;
        }

        Console.WriteLine($"Analyzing coordinate distribution in {pm4Files.Length} files...");

        var coordinateAnalysis = new Dictionary<string, (Vector3 min, Vector3 max, int count)>();

        foreach (var filePath in pm4Files)
        {
            try
            {
                if (!File.Exists(filePath) || new FileInfo(filePath).Length == 0)
                    continue;

                var pm4File = PM4File.FromFile(filePath);
                var fileName = Path.GetFileName(filePath);

                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var vertices = pm4File.MSVT.Vertices;
                    var minX = vertices.Min(v => v.X);
                    var maxX = vertices.Max(v => v.X);
                    var minY = vertices.Min(v => v.Y);
                    var maxY = vertices.Max(v => v.Y);
                    var minZ = vertices.Min(v => v.Z);
                    var maxZ = vertices.Max(v => v.Z);

                    coordinateAnalysis[fileName] = (
                        new Vector3(minX, minY, minZ),
                        new Vector3(maxX, maxY, maxZ),
                        vertices.Count
                    );
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing {Path.GetFileName(filePath)}: {ex.Message}");
            }
        }

        // Print coordinate analysis
        Console.WriteLine("\nCoordinate Analysis:");
        Console.WriteLine("===================");
        foreach (var kvp in coordinateAnalysis.OrderBy(x => x.Key))
        {
            var (min, max, count) = kvp.Value;
            Console.WriteLine($"{kvp.Key}:");
            Console.WriteLine($"  Vertices: {count:N0}");
            Console.WriteLine($"  Bounds: ({min.X:F2}, {min.Y:F2}, {min.Z:F2}) to ({max.X:F2}, {max.Y:F2}, {max.Z:F2})");
            Console.WriteLine($"  Size: {max.X - min.X:F2} x {max.Y - min.Y:F2} x {max.Z - min.Z:F2}");
            Console.WriteLine();
        }
    }
} 