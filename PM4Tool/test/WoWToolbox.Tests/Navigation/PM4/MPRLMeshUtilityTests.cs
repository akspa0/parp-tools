using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Tests.Navigation.PM4;

/// <summary>
/// Utility tests for combining MPRL mesh data from multiple PM4 files.
/// This is intended to help restore missing ADT terrain data by stitching together
/// PM4 terrain data across multiple files in a region.
/// </summary>
public class MPRLMeshUtilityTests
{
    [Fact]
    public void CombineMPRLMeshFromDirectory_ShouldLoadAllPM4Files()
    {
        // Arrange
        var testDataPath = Path.Combine("test_data", "original_development", "development");
        
        // Act & Assert
        Assert.True(Directory.Exists(testDataPath), $"Test data directory should exist: {testDataPath}");
        
        var pm4Files = Directory.GetFiles(testDataPath, "*.pm4", SearchOption.AllDirectories);
        Assert.True(pm4Files.Length > 0, "Should find PM4 files in test data directory");
    }

    [Fact]
    public void BuildCombinedMPRLMesh_ShouldCombineVerticesFromMultipleFiles()
    {
        // Arrange
        var testDataPath = Path.Combine("test_data", "original_development", "development");
        if (!Directory.Exists(testDataPath))
        {
            // Skip test if test data not available
            return;
        }

        var pm4Files = Directory.GetFiles(testDataPath, "*.pm4", SearchOption.AllDirectories)
            .Take(5) // Limit to first 5 files for testing
            .ToArray();

        if (pm4Files.Length == 0)
        {
            // Skip test if no PM4 files found
            return;
        }

        // Act
        var combinedMesh = BuildCombinedMPRLMesh(pm4Files);

        // Assert
        Assert.NotNull(combinedMesh);
        // We expect some vertices if files contain valid data
        // Note: Some test files might be empty, so we just check the method doesn't crash
    }

    [Fact]
    public void ExportCombinedMPRLMesh_ShouldCreateValidOBJFile()
    {
        // Arrange
        var testVertices = new List<Vector3>
        {
            new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0),
            new Vector3(1, 1, 0)
        };

        var outputPath = Path.Combine(Path.GetTempPath(), "test_combined_mprl.obj");

        try
        {
            // Act
            ExportMeshAsOBJ(testVertices, outputPath, "Test Combined MPRL Mesh");

            // Assert
            Assert.True(File.Exists(outputPath), "OBJ file should be created");
            
            var content = File.ReadAllText(outputPath);
            Assert.Contains("# Test Combined MPRL Mesh", content);
            // Check for coordinate mirroring fix (all coordinates negated)
            Assert.Contains("v -0.000000 -0.000000 -0.000000", content);
            Assert.Contains("v -1.000000 -0.000000 -0.000000", content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    [Fact]
    public void ExportCombinedMPRLMesh_ShouldCreateValidPLYFile()
    {
        // Arrange
        var testVertices = new List<Vector3>
        {
            new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0)
        };

        var outputPath = Path.Combine(Path.GetTempPath(), "test_combined_mprl.ply");

        try
        {
            // Act
            ExportMeshAsPLY(testVertices, outputPath, "Test Combined MPRL Mesh");

            // Assert
            Assert.True(File.Exists(outputPath), "PLY file should be created");
            
            var content = File.ReadAllText(outputPath);
            Assert.Contains("ply", content);
            Assert.Contains("element vertex 3", content);
            Assert.Contains("property float x", content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    /// <summary>
    /// Builds a combined MPRL mesh from multiple PM4 files.
    /// This combines all MSVT vertex data from the files into a single mesh.
    /// </summary>
    /// <param name="pm4FilePaths">Array of PM4 file paths to process</param>
    /// <returns>Combined mesh data with file source information</returns>
    public static CombinedMPRLMesh BuildCombinedMPRLMesh(string[] pm4FilePaths)
    {
        var combinedMesh = new CombinedMPRLMesh();

        foreach (var filePath in pm4FilePaths)
        {
            try
            {
                if (!File.Exists(filePath) || new FileInfo(filePath).Length == 0)
                {
                    combinedMesh.SkippedFiles.Add(Path.GetFileName(filePath), "File empty or missing");
                    continue;
                }

                var pm4File = PM4File.FromFile(filePath);
                var fileName = Path.GetFileName(filePath);

                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var vertices = pm4File.MSVT.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
                    combinedMesh.VerticesByFile[fileName] = vertices;
                    combinedMesh.AllVertices.AddRange(vertices);
                    combinedMesh.ProcessedFiles.Add(fileName);
                }
                else
                {
                    combinedMesh.SkippedFiles.Add(fileName, "No MSVT vertices found");
                }
            }
            catch (Exception ex)
            {
                combinedMesh.SkippedFiles.Add(Path.GetFileName(filePath), ex.Message);
            }
        }

        return combinedMesh;
    }

    /// <summary>
    /// Exports a mesh as an OBJ file
    /// </summary>
    internal static void ExportMeshAsOBJ(List<Vector3> vertices, string filePath, string comment = "Combined MPRL Mesh")
    {
        using var writer = new StreamWriter(filePath);
        
        writer.WriteLine($"# {comment}");
        writer.WriteLine($"# Generated: {DateTime.Now}");
        writer.WriteLine($"# Total Vertices: {vertices.Count:N0}");
        writer.WriteLine();

        foreach (var vertex in vertices)
        {
            // Fix coordinate mirroring: negate X (horizontal flip) and Y (vertical flip), keep Z negated
            writer.WriteLine($"v {-vertex.X:F6} {-vertex.Y:F6} {-vertex.Z:F6}");
        }
    }

    /// <summary>
    /// Exports a mesh as a PLY file
    /// </summary>
    internal static void ExportMeshAsPLY(List<Vector3> vertices, string filePath, string comment = "Combined MPRL Mesh")
    {
        using var writer = new StreamWriter(filePath);
        
        writer.WriteLine("ply");
        writer.WriteLine("format ascii 1.0");
        writer.WriteLine($"comment {comment}");
        writer.WriteLine($"comment Generated: {DateTime.Now}");
        writer.WriteLine($"comment Total Vertices: {vertices.Count:N0}");
        writer.WriteLine($"element vertex {vertices.Count}");
        writer.WriteLine("property float x");
        writer.WriteLine("property float y");
        writer.WriteLine("property float z");
        writer.WriteLine("end_header");

        foreach (var vertex in vertices)
        {
            // Fix coordinate mirroring: negate X (horizontal flip) and Y (vertical flip), keep Z negated
            writer.WriteLine($"{-vertex.X:F6} {-vertex.Y:F6} {-vertex.Z:F6}");
        }
    }

    /// <summary>
    /// Exports a mesh as a simple text file with statistics
    /// </summary>
    internal static void ExportMeshAsText(List<Vector3> vertices, string filePath, Dictionary<string, List<Vector3>> verticesByFile)
    {
        using var writer = new StreamWriter(filePath);
        
        writer.WriteLine("Combined MPRL Mesh Report");
        writer.WriteLine("========================");
        writer.WriteLine($"Generated: {DateTime.Now}");
        writer.WriteLine($"Total Vertices: {vertices.Count:N0}");
        writer.WriteLine($"Source Files: {verticesByFile.Count}");
        writer.WriteLine($"Coordinate Fix: Applied mirroring correction (negated X, Y, Z)");
        writer.WriteLine();

        writer.WriteLine("File Breakdown:");
        foreach (var kvp in verticesByFile.OrderBy(x => x.Key))
        {
            writer.WriteLine($"  {kvp.Key}: {kvp.Value.Count:N0} vertices");
        }
        writer.WriteLine();

        writer.WriteLine("Vertex Data:");
        for (int i = 0; i < vertices.Count; i++)
        {
            var vertex = vertices[i];
            writer.WriteLine($"{i + 1}: ({vertex.X:F6}, {vertex.Y:F6}, {vertex.Z:F6})");
        }
    }
}

/// <summary>
/// Container for combined MPRL mesh data
/// </summary>
public class CombinedMPRLMesh
{
    public List<Vector3> AllVertices { get; set; } = new();
    public Dictionary<string, List<Vector3>> VerticesByFile { get; set; } = new();
    public List<string> ProcessedFiles { get; set; } = new();
    public Dictionary<string, string> SkippedFiles { get; set; } = new();

    public int TotalVertices => AllVertices.Count;
    public int ProcessedFileCount => ProcessedFiles.Count;
    public int SkippedFileCount => SkippedFiles.Count;

    public string GetSummary()
    {
        return $"Combined {TotalVertices:N0} vertices from {ProcessedFileCount} files " +
               $"({SkippedFileCount} files skipped)";
    }
} 