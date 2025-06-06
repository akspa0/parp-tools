using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Tests.Navigation.PM4;
using Xunit;

namespace WoWToolbox.Tests.Analysis;

/// <summary>
/// Comprehensive PM4 data processing pipeline that combines multiple analysis operations
/// including MPRL mesh combination, structural analysis, and data export.
/// </summary>
public class ComprehensivePM4PipelineTests
{
    [Fact]
    public async Task RunComprehensivePM4Pipeline_ShouldProcessAllData()
    {
        // Arrange
        var testDataPath = Path.Combine("test_data", "original_development", "development");
        var outputRoot = Path.Combine("output", $"comprehensive_pipeline_{DateTime.Now:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(outputRoot);
        
        var results = new PipelineResults();
        
        // Act & Assert
        Assert.True(Directory.Exists(testDataPath), $"Test data directory should exist: {testDataPath}");
        
        // Step 1: MPRL Mesh Combination
        results.MprlMeshResult = await ProcessMPRLMeshData(testDataPath, outputRoot);
        Assert.True(results.MprlMeshResult.Success, "MPRL mesh processing should succeed");
        
        // Step 2: Individual PM4 Analysis
        results.IndividualAnalysisResults = await ProcessIndividualPM4Files(testDataPath, outputRoot);
        Assert.True(results.IndividualAnalysisResults.Count > 0, "Should process at least one PM4 file");
        
        // Step 3: Cross-File Analysis
        results.CrossFileAnalysis = await PerformCrossFileAnalysis(results.IndividualAnalysisResults, outputRoot);
        Assert.NotNull(results.CrossFileAnalysis);
        
        // Step 4: Generate Comprehensive Report
        await GenerateComprehensiveReport(results, outputRoot);
        
        // Verify outputs exist
        Assert.True(File.Exists(Path.Combine(outputRoot, "pipeline_summary.txt")));
        Assert.True(Directory.Exists(Path.Combine(outputRoot, "mprl_mesh")));
        Assert.True(Directory.Exists(Path.Combine(outputRoot, "individual_analysis")));
        Assert.True(Directory.Exists(Path.Combine(outputRoot, "cross_file_analysis")));
        
        Console.WriteLine($"Comprehensive PM4 pipeline completed successfully!");
        Console.WriteLine($"Output directory: {outputRoot}");
        Console.WriteLine($"Total vertices processed: {results.MprlMeshResult.TotalVertices:N0}");
        Console.WriteLine($"Files analyzed: {results.IndividualAnalysisResults.Count}");
    }
    
    private async Task<MPRLMeshResult> ProcessMPRLMeshData(string testDataPath, string outputRoot)
    {
        var mprlOutputDir = Path.Combine(outputRoot, "mprl_mesh");
        Directory.CreateDirectory(mprlOutputDir);
        
        var result = new MPRLMeshResult();
        var combinedMesh = new CombinedMPRLMesh();
        
        var pm4Files = Directory.GetFiles(testDataPath, "*.pm4")
            .Where(f => new FileInfo(f).Length > 0)
            .ToList();
        
        foreach (var filePath in pm4Files)
        {
            try
            {
                var fileName = Path.GetFileName(filePath);
                var fileBytes = await File.ReadAllBytesAsync(filePath);
                var pm4File = new PM4File(fileBytes);
                
                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var vertices = pm4File.MSVT.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
                    combinedMesh.VerticesByFile[fileName] = vertices;
                    combinedMesh.AllVertices.AddRange(vertices);
                    combinedMesh.ProcessedFiles.Add(fileName);
                    result.ProcessedFiles++;
                }
                else
                {
                    result.SkippedFiles++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing {filePath}: {ex.Message}");
                result.SkippedFiles++;
            }
        }
        
        result.TotalVertices = combinedMesh.AllVertices.Count;
        result.Success = result.TotalVertices > 0;
        
        if (result.Success)
        {
            // Export combined mesh using the utility methods
            MPRLMeshUtilityTests.ExportMeshAsOBJ(
                combinedMesh.AllVertices, 
                Path.Combine(mprlOutputDir, "combined_mprl_mesh.obj"),
                "Combined MPRL Mesh from Comprehensive Pipeline"
            );
            
            MPRLMeshUtilityTests.ExportMeshAsPLY(
                combinedMesh.AllVertices,
                Path.Combine(mprlOutputDir, "combined_mprl_mesh.ply"),
                "Combined MPRL Mesh from Comprehensive Pipeline"
            );
            
            MPRLMeshUtilityTests.ExportMeshAsText(
                combinedMesh.AllVertices,
                Path.Combine(mprlOutputDir, "combined_mprl_report.txt"),
                combinedMesh.VerticesByFile
            );
        }
        
        return result;
    }
    
    private async Task<List<IndividualPM4Analysis>> ProcessIndividualPM4Files(string testDataPath, string outputRoot)
    {
        var analysisOutputDir = Path.Combine(outputRoot, "individual_analysis");
        Directory.CreateDirectory(analysisOutputDir);
        
        var results = new List<IndividualPM4Analysis>();
        var pm4Files = Directory.GetFiles(testDataPath, "*.pm4")
            .Where(f => new FileInfo(f).Length > 0)
            .Take(10) // Limit to first 10 files for pipeline performance
            .ToList();
        
        foreach (var filePath in pm4Files)
        {
            try
            {
                var fileName = Path.GetFileName(filePath);
                var fileBytes = await File.ReadAllBytesAsync(filePath);
                var pm4File = new PM4File(fileBytes);
                
                var analysis = new IndividualPM4Analysis
                {
                    FileName = fileName,
                    FilePath = filePath,
                    FileSize = new FileInfo(filePath).Length,
                    VertexCount = pm4File.MSVT?.Vertices?.Count ?? 0,
                    SurfaceCount = pm4File.MSUR?.Entries?.Count ?? 0,
                    LinkCount = pm4File.MSLK?.Entries?.Count ?? 0,
                    HasValidData = (pm4File.MSVT?.Vertices?.Count ?? 0) > 0
                };
                
                // Analyze object types if MSLK data is available
                if (pm4File.MSLK?.Entries != null)
                {
                    analysis.ObjectTypes = pm4File.MSLK.Entries
                        .GroupBy(e => e.ObjectTypeFlags)
                        .ToDictionary(g => g.Key, g => g.Count());
                        
                    analysis.MaterialTypes = pm4File.MSLK.Entries
                        .GroupBy(e => e.MaterialColorId)
                        .ToDictionary(g => g.Key, g => g.Count());
                }
                
                // Export individual analysis
                var analysisFile = Path.Combine(analysisOutputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_analysis.txt");
                await File.WriteAllTextAsync(analysisFile, analysis.ToDetailedString());
                
                results.Add(analysis);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error analyzing {filePath}: {ex.Message}");
            }
        }
        
        return results;
    }
    
    private async Task<CrossFileAnalysisResult> PerformCrossFileAnalysis(List<IndividualPM4Analysis> individualResults, string outputRoot)
    {
        var crossAnalysisDir = Path.Combine(outputRoot, "cross_file_analysis");
        Directory.CreateDirectory(crossAnalysisDir);
        
        var result = new CrossFileAnalysisResult
        {
            TotalFilesAnalyzed = individualResults.Count,
            TotalVertices = individualResults.Sum(r => r.VertexCount),
            TotalSurfaces = individualResults.Sum(r => r.SurfaceCount),
            TotalLinks = individualResults.Sum(r => r.LinkCount),
            FilesWithValidData = individualResults.Count(r => r.HasValidData)
        };
        
        // Analyze object type distribution across all files
        result.GlobalObjectTypeDistribution = individualResults
            .Where(r => r.ObjectTypes != null)
            .SelectMany(r => r.ObjectTypes)
            .GroupBy(kvp => kvp.Key)
            .ToDictionary(g => g.Key, g => g.Sum(kvp => kvp.Value));
            
        // Analyze material distribution across all files
        result.GlobalMaterialDistribution = individualResults
            .Where(r => r.MaterialTypes != null)
            .SelectMany(r => r.MaterialTypes)
            .GroupBy(kvp => kvp.Key)
            .ToDictionary(g => g.Key, g => g.Sum(kvp => kvp.Value));
        
        // Export cross-file analysis
        var crossAnalysisFile = Path.Combine(crossAnalysisDir, "cross_file_analysis.txt");
        await File.WriteAllTextAsync(crossAnalysisFile, result.ToDetailedString());
        
        // Export summary CSV
        var csvFile = Path.Combine(crossAnalysisDir, "file_summary.csv");
        await ExportSummaryCSV(individualResults, csvFile);
        
        return result;
    }
    
    private async Task GenerateComprehensiveReport(PipelineResults results, string outputRoot)
    {
        var reportFile = Path.Combine(outputRoot, "pipeline_summary.txt");
        var report = new List<string>
        {
            "=== Comprehensive PM4 Pipeline Results ===",
            $"Generated: {DateTime.Now}",
            $"Output Directory: {outputRoot}",
            "",
            "=== MPRL Mesh Combination ===",
            $"Success: {results.MprlMeshResult.Success}",
            $"Total Vertices: {results.MprlMeshResult.TotalVertices:N0}",
            $"Processed Files: {results.MprlMeshResult.ProcessedFiles}",
            $"Skipped Files: {results.MprlMeshResult.SkippedFiles}",
            "",
            "=== Individual File Analysis ===",
            $"Files Analyzed: {results.IndividualAnalysisResults.Count}",
            $"Files with Valid Data: {results.IndividualAnalysisResults.Count(r => r.HasValidData)}",
            $"Average Vertices per File: {(results.IndividualAnalysisResults.Count > 0 ? results.IndividualAnalysisResults.Average(r => r.VertexCount) : 0):F1}",
            "",
            "=== Cross-File Analysis ===",
            $"Total Vertices Across All Files: {results.CrossFileAnalysis.TotalVertices:N0}",
            $"Total Surfaces: {results.CrossFileAnalysis.TotalSurfaces:N0}",
            $"Total Links: {results.CrossFileAnalysis.TotalLinks:N0}",
            $"Unique Object Types: {results.CrossFileAnalysis.GlobalObjectTypeDistribution.Count}",
            $"Unique Material Types: {results.CrossFileAnalysis.GlobalMaterialDistribution.Count}",
            "",
            "=== Output Files Generated ===",
            "• mprl_mesh/combined_mprl_mesh.obj - Combined terrain mesh",
            "• mprl_mesh/combined_mprl_mesh.ply - Combined terrain mesh (PLY format)",
            "• mprl_mesh/combined_mprl_report.txt - Detailed mesh statistics",
            "• individual_analysis/*.txt - Per-file analysis reports",
            "• cross_file_analysis/cross_file_analysis.txt - Cross-file patterns",
            "• cross_file_analysis/file_summary.csv - Summary data table",
            "• pipeline_summary.txt - This summary report",
            "",
            "=== Pipeline Status ===",
            "✅ MPRL mesh combination completed",
            "✅ Individual file analysis completed", 
            "✅ Cross-file analysis completed",
            "✅ Comprehensive report generated",
            "",
            "Pipeline completed successfully!"
        };
        
        await File.WriteAllLinesAsync(reportFile, report);
    }
    
    private async Task ExportSummaryCSV(List<IndividualPM4Analysis> results, string csvFile)
    {
        var lines = new List<string>
        {
            "FileName,FileSize,VertexCount,SurfaceCount,LinkCount,HasValidData,ObjectTypes,MaterialTypes"
        };
        
        foreach (var result in results)
        {
            var objectTypeCount = result.ObjectTypes?.Count ?? 0;
            var materialTypeCount = result.MaterialTypes?.Count ?? 0;
            
            lines.Add($"{result.FileName},{result.FileSize},{result.VertexCount},{result.SurfaceCount},{result.LinkCount},{result.HasValidData},{objectTypeCount},{materialTypeCount}");
        }
        
        await File.WriteAllLinesAsync(csvFile, lines);
    }
}

// Data classes for pipeline results
public class PipelineResults
{
    public MPRLMeshResult MprlMeshResult { get; set; } = new();
    public List<IndividualPM4Analysis> IndividualAnalysisResults { get; set; } = new();
    public CrossFileAnalysisResult CrossFileAnalysis { get; set; } = new();
}

public class MPRLMeshResult
{
    public bool Success { get; set; }
    public int TotalVertices { get; set; }
    public int ProcessedFiles { get; set; }
    public int SkippedFiles { get; set; }
}

public class IndividualPM4Analysis
{
    public string FileName { get; set; } = string.Empty;
    public string FilePath { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public int VertexCount { get; set; }
    public int SurfaceCount { get; set; }
    public int LinkCount { get; set; }
    public bool HasValidData { get; set; }
    public Dictionary<byte, int>? ObjectTypes { get; set; }
    public Dictionary<uint, int>? MaterialTypes { get; set; }
    
    public string ToDetailedString()
    {
        var result = new List<string>
        {
            $"=== PM4 File Analysis: {FileName} ===",
            $"File Path: {FilePath}",
            $"File Size: {FileSize:N0} bytes",
            $"Vertex Count: {VertexCount:N0}",
            $"Surface Count: {SurfaceCount:N0}",
            $"Link Count: {LinkCount:N0}",
            $"Has Valid Data: {HasValidData}",
            ""
        };
        
        if (ObjectTypes != null && ObjectTypes.Count > 0)
        {
            result.Add("Object Type Distribution:");
            foreach (var kvp in ObjectTypes.OrderByDescending(x => x.Value))
            {
                result.Add($"  Type {kvp.Key}: {kvp.Value:N0} entries");
            }
            result.Add("");
        }
        
        if (MaterialTypes != null && MaterialTypes.Count > 0)
        {
            result.Add("Material Type Distribution:");
            foreach (var kvp in MaterialTypes.OrderByDescending(x => x.Value).Take(10))
            {
                result.Add($"  Material 0x{kvp.Key:X8}: {kvp.Value:N0} entries");
            }
            if (MaterialTypes.Count > 10)
            {
                result.Add($"  ... and {MaterialTypes.Count - 10} more material types");
            }
            result.Add("");
        }
        
        return string.Join(Environment.NewLine, result);
    }
}

public class CrossFileAnalysisResult
{
    public int TotalFilesAnalyzed { get; set; }
    public int TotalVertices { get; set; }
    public int TotalSurfaces { get; set; }
    public int TotalLinks { get; set; }
    public int FilesWithValidData { get; set; }
    public Dictionary<byte, int> GlobalObjectTypeDistribution { get; set; } = new();
    public Dictionary<uint, int> GlobalMaterialDistribution { get; set; } = new();
    
    public string ToDetailedString()
    {
        var result = new List<string>
        {
            "=== Cross-File Analysis Results ===",
            $"Total Files Analyzed: {TotalFilesAnalyzed:N0}",
            $"Files with Valid Data: {FilesWithValidData:N0} ({(double)FilesWithValidData / TotalFilesAnalyzed * 100:F1}%)",
            $"Total Vertices: {TotalVertices:N0}",
            $"Total Surfaces: {TotalSurfaces:N0}",
            $"Total Links: {TotalLinks:N0}",
            "",
            "Global Object Type Distribution:"
        };
        
        foreach (var kvp in GlobalObjectTypeDistribution.OrderByDescending(x => x.Value))
        {
            var typeName = GetObjectTypeName(kvp.Key);
            result.Add($"  {typeName} (Type {kvp.Key}): {kvp.Value:N0} entries");
        }
        
        result.Add("");
        result.Add("Top Material Types:");
        foreach (var kvp in GlobalMaterialDistribution.OrderByDescending(x => x.Value).Take(15))
        {
            result.Add($"  Material 0x{kvp.Key:X8}: {kvp.Value:N0} entries");
        }
        
        return string.Join(Environment.NewLine, result);
    }
    
    private string GetObjectTypeName(byte objectType)
    {
        return objectType switch
        {
            1 => "Terrain Base",
            2 => "Structure Foundation", 
            4 => "Doodad Placement",
            10 => "Terrain Detail",
            12 => "Object Reference",
            17 => "Special Feature",
            _ => $"Type {objectType}"
        };
    }
}

public class CombinedMPRLMesh
{
    public List<Vector3> AllVertices { get; set; } = new();
    public Dictionary<string, List<Vector3>> VerticesByFile { get; set; } = new();
    public List<string> ProcessedFiles { get; set; } = new();
} 