using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PM4Rebuilder;

/// <summary>
/// Comprehensive summary of a PM4 unified architecture export operation.
/// Contains statistics, validation results, and export metadata.
/// </summary>
public class PM4ExportSummary
{
    /// <summary>
    /// Export timing information.
    /// </summary>
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan TotalDuration { get; set; }

    /// <summary>
    /// Input/output paths and strategy.
    /// </summary>
    public string InputDirectory { get; set; } = string.Empty;
    public string OutputDirectory { get; set; } = string.Empty;
    public PM4ExportStrategy ExportStrategy { get; set; }

    /// <summary>
    /// Unified map statistics.
    /// </summary>
    public int TileCount { get; set; }
    public int TotalVertices { get; set; }
    public int TotalIndices { get; set; }
    public int TotalLinkageEntries { get; set; }

    /// <summary>
    /// Building assembly results.
    /// </summary>
    public int BuildingCount { get; set; }
    public List<PM4Building> Buildings { get; set; } = new();

    /// <summary>
    /// Building scale validation results.
    /// </summary>
    public int MinTrianglesPerBuilding { get; set; }
    public int MaxTrianglesPerBuilding { get; set; }
    public int AvgTrianglesPerBuilding { get; set; }

    /// <summary>
    /// Export results.
    /// </summary>
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public int ExportedFileCount { get; set; }
    public List<string> ExportErrors { get; set; } = new();
    public List<string> ValidationIssues { get; set; } = new();

    /// <summary>
    /// Quality metrics.
    /// </summary>
    public PM4ExportQuality Quality => CalculateQuality();

    /// <summary>
    /// Calculate overall export quality based on various metrics.
    /// </summary>
    private PM4ExportQuality CalculateQuality()
    {
        var quality = new PM4ExportQuality();

        // Expected values from memory bank
        var expectedBuildingCount = 458;
        var expectedMinTriangles = 38000;

        // Building count quality
        if (BuildingCount >= expectedBuildingCount * 0.8f)
            quality.BuildingCountQuality = PM4QualityLevel.Excellent;
        else if (BuildingCount >= expectedBuildingCount * 0.5f)
            quality.BuildingCountQuality = PM4QualityLevel.Good;
        else if (BuildingCount >= expectedBuildingCount * 0.25f)
            quality.BuildingCountQuality = PM4QualityLevel.Fair;
        else
            quality.BuildingCountQuality = PM4QualityLevel.Poor;

        // Building scale quality
        if (AvgTrianglesPerBuilding >= expectedMinTriangles)
            quality.BuildingScaleQuality = PM4QualityLevel.Excellent;
        else if (MaxTrianglesPerBuilding >= expectedMinTriangles)
            quality.BuildingScaleQuality = PM4QualityLevel.Good;
        else if (MaxTrianglesPerBuilding >= expectedMinTriangles * 0.1f)
            quality.BuildingScaleQuality = PM4QualityLevel.Fair;
        else
            quality.BuildingScaleQuality = PM4QualityLevel.Poor;

        // Cross-tile reference quality (based on vertex coverage)
        if (TotalVertices > 800000) // Expected from unified map loading
            quality.CrossTileQuality = PM4QualityLevel.Excellent;
        else if (TotalVertices > 400000)
            quality.CrossTileQuality = PM4QualityLevel.Good;
        else if (TotalVertices > 100000)
            quality.CrossTileQuality = PM4QualityLevel.Fair;
        else
            quality.CrossTileQuality = PM4QualityLevel.Poor;

        // Export success quality
        if (Success && ExportErrors.Count == 0)
            quality.ExportSuccessQuality = PM4QualityLevel.Excellent;
        else if (Success && ExportErrors.Count <= 3)
            quality.ExportSuccessQuality = PM4QualityLevel.Good;
        else if (Success)
            quality.ExportSuccessQuality = PM4QualityLevel.Fair;
        else
            quality.ExportSuccessQuality = PM4QualityLevel.Poor;

        // Calculate overall quality
        var qualityLevels = new[] 
        { 
            quality.BuildingCountQuality, 
            quality.BuildingScaleQuality, 
            quality.CrossTileQuality, 
            quality.ExportSuccessQuality 
        };

        var avgQuality = qualityLevels.Average(q => (int)q);
        quality.OverallQuality = (PM4QualityLevel)Math.Round(avgQuality);

        return quality;
    }

    /// <summary>
    /// Generate a comprehensive human-readable report.
    /// </summary>
    public string GenerateReport()
    {
        var report = new StringBuilder();
        
        report.AppendLine("PM4 UNIFIED ARCHITECTURE EXPORT REPORT");
        report.AppendLine("=" + new string('=', 60));
        report.AppendLine();

        // Basic information
        report.AppendLine("EXPORT INFORMATION");
        report.AppendLine("-" + new string('-', 30));
        report.AppendLine($"Start Time: {StartTime:yyyy-MM-dd HH:mm:ss} UTC");
        report.AppendLine($"End Time: {EndTime:yyyy-MM-dd HH:mm:ss} UTC");
        report.AppendLine($"Duration: {TotalDuration.TotalSeconds:F1} seconds");
        report.AppendLine($"Input Directory: {InputDirectory}");
        report.AppendLine($"Output Directory: {OutputDirectory}");
        report.AppendLine($"Export Strategy: {ExportStrategy}");
        report.AppendLine($"Success: {Success}");
        if (!Success && !string.IsNullOrEmpty(ErrorMessage))
        {
            report.AppendLine($"Error: {ErrorMessage}");
        }
        report.AppendLine();

        // Unified map statistics
        report.AppendLine("UNIFIED MAP STATISTICS");
        report.AppendLine("-" + new string('-', 30));
        report.AppendLine($"Tiles Loaded: {TileCount}");
        report.AppendLine($"Total Vertices: {TotalVertices:N0}");
        report.AppendLine($"Total Indices: {TotalIndices:N0}");
        report.AppendLine($"Total Linkage Entries: {TotalLinkageEntries:N0}");
        report.AppendLine();

        // Building assembly results
        report.AppendLine("BUILDING ASSEMBLY RESULTS");
        report.AppendLine("-" + new string('-', 30));
        report.AppendLine($"Buildings Assembled: {BuildingCount}");
        if (BuildingCount > 0)
        {
            report.AppendLine($"Triangle Range: {MinTrianglesPerBuilding:N0} - {MaxTrianglesPerBuilding:N0}");
            report.AppendLine($"Average Triangles: {AvgTrianglesPerBuilding:N0}");
            
            var buildingScaleCount = Buildings.Count(b => b.TriangleCount >= 38000);
            report.AppendLine($"Building-Scale Objects: {buildingScaleCount} ({(buildingScaleCount * 100f / BuildingCount):F1}%)");
        }
        report.AppendLine();

        // Export results
        report.AppendLine("EXPORT RESULTS");
        report.AppendLine("-" + new string('-', 30));
        report.AppendLine($"Files Exported: {ExportedFileCount}");
        report.AppendLine($"Export Errors: {ExportErrors.Count}");
        if (ExportErrors.Any())
        {
            foreach (var error in ExportErrors.Take(5))
            {
                report.AppendLine($"  - {error}");
            }
            if (ExportErrors.Count > 5)
            {
                report.AppendLine($"  ... and {ExportErrors.Count - 5} more errors");
            }
        }
        report.AppendLine();

        // Validation issues
        if (ValidationIssues.Any())
        {
            report.AppendLine("VALIDATION ISSUES");
            report.AppendLine("-" + new string('-', 30));
            foreach (var issue in ValidationIssues.Take(10))
            {
                report.AppendLine($"  - {issue}");
            }
            if (ValidationIssues.Count > 10)
            {
                report.AppendLine($"  ... and {ValidationIssues.Count - 10} more issues");
            }
            report.AppendLine();
        }

        // Quality assessment
        report.AppendLine("QUALITY ASSESSMENT");
        report.AppendLine("-" + new string('-', 30));
        var quality = Quality;
        report.AppendLine($"Overall Quality: {quality.OverallQuality}");
        report.AppendLine($"Building Count: {quality.BuildingCountQuality}");
        report.AppendLine($"Building Scale: {quality.BuildingScaleQuality}");
        report.AppendLine($"Cross-Tile Resolution: {quality.CrossTileQuality}");
        report.AppendLine($"Export Success: {quality.ExportSuccessQuality}");
        report.AppendLine();

        // Memory bank comparison
        report.AppendLine("MEMORY BANK COMPARISON");
        report.AppendLine("-" + new string('-', 30));
        report.AppendLine($"Expected Buildings: ~458");
        report.AppendLine($"Actual Buildings: {BuildingCount} ({(BuildingCount * 100f / 458f):F1}% of expected)");
        report.AppendLine($"Expected Triangle Range: 38K - 654K per building");
        report.AppendLine($"Actual Triangle Range: {MinTrianglesPerBuilding:N0} - {MaxTrianglesPerBuilding:N0}");
        
        var architecturalGoalsMet = BuildingCount >= 100 && MaxTrianglesPerBuilding >= 38000;
        report.AppendLine($"Unified Architecture Goals Met: {(architecturalGoalsMet ? "✅ YES" : "❌ NO")}");
        report.AppendLine();

        // Recommendations
        report.AppendLine("RECOMMENDATIONS");
        report.AppendLine("-" + new string('-', 30));
        
        if (quality.OverallQuality == PM4QualityLevel.Excellent)
        {
            report.AppendLine("✅ Export quality is excellent! The unified architecture is working correctly.");
        }
        else
        {
            if (quality.BuildingCountQuality == PM4QualityLevel.Poor)
            {
                report.AppendLine("- Investigate PM4 linkage system - building count is too low");
            }
            
            if (quality.BuildingScaleQuality == PM4QualityLevel.Poor)
            {
                report.AppendLine("- Check building assembly logic - objects are fragmented");
            }
            
            if (quality.CrossTileQuality == PM4QualityLevel.Poor)
            {
                report.AppendLine("- Verify cross-tile reference resolution - vertex count is too low");
            }
            
            if (quality.ExportSuccessQuality == PM4QualityLevel.Poor)
            {
                report.AppendLine("- Fix export errors and validation issues");
            }
        }
        
        report.AppendLine();
        report.AppendLine("Generated by PM4 Unified Architecture Export System");
        report.AppendLine($"Report timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");

        return report.ToString();
    }

    /// <summary>
    /// Get a short summary line for logging.
    /// </summary>
    public override string ToString()
    {
        var status = Success ? "SUCCESS" : "FAILED";
        return $"PM4 Export {status}: {BuildingCount} buildings, {ExportedFileCount} files, {TotalDuration.TotalSeconds:F1}s";
    }
}

/// <summary>
/// Quality assessment for PM4 export operation.
/// </summary>
public class PM4ExportQuality
{
    public PM4QualityLevel OverallQuality { get; set; }
    public PM4QualityLevel BuildingCountQuality { get; set; }
    public PM4QualityLevel BuildingScaleQuality { get; set; }
    public PM4QualityLevel CrossTileQuality { get; set; }
    public PM4QualityLevel ExportSuccessQuality { get; set; }
}

/// <summary>
/// Quality level enumeration.
/// </summary>
public enum PM4QualityLevel
{
    Poor = 1,
    Fair = 2,
    Good = 3,
    Excellent = 4
}
