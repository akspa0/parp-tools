using System.CommandLine;
using System.Numerics;
using System.Text.Json;
using ParpToolbox.Services;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands;

public static class BatchMscnWmoCorrelationCommand
{
    public static Command CreateCommand()
    {
        var command = new Command("batch-mscn-wmo-correlation", "Batch correlate all PM4 MSCN anchors with all available WMO files");
        
        var pm4DirectoryOption = new Option<string>("--pm4-dir", "Directory containing PM4 files") { IsRequired = true };
        var wmoDirectoryOption = new Option<string>("--wmo-dir", "Directory containing WMO files") { IsRequired = true };
        var outputDirectoryOption = new Option<string>("--output-dir", "Output directory for results") { IsRequired = true };
        var toleranceOption = new Option<float>("--tolerance", () => 5.0f, "Distance tolerance for spatial matching");
        var parallelismOption = new Option<int>("--parallelism", () => Environment.ProcessorCount, "Number of parallel processing threads");
        var minMatchThresholdOption = new Option<float>("--min-match-threshold", () => 1.0f, "Minimum match percentage to include in results");
        var cascModeOption = new Option<bool>("--casc-mode", () => false, "Use CASC file discovery instead of local files");
        
        command.AddOption(pm4DirectoryOption);
        command.AddOption(wmoDirectoryOption);
        command.AddOption(outputDirectoryOption);
        command.AddOption(toleranceOption);
        command.AddOption(parallelismOption);
        command.AddOption(minMatchThresholdOption);
        command.AddOption(cascModeOption);
        
        command.SetHandler(async (pm4Dir, wmoDir, outputDir, tolerance, parallelism, minMatchThreshold, cascMode) =>
        {
            await ExecuteBatchCorrelation(pm4Dir, wmoDir, outputDir, tolerance, parallelism, minMatchThreshold, cascMode);
        }, pm4DirectoryOption, wmoDirectoryOption, outputDirectoryOption, toleranceOption, parallelismOption, minMatchThresholdOption, cascModeOption);
        
        return command;
    }
    
    private static async Task ExecuteBatchCorrelation(
        string pm4Directory, 
        string wmoDirectory, 
        string outputDirectory, 
        float tolerance, 
        int parallelism, 
        float minMatchThreshold,
        bool cascMode)
    {
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var sessionOutputDir = Path.Combine(outputDirectory, $"batch_correlation_{timestamp}");
        Directory.CreateDirectory(sessionOutputDir);
        
        ConsoleLogger.WriteLine($"=== BATCH MSCN-WMO CORRELATION ===");
        ConsoleLogger.WriteLine($"PM4 Directory: {pm4Directory}");
        ConsoleLogger.WriteLine($"WMO Directory: {wmoDirectory}");
        ConsoleLogger.WriteLine($"Output Directory: {sessionOutputDir}");
        ConsoleLogger.WriteLine($"Tolerance: {tolerance}");
        ConsoleLogger.WriteLine($"Parallelism: {parallelism}");
        ConsoleLogger.WriteLine($"Min Match Threshold: {minMatchThreshold}%");
        ConsoleLogger.WriteLine($"CASC Mode: {cascMode}");
        ConsoleLogger.WriteLine();
        
        // Discover files
        var pm4Files = await DiscoverPm4Files(pm4Directory, cascMode);
        var wmoFiles = await DiscoverWmoFiles(wmoDirectory, cascMode);
        
        ConsoleLogger.WriteLine($"Discovered {pm4Files.Count} PM4 files");
        ConsoleLogger.WriteLine($"Discovered {wmoFiles.Count} WMO files");
        ConsoleLogger.WriteLine();
        
        // Create correlation matrix
        var correlationResults = new List<BatchCorrelationResult>();
        var totalCombinations = pm4Files.Count * wmoFiles.Count;
        
        // Process in parallel batches
        var semaphore = new SemaphoreSlim(parallelism);
        var tasks = new List<Task>();
        
        foreach (var pm4File in pm4Files)
        {
            foreach (var wmoFile in wmoFiles)
            {
                tasks.Add(ProcessCorrelationPair(pm4File, wmoFile, tolerance, minMatchThreshold, correlationResults, semaphore, totalCombinations));
            }
        }
        
        await Task.WhenAll(tasks);
        
        // Generate comprehensive report
        await GenerateBatchReport(correlationResults, sessionOutputDir, tolerance, minMatchThreshold);
        
        ConsoleLogger.WriteLine($"\\nBatch correlation completed. Results saved to: {sessionOutputDir}");
    }
    
    private static async Task<List<string>> DiscoverPm4Files(string directory, bool cascMode)
    {
        if (cascMode)
        {
            // TODO: Implement CASC discovery via wow.tools.local
            ConsoleLogger.WriteLine("CASC mode not yet implemented, falling back to local file discovery");
        }
        
        var pm4Files = new List<string>();
        
        if (Directory.Exists(directory))
        {
            pm4Files.AddRange(Directory.GetFiles(directory, "*.pm4", SearchOption.AllDirectories));
        }
        
        return pm4Files;
    }
    
    private static async Task<List<string>> DiscoverWmoFiles(string directory, bool cascMode)
    {
        if (cascMode)
        {
            // TODO: Implement CASC discovery via wow.tools.local
            ConsoleLogger.WriteLine("CASC mode not yet implemented, falling back to local file discovery");
        }
        
        var wmoFiles = new List<string>();
        
        if (Directory.Exists(directory))
        {
            wmoFiles.AddRange(Directory.GetFiles(directory, "*.wmo", SearchOption.AllDirectories));
        }
        
        return wmoFiles;
    }
    
    private static async Task ProcessCorrelationPair(
        string pm4File, 
        string wmoFile, 
        float tolerance, 
        float minMatchThreshold,
        List<BatchCorrelationResult> results,
        SemaphoreSlim semaphore,
        int totalCombinations)
    {
        await semaphore.WaitAsync();
        
        try
        {
            // Extract tile coordinates from PM4 filename for intelligent matching
            var pm4TileInfo = ExtractTileInfo(pm4File);
            var wmoInfo = ExtractWmoInfo(wmoFile);
            
            // Skip obviously unrelated pairs (performance optimization)
            if (!ShouldProcessPair(pm4TileInfo, wmoInfo))
            {
                return;
            }
            
            try
            {
                // Load MSCN anchors from PM4
                var mscnVertices = await LoadMscnAnchorsFromPm4(pm4File, pm4TileInfo);
                
                // Load WMO geometry
                var wmoVertices = await LoadWmoGeometry(wmoFile);
                
                if (mscnVertices.Count == 0 || wmoVertices.Count == 0)
                {
                    return; // Skip empty files
                }
                
                // Perform spatial correlation
                var correlationResult = AnalyzeCorrelation(mscnVertices, wmoVertices, tolerance);
                
                // Only include results above threshold
                if (correlationResult.MatchPercentage >= minMatchThreshold)
                {
                    var batchResult = new BatchCorrelationResult
                    {
                        Pm4File = pm4File,
                        WmoFile = wmoFile,
                        Pm4TileInfo = pm4TileInfo,
                        WmoInfo = wmoInfo,
                        MatchPercentage = correlationResult.MatchPercentage,
                        TotalMatches = correlationResult.TotalMatches,
                        MscnVertexCount = mscnVertices.Count,
                        WmoVertexCount = wmoVertices.Count,
                        ProcessingTime = DateTime.Now
                    };
                    
                    lock (results)
                    {
                        results.Add(batchResult);
                    }
                    
                    ConsoleLogger.WriteLine($"MATCH: {Path.GetFileName(pm4File)} <-> {Path.GetFileName(wmoFile)} ({correlationResult.MatchPercentage:F1}%, {correlationResult.TotalMatches} matches)");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error processing {Path.GetFileName(pm4File)} <-> {Path.GetFileName(wmoFile)}: {ex.Message}");
            }
        }
        finally
        {
            semaphore.Release();
        }
    }
    
    private static TileInfo ExtractTileInfo(string pm4FilePath)
    {
        var filename = Path.GetFileNameWithoutExtension(pm4FilePath);
        
        // Extract tile coordinates from filename (e.g., "development_15_37.pm4" -> X=15, Y=37)
        var parts = filename.Split('_');
        if (parts.Length >= 3 && 
            int.TryParse(parts[^2], out int tileX) && 
            int.TryParse(parts[^1], out int tileY))
        {
            return new TileInfo { TileX = tileX, TileY = tileY, Region = string.Join("_", parts.Take(parts.Length - 2)) };
        }
        
        return new TileInfo { TileX = -1, TileY = -1, Region = filename };
    }
    
    private static WmoInfo ExtractWmoInfo(string wmoFilePath)
    {
        var filename = Path.GetFileNameWithoutExtension(wmoFilePath);
        var directory = Path.GetDirectoryName(wmoFilePath) ?? "";
        
        // Extract region/zone information from path
        var pathParts = directory.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var region = pathParts.Length > 0 ? pathParts[^1] : "Unknown";
        
        return new WmoInfo { Name = filename, Region = region, FullPath = wmoFilePath };
    }
    
    private static bool ShouldProcessPair(TileInfo pm4Info, WmoInfo wmoInfo)
    {
        // Implement intelligent filtering logic here
        // For now, process all pairs, but this could be optimized based on:
        // - Geographic proximity (tile coordinates vs WMO region)
        // - Naming patterns (e.g., Stormwind PM4s with Stormwind WMOs)
        // - File size heuristics
        
        return true;
    }
    
    private static async Task<List<Vector3>> LoadMscnAnchorsFromPm4(string pm4FilePath, TileInfo tileInfo)
    {
        // Reuse the existing MSCN extraction logic from MscnWmoComparisonCommand
        return MscnWmoComparisonCommand.ExtractMscnAnchorsFromPm4(pm4FilePath);
    }
    
    private static async Task<List<Vector3>> LoadWmoGeometry(string wmoFilePath)
    {
        // Reuse the existing WMO loading logic from MscnWmoComparisonCommand
        return MscnWmoComparisonCommand.LoadWmoVertices(wmoFilePath, null);
    }
    
    private static MscnWmoComparisonCommand.SpatialCorrelationResult AnalyzeCorrelation(List<Vector3> mscnVertices, List<Vector3> wmoVertices, float tolerance)
    {
        // Reuse the existing correlation analysis from MscnWmoComparisonCommand
        return MscnWmoComparisonCommand.AnalyzeObjectCorrelation(mscnVertices, wmoVertices, tolerance, null);
    }
    
    private static async Task GenerateBatchReport(List<BatchCorrelationResult> results, string outputDir, float tolerance, float minMatchThreshold)
    {
        // Sort results by match percentage (best matches first)
        var sortedResults = results.OrderByDescending(r => r.MatchPercentage).ToList();
        
        // Generate summary report
        var reportPath = Path.Combine(outputDir, "batch_correlation_report.txt");
        using (var writer = new StreamWriter(reportPath))
        {
            await writer.WriteLineAsync("=== BATCH MSCN-WMO CORRELATION REPORT ===");
            await writer.WriteLineAsync($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            await writer.WriteLineAsync($"Tolerance: {tolerance}");
            await writer.WriteLineAsync($"Min Match Threshold: {minMatchThreshold}%");
            await writer.WriteLineAsync($"Total Correlations Found: {sortedResults.Count}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync("=== TOP CORRELATIONS ===");
            foreach (var result in sortedResults.Take(50)) // Top 50 matches
            {
                await writer.WriteLineAsync($"{result.MatchPercentage:F2}% | {Path.GetFileName(result.Pm4File)} <-> {Path.GetFileName(result.WmoFile)} | {result.TotalMatches} matches");
            }
            
            await writer.WriteLineAsync();
            await writer.WriteLineAsync("=== STATISTICS ===");
            if (sortedResults.Count > 0)
            {
                await writer.WriteLineAsync($"Best Match: {sortedResults[0].MatchPercentage:F2}%");
                await writer.WriteLineAsync($"Average Match: {sortedResults.Average(r => r.MatchPercentage):F2}%");
                await writer.WriteLineAsync($"Median Match: {sortedResults[sortedResults.Count / 2].MatchPercentage:F2}%");
            }
        }
        
        // Generate detailed JSON results
        var jsonPath = Path.Combine(outputDir, "batch_correlation_results.json");
        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
        var jsonContent = JsonSerializer.Serialize(sortedResults, jsonOptions);
        await File.WriteAllTextAsync(jsonPath, jsonContent);
        
        ConsoleLogger.WriteLine($"Reports generated:");
        ConsoleLogger.WriteLine($"  - Summary: {Path.GetFileName(reportPath)}");
        ConsoleLogger.WriteLine($"  - Detailed JSON: {Path.GetFileName(jsonPath)}");
    }
}

public class TileInfo
{
    public int TileX { get; set; }
    public int TileY { get; set; }
    public string Region { get; set; } = "";
}

public class WmoInfo
{
    public string Name { get; set; } = "";
    public string Region { get; set; } = "";
    public string FullPath { get; set; } = "";
}

public class BatchCorrelationResult
{
    public string Pm4File { get; set; } = "";
    public string WmoFile { get; set; } = "";
    public TileInfo Pm4TileInfo { get; set; } = new();
    public WmoInfo WmoInfo { get; set; } = new();
    public float MatchPercentage { get; set; }
    public int TotalMatches { get; set; }
    public int MscnVertexCount { get; set; }
    public int WmoVertexCount { get; set; }
    public DateTime ProcessingTime { get; set; }
}
