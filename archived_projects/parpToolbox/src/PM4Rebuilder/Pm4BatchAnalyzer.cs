using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Batch analyzer for PM4 data recovery. Processes ALL PM4 files in a directory,
    /// handling corruption gracefully and extracting whatever data is available.
    /// </summary>
    internal static class Pm4BatchAnalyzer
    {
        public static async Task<int> ProcessAllFiles(string inputDirectory, string outputDirectory)
        {
            Console.WriteLine($"[BATCH ANALYZER] Starting data recovery batch processing...");
            Console.WriteLine($"[BATCH ANALYZER] Input: {inputDirectory}");
            Console.WriteLine($"[BATCH ANALYZER] Output: {outputDirectory}");
            
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var batchOutputDir = Path.Combine(outputDirectory, $"batch_analysis_{timestamp}");
            Directory.CreateDirectory(batchOutputDir);
            
            var individualDir = Path.Combine(batchOutputDir, "individual_files");
            var mergedDir = Path.Combine(batchOutputDir, "merged_analysis");
            Directory.CreateDirectory(individualDir);
            Directory.CreateDirectory(mergedDir);
            
            // Find ALL PM4 files
            var pm4Files = Directory.GetFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly);
            Console.WriteLine($"[BATCH ANALYZER] Found {pm4Files.Length} PM4 files to process");
            
            var results = new List<BatchProcessingResult>();
            var allChunkData = new Dictionary<string, List<Dictionary<string, object>>>();
            
            // Process each file with error handling
            for (int i = 0; i < pm4Files.Length; i++)
            {
                var pm4File = pm4Files[i];
                var fileName = Path.GetFileNameWithoutExtension(pm4File);
                
                Console.WriteLine($"[BATCH ANALYZER] Processing {i + 1}/{pm4Files.Length}: {fileName}");
                
                var result = await ProcessSingleFile(pm4File, individualDir);
                results.Add(result);
                
                // Accumulate chunk data for merging (only from successful files)
                if (result.Success && result.ChunkData != null)
                {
                    foreach (var kvp in result.ChunkData)
                    {
                        if (!allChunkData.ContainsKey(kvp.Key))
                            allChunkData[kvp.Key] = new List<Dictionary<string, object>>();
                        
                        allChunkData[kvp.Key].AddRange(kvp.Value);
                    }
                }
                
                // Progress update every 10 files
                if ((i + 1) % 10 == 0)
                {
                    var successCount = results.Count(r => r.Success);
                    Console.WriteLine($"[BATCH ANALYZER] Progress: {i + 1}/{pm4Files.Length} ({successCount} successful)");
                }
            }
            
            // Generate merged analysis
            await GenerateMergedAnalysis(allChunkData, mergedDir);
            
            // Generate summary report
            await GenerateSummaryReport(results, batchOutputDir);
            
            var totalSuccess = results.Count(r => r.Success);
            Console.WriteLine($"[BATCH ANALYZER] Batch processing complete!");
            Console.WriteLine($"[BATCH ANALYZER] Successfully processed: {totalSuccess}/{pm4Files.Length} files");
            Console.WriteLine($"[BATCH ANALYZER] Output directory: {batchOutputDir}");
            
            return 0;
        }
        
        private static async Task<BatchProcessingResult> ProcessSingleFile(string pm4FilePath, string individualOutputDir)
        {
            var fileName = Path.GetFileNameWithoutExtension(pm4FilePath);
            var fileOutputDir = Path.Combine(individualOutputDir, fileName);
            
            try
            {
                Directory.CreateDirectory(fileOutputDir);
                
                // Load PM4 scene with error handling
                var scene = await LoadPm4WithErrorHandling(pm4FilePath);
                if (scene == null)
                {
                    return new BatchProcessingResult
                    {
                        FileName = fileName,
                        Success = false,
                        ErrorMessage = "Failed to load PM4 file",
                        ChunkData = null
                    };
                }
                
                // Run data validation and CSV export
                Pm4DataValidator.ValidateAndExport(scene, fileOutputDir);
                
                // Extract chunk data for merging
                var chunkData = ExtractChunkDataForMerging(scene, fileName);
                
                return new BatchProcessingResult
                {
                    FileName = fileName,
                    Success = true,
                    ErrorMessage = null,
                    ChunkData = chunkData,
                    VertexCount = scene.Vertices?.Count ?? 0,
                    TriangleCount = scene.Indices?.Count ?? 0,
                    MscnCount = scene.MscnVertices?.Count ?? 0,
                    MslkCount = scene.Links?.Count ?? 0,
                    MsurCount = scene.Surfaces?.Count ?? 0,
                    MprlCount = scene.Placements?.Count ?? 0
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BATCH ANALYZER] ERROR processing {fileName}: {ex.Message}");
                
                // Try to extract partial data even from corrupted files
                var partialData = await TryExtractPartialData(pm4FilePath, fileOutputDir);
                
                return new BatchProcessingResult
                {
                    FileName = fileName,
                    Success = false,
                    ErrorMessage = ex.Message,
                    ChunkData = partialData
                };
            }
        }
        
        private static async Task<Pm4Scene?> LoadPm4WithErrorHandling(string pm4FilePath)
        {
            try
            {
                // Try multiple loading approaches
                var adapter = new Pm4Adapter();
                
                // First try: normal single file load
                try
                {
                    return await Task.Run(() => adapter.Load(pm4FilePath));
                }
                catch
                {
                    // Second try: region load with just this file
                    return await Task.Run(() => adapter.LoadRegion(pm4FilePath));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BATCH ANALYZER] Failed to load {pm4FilePath}: {ex.Message}");
                return null;
            }
        }
        
        private static async Task<Dictionary<string, List<Dictionary<string, object>>>?> TryExtractPartialData(string pm4FilePath, string outputDir)
        {
            try
            {
                // Attempt to read raw file and extract whatever chunks we can
                Console.WriteLine($"[BATCH ANALYZER] Attempting partial data recovery for {Path.GetFileName(pm4FilePath)}");
                
                // This is a placeholder for more sophisticated partial extraction
                // For now, just log that we attempted it
                await File.WriteAllTextAsync(Path.Combine(outputDir, "partial_recovery_attempted.txt"), 
                    $"Attempted partial data recovery at {DateTime.Now}\nFile: {pm4FilePath}\n");
                
                return null;
            }
            catch
            {
                return null;
            }
        }
        
        private static Dictionary<string, List<Dictionary<string, object>>> ExtractChunkDataForMerging(Pm4Scene scene, string fileName)
        {
            var chunkData = new Dictionary<string, List<Dictionary<string, object>>>();
            
            // MSVT data
            if (scene.Vertices?.Any() == true)
            {
                chunkData["MSVT"] = scene.Vertices.Select((v, i) => new Dictionary<string, object>
                {
                    ["SourceFile"] = fileName,
                    ["Index"] = i,
                    ["X"] = v.X,
                    ["Y"] = v.Y,
                    ["Z"] = v.Z
                }).ToList<Dictionary<string, object>>();
            }
            
            // MSCN data
            if (scene.MscnVertices?.Any() == true)
            {
                chunkData["MSCN"] = scene.MscnVertices.Select((v, i) => new Dictionary<string, object>
                {
                    ["SourceFile"] = fileName,
                    ["Index"] = i,
                    ["X"] = v.X,
                    ["Y"] = v.Y,
                    ["Z"] = v.Z
                }).ToList<Dictionary<string, object>>();
            }
            
            // MSVI data
            if (scene.Indices?.Any() == true)
            {
                chunkData["MSVI"] = scene.Indices.Select((idx, i) => new Dictionary<string, object>
                {
                    ["SourceFile"] = fileName,
                    ["Index"] = i,
                    ["VertexIndex"] = idx
                }).ToList<Dictionary<string, object>>();
            }
            
            // MSLK data
            if (scene.Links?.Any() == true)
            {
                chunkData["MSLK"] = scene.Links.Select((l, i) => {
                    var dict = new Dictionary<string, object>
                    {
                        ["SourceFile"] = fileName,
                        ["Index"] = i
                    };
                    
                    // Use reflection to get all properties dynamically
                    var properties = l.GetType().GetProperties();
                    foreach (var prop in properties)
                    {
                        try
                        {
                            var value = prop.GetValue(l);
                            dict[prop.Name] = value ?? "";
                        }
                        catch
                        {
                            dict[prop.Name] = "";
                        }
                    }
                    
                    return dict;
                }).ToList<Dictionary<string, object>>();
            }
            
            // MSUR data
            if (scene.Surfaces?.Any() == true)
            {
                chunkData["MSUR"] = scene.Surfaces.Select((s, i) => {
                    var dict = new Dictionary<string, object>
                    {
                        ["SourceFile"] = fileName,
                        ["Index"] = i
                    };
                    
                    // Use reflection to get all properties dynamically
                    var properties = s.GetType().GetProperties();
                    foreach (var prop in properties)
                    {
                        try
                        {
                            var value = prop.GetValue(s);
                            dict[prop.Name] = value ?? "";
                        }
                        catch
                        {
                            dict[prop.Name] = "";
                        }
                    }
                    
                    return dict;
                }).ToList<Dictionary<string, object>>();
            }
            
            // MPRL data
            if (scene.Placements?.Any() == true)
            {
                chunkData["MPRL"] = scene.Placements.Select((p, i) => {
                    var dict = new Dictionary<string, object>
                    {
                        ["SourceFile"] = fileName,
                        ["Index"] = i,
                        ["X"] = p.Position.X,
                        ["Y"] = p.Position.Y,
                        ["Z"] = p.Position.Z
                    };
                    
                    // Use reflection to get all properties dynamically
                    var properties = p.GetType().GetProperties();
                    foreach (var prop in properties)
                    {
                        if (prop.Name == "Position") continue; // Already handled above
                        
                        try
                        {
                            var value = prop.GetValue(p);
                            dict[prop.Name] = value ?? "";
                        }
                        catch
                        {
                            dict[prop.Name] = "";
                        }
                    }
                    
                    return dict;
                }).ToList<Dictionary<string, object>>();
            }
            
            return chunkData;
        }
        
        private static async Task GenerateMergedAnalysis(Dictionary<string, List<Dictionary<string, object>>> allChunkData, string mergedDir)
        {
            Console.WriteLine($"[BATCH ANALYZER] Generating merged analysis...");
            
            foreach (var kvp in allChunkData)
            {
                var chunkType = kvp.Key;
                var data = kvp.Value;
                
                if (!data.Any()) continue;
                
                var csvPath = Path.Combine(mergedDir, $"all_{chunkType.ToLower()}_merged.csv");
                await WriteMergedCSV(data, csvPath);
                
                Console.WriteLine($"[BATCH ANALYZER] Wrote merged {chunkType}: {data.Count} entries from {data.Select(d => d["SourceFile"]).Distinct().Count()} files");
            }
            
            // Generate cross-reference analysis
            await GenerateCrossReferenceAnalysis(allChunkData, mergedDir);
        }
        
        private static async Task WriteMergedCSV(List<Dictionary<string, object>> data, string csvPath)
        {
            if (!data.Any()) return;
            
            var headers = data.First().Keys.ToList();
            var csv = new StringBuilder();
            
            // Write header
            csv.AppendLine(string.Join(",", headers.Select(h => $"\"{h}\"")));
            
            // Write data
            foreach (var row in data)
            {
                var values = headers.Select(h => row.ContainsKey(h) ? $"\"{row[h]}\"" : "\"\"");
                csv.AppendLine(string.Join(",", values));
            }
            
            await File.WriteAllTextAsync(csvPath, csv.ToString());
        }
        
        private static async Task GenerateCrossReferenceAnalysis(Dictionary<string, List<Dictionary<string, object>>> allChunkData, string mergedDir)
        {
            var analysis = new StringBuilder();
            analysis.AppendLine("CROSS-REFERENCE ANALYSIS");
            analysis.AppendLine("======================");
            analysis.AppendLine($"Generated: {DateTime.Now}");
            analysis.AppendLine();
            
            // Summary statistics
            foreach (var kvp in allChunkData)
            {
                var chunkType = kvp.Key;
                var data = kvp.Value;
                var fileCount = data.Select(d => d["SourceFile"]).Distinct().Count();
                
                analysis.AppendLine($"{chunkType}: {data.Count} entries across {fileCount} files");
            }
            
            analysis.AppendLine();
            analysis.AppendLine("UNKNOWN FIELD VALUE RANGES:");
            analysis.AppendLine("===========================");
            
            // Analyze unknown field ranges across all files
            if (allChunkData.ContainsKey("MSLK"))
            {
                AnalyzeUnknownFieldRanges(allChunkData["MSLK"], "MSLK", analysis);
            }
            
            if (allChunkData.ContainsKey("MPRL"))
            {
                AnalyzeUnknownFieldRanges(allChunkData["MPRL"], "MPRL", analysis);
            }
            
            await File.WriteAllTextAsync(Path.Combine(mergedDir, "cross_reference_analysis.txt"), analysis.ToString());
        }
        
        private static void AnalyzeUnknownFieldRanges(List<Dictionary<string, object>> data, string chunkType, StringBuilder analysis)
        {
            analysis.AppendLine($"\n{chunkType} Unknown Fields:");
            
            var unknownFields = data.First().Keys.Where(k => k.StartsWith("Unknown")).ToList();
            
            foreach (var field in unknownFields)
            {
                var values = data.Select(d => d[field]).Where(v => v != null).ToList();
                if (!values.Any()) continue;
                
                if (values.First() is int || values.First() is uint)
                {
                    var numericValues = values.Cast<object>().Select(v => Convert.ToInt64(v)).ToList();
                    var min = numericValues.Min();
                    var max = numericValues.Max();
                    var distinct = numericValues.Distinct().Count();
                    
                    analysis.AppendLine($"  {field}: Range [{min} to {max}], {distinct} distinct values, {values.Count} total");
                }
            }
        }
        
        private static async Task GenerateSummaryReport(List<BatchProcessingResult> results, string batchOutputDir)
        {
            var report = new StringBuilder();
            report.AppendLine("PM4 BATCH ANALYSIS SUMMARY REPORT");
            report.AppendLine("=================================");
            report.AppendLine($"Generated: {DateTime.Now}");
            report.AppendLine();
            
            var successful = results.Where(r => r.Success).ToList();
            var failed = results.Where(r => !r.Success).ToList();
            
            report.AppendLine($"Total files processed: {results.Count}");
            report.AppendLine($"Successful: {successful.Count} ({(double)successful.Count / results.Count * 100:F1}%)");
            report.AppendLine($"Failed: {failed.Count} ({(double)failed.Count / results.Count * 100:F1}%)");
            report.AppendLine();
            
            if (successful.Any())
            {
                report.AppendLine("SUCCESSFUL FILES SUMMARY:");
                report.AppendLine($"Total vertices (MSVT): {successful.Sum(r => r.VertexCount):N0}");
                report.AppendLine($"Total triangles (MSVI): {successful.Sum(r => r.TriangleCount):N0}");
                report.AppendLine($"Total MSCN vertices: {successful.Sum(r => r.MscnCount):N0}");
                report.AppendLine($"Total MSLK links: {successful.Sum(r => r.MslkCount):N0}");
                report.AppendLine($"Total MSUR surfaces: {successful.Sum(r => r.MsurCount):N0}");
                report.AppendLine($"Total MPRL placements: {successful.Sum(r => r.MprlCount):N0}");
                report.AppendLine();
            }
            
            if (failed.Any())
            {
                report.AppendLine("FAILED FILES:");
                foreach (var failure in failed)
                {
                    report.AppendLine($"  {failure.FileName}: {failure.ErrorMessage}");
                }
            }
            
            await File.WriteAllTextAsync(Path.Combine(batchOutputDir, "summary_report.txt"), report.ToString());
        }
    }
    
    internal class BatchProcessingResult
    {
        public string FileName { get; set; } = "";
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        public Dictionary<string, List<Dictionary<string, object>>>? ChunkData { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public int MscnCount { get; set; }
        public int MslkCount { get; set; }
        public int MsurCount { get; set; }
        public int MprlCount { get; set; }
    }
}
